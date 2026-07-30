r"""Run per-plot work in parallel, limited by MEMORY rather than by core count.

Every batch stage here processes plots that are completely independent of each
other, so they can run at the same time. What stops us simply using one worker
per core is size: these clouds differ by more than an order of magnitude
(18.9 M to 280.5 M points across the 14 SegmentedForests plots), and peak RSS is
roughly 60 bytes per point -- about **16 GB for the biggest plot on its own**.
Sixteen workers would exhaust a 64 GB machine; a fixed small number would have to
be chosen for the worst case and would then leave the cores idle on the small
plots.

So the scheduler admits work against a memory budget:

* estimate each plot's RSS from its point count (LAS/LAZ expose that in the
  header, so it costs nothing to read even for a 15 GB file);
* start with the biggest -- longest-job-first keeps the tail short;
* submit while the committed estimate plus the next job still fits the budget;
* when a job finishes, release its share and top the pool back up.

The effect is self-balancing: a 280 M-point plot runs nearly alone, while small
plots pack several deep.

**Processes, not threads.** The work is numpy and LAZ decompression inside this
interpreter, so threads would serialise on the GIL. (Stage 1 is the opposite case
-- it only waits on the external 3DFin binary -- which is one reason it stays
sequential and does not use this module.)
"""
from __future__ import annotations

import logging
import os
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

# MEASURED, not guessed: plot_09 (72.6 M points) peaks at 9.1 GB in Stage 2, i.e.
# ~134 bytes per point. laspy holds the whole source cloud, the export builds its
# own structured array alongside it, and then `plyfile.write()` does
# `data.astype(...).tobytes()` -- which materialises another full copy just to
# write it. Rounded up to 140 for headroom.
#
# An earlier 60 here (counting only the obvious arrays) let too many big plots run
# together and they died with MemoryError inside that write. Under-estimating costs
# a failed batch; over-estimating only costs some idle cores, so err high.
DEFAULT_BYTES_PER_POINT = 140
# Leave the rest of RAM for the OS, the file cache and whatever else is running.
DEFAULT_MEMORY_BUDGET_FRAC = 0.7
# Bytes of file per point, used only when the format has no cheap point count.
_FALLBACK_BYTES_PER_POINT_ON_DISK = 40


# --------------------------------------------------------------------------- #
# Sizing
# --------------------------------------------------------------------------- #

def estimate_points(path: Path) -> int:
    """Point count of a cloud, as cheaply as the format allows.

    LAS/LAZ keep it in the header, so this never touches the point data -- it is
    instant even on a 15 GB file. Other formats fall back to a guess from the file
    size, which only needs to be good enough to rank plots and size the budget.
    """
    path = Path(path)
    try:
        if path.suffix.lower() in (".las", ".laz"):
            import laspy
            with laspy.open(str(path)) as reader:
                return int(reader.header.point_count)
    except Exception:  # unreadable header: fall through to the size heuristic
        log.debug("could not read a point count from %s", path.name, exc_info=True)
    try:
        return int(path.stat().st_size / _FALLBACK_BYTES_PER_POINT_ON_DISK)
    except OSError:
        return 0


def total_memory_bytes() -> int:
    """Installed RAM. Falls back to a conservative 8 GB if it cannot be read."""
    try:                                   # Windows
        import ctypes

        class _Status(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        st = _Status()
        st.dwLength = ctypes.sizeof(_Status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return int(st.ullTotalPhys)
    except Exception:
        pass
    try:                                   # Linux / macOS
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        return 8 * 1024 ** 3


# --------------------------------------------------------------------------- #
# Jobs and results
# --------------------------------------------------------------------------- #

@dataclass
class Job:
    """One unit of per-plot work.

    ``fn`` and every entry of ``args``/``kwargs`` must be picklable, because they
    cross into a worker process: module-level functions and plain
    paths/dicts/ints, which is what the engines already pass around.
    """
    name: str                       # the plot name, for logs and the tally
    fn: Callable
    args: Tuple = ()
    kwargs: Dict = field(default_factory=dict)
    n_points: int = 0               # 0 = unknown; sizes its share of the budget


@dataclass
class Result:
    name: str
    ok: bool
    logs: List[Tuple[int, str]] = field(default_factory=list)
    error: Optional[str] = None
    value: object = None


class _CollectHandler(logging.Handler):
    """Keeps formatted records so a worker can hand them back to the parent."""

    def __init__(self, sink: List[Tuple[int, str]]):
        super().__init__()
        self.sink = sink

    def emit(self, record):
        try:
            self.sink.append((record.levelno, record.getMessage()))
        except Exception:
            pass


def _init_worker() -> None:
    """Stop each worker from starting its own thread pool.

    numpy's BLAS and OpenMP size themselves to the whole machine, so N workers
    would each spawn N threads and fight over the same cores. One thread apiece is
    right here: the parallelism is across plots.
    """
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")


def _run_job(job: Job) -> Result:
    """Worker side: run one job, capturing its log lines and any traceback.

    Exceptions are returned rather than raised so one bad plot cannot take the
    batch down -- the parent decides what to do, exactly as the sequential loop
    did.
    """
    sink: List[Tuple[int, str]] = []
    handler = _CollectHandler(sink)
    root = logging.getLogger("pipeline")
    root.addHandler(handler)
    previous = root.level
    root.setLevel(logging.INFO)
    try:
        value = job.fn(*job.args, **job.kwargs)
        return Result(job.name, True, sink, None, value)
    except Exception:
        return Result(job.name, False, sink, traceback.format_exc(), None)
    finally:
        root.removeHandler(handler)
        root.setLevel(previous)


# --------------------------------------------------------------------------- #
# The scheduler
# --------------------------------------------------------------------------- #

def run_jobs(
    jobs: Sequence[Job],
    workers: Optional[int] = None,
    memory_budget_frac: float = DEFAULT_MEMORY_BUDGET_FRAC,
    bytes_per_point: int = DEFAULT_BYTES_PER_POINT,
    continue_on_error: bool = True,
) -> List[Result]:
    """Run ``jobs``, several at a time, without exceeding the memory budget.

    ``workers=1`` runs everything inline in this process -- no pool, no pickling,
    identical to the old sequential loop. That is the escape hatch when something
    looks wrong.

    ``workers=None`` lets the memory budget decide, capped by the CPU count.
    Any other integer is a hard ceiling on concurrency, still subject to the
    budget.

    Results come back in completion order; callers that want a stable report
    should sort by name.
    """
    jobs = list(jobs)
    if not jobs:
        return []

    if workers == 1 or len(jobs) == 1:
        return [_run_inline(job) for job in jobs]

    cpu_cap = workers or (os.cpu_count() or 2)
    budget = int(total_memory_bytes() * memory_budget_frac)

    def cost(job: Job) -> int:
        # A job of unknown size still has to reserve something, or unknown-size
        # jobs would all be admitted at once.
        return max(int(job.n_points) * bytes_per_point, 256 * 1024 ** 2)

    # Biggest first: the long pole starts early instead of running alone at the end.
    pending = sorted(jobs, key=cost, reverse=True)
    log.info("Running %d job(s) on up to %d worker(s), memory budget %.1f GB "
             "(largest job ~%.1f GB)",
             len(pending), cpu_cap, budget / 1024 ** 3, cost(pending[0]) / 1024 ** 3)

    results: List[Result] = []
    running: Dict[object, Job] = {}
    committed = 0
    stop = False

    with ProcessPoolExecutor(max_workers=cpu_cap, initializer=_init_worker) as pool:
        while (pending and not stop) or running:
            while pending and not stop and len(running) < cpu_cap:
                nxt = pending[0]
                # Admit if it fits -- or unconditionally when nothing is running,
                # which is what lets a job larger than the whole budget proceed
                # (alone) instead of deadlocking.
                if running and committed + cost(nxt) > budget:
                    break
                pending.pop(0)
                fut = pool.submit(_run_job, nxt)
                running[fut] = nxt
                committed += cost(nxt)

            if not running:
                break

            done, _ = wait(list(running), return_when=FIRST_COMPLETED)
            for fut in done:
                job = running.pop(fut)
                committed -= cost(job)
                try:
                    result = fut.result()
                except Exception:                     # worker died outright
                    result = Result(job.name, False, [], traceback.format_exc(), None)
                results.append(result)
                if not result.ok and not continue_on_error:
                    stop = True

    if stop and pending:
        log.error("continue_on_error=False — %d job(s) not started.", len(pending))
    return results


def _run_inline(job: Job) -> Result:
    """Run a job in this process, letting its log records flow out normally."""
    try:
        return Result(job.name, True, [], None, job.fn(*job.args, **job.kwargs))
    except Exception:
        return Result(job.name, False, [], traceback.format_exc(), None)


def replay(result: Result, logger: logging.Logger) -> None:
    """Re-emit a worker's captured log lines through the parent's logger.

    Worker processes cannot reach Hydra's file handler, so their output would
    otherwise vanish. Replaying keeps ``outputs/<date>/<time>/*.log`` showing the
    same per-plot detail as a sequential run (interleaved differently, since the
    plots genuinely overlap).
    """
    for levelno, message in result.logs:
        logger.log(levelno, message)
