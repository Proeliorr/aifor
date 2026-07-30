r"""Format conversion between LAZ/LAS and PLY — the last step before training.

Why this exists
---------------
Stage 2 exports **LAZ** by default, but ForAINet reads **PLY** only. Something has to
expand one into the other, and *where* it happens decides how much data crosses a
network.

A binary PLY is a raw memory dump — measured at exactly 37.00 bytes/point for the six
ForAINet fields (``f8*3 + f8 + u1 + u4``), so the 14 SegmentedForests plots come to
**29.2 GB** from 4.4 GB of source. The same points as LAZ are a fraction of that. When
training runs on a rented GPU the clouds have to be uploaded, so the export travels
compressed and this module expands it again *inside the container*, in seconds, on a
disk you are already paying for.

What it does NOT do
-------------------
This is a **pure format conversion**. No centering, no class remapping, no tree-id
zeroing — all of that happened in Stage 2, whose output already carries centered
coordinates (LAS header offset 0), the unified ``semantic_seg`` and the zeroed
``treeID``. Everything that could change what the model learns stays in the pipeline,
where it is tested; this module just moves bytes between containers.

Two consequences follow, and both matter:

* **The file stem is preserved exactly.** ``plot_11_val.laz`` becomes
  ``plot_11_val.ply``. ForAINet reads the train/val/test split from the FILE NAME
  (``name[-7:-4] == "val"``, then ``name[-8:-4] == "test"``), so renaming a file — or
  adding a prefix — silently moves a plot between sets.
* **No offsets file is needed** to convert or to train. It is needed only to put
  predictions back into real-world coordinates, which happens later and elsewhere
  (``forainet_prep.py forainet_prep.mode=restore``).

Running it
----------
Inside a container, with no Hydra installed — this module deliberately imports nothing
beyond ``laspy``, ``lazrs``, ``numpy``, ``plyfile``, ``yaml`` and the standard library::

    python -m pipeline.convert --to ply /data/export /workspace/.../treeinsfused/raw/SegmentedForests

As a pipeline stage, from the repo root::

    mamba run -n aifor python convert.py                    # reads the `convert:` section
    mamba run -n aifor python convert.py convert.dry_run=true

Fidelity
--------
``semantic_seg`` (u1) and ``treeID`` (u4) survive **bit-identically** — they travel as
LAS extra-bytes dimensions. ``intensity`` is exact for clouds that came from LAS/LAZ
(the value originated as u16). Coordinates are quantised to the LAS scale, so a
round trip is accurate to **half a scale step** — 5e-8 m at the 1e-07 scale this
dataset uses, against a model that voxelises at 0.2 m.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from pipeline.forainet_prep import (
    _FORAINET_DTYPE,
    _read_cloud,
    _write_las,
    _write_ply,
)
from pipeline.parallel import Job, estimate_points, replay, run_jobs

log = logging.getLogger(__name__)

# What we can read and write. `.ply` on one side, a LAS container on the other.
_PLY = ".ply"
_LAS_SUFFIXES = (".las", ".laz")
SUPPORTED = (_PLY,) + _LAS_SUFFIXES

DEFAULT_PATTERNS = ["*.laz", "*.las", "*.ply"]


def _target_suffix(to: str) -> str:
    """Normalise ``to`` ('ply' / '.ply' / 'LAZ') into a suffix like ``.ply``."""
    suffix = "." + str(to).lower().lstrip(".")
    if suffix not in SUPPORTED:
        raise ValueError(f"--to must be one of {[s.lstrip('.') for s in SUPPORTED]}, got {to!r}")
    return suffix


def convert_one(src: Path, dst: Path, meta_override: Optional[Dict] = None) -> int:
    """Convert a single cloud, preserving every field. Returns the point count.

    ``meta_override`` supplies the LAS header details (scales, CRS) when writing a LAS
    container from a PLY, which carries none of its own — normally the ``<plot>_offsets.yml``
    written by Stage 2. Without it the writer falls back to millimetre scales and no CRS.
    """
    src, dst = Path(src), Path(dst)
    if src.suffix.lower() not in SUPPORTED:
        raise ValueError(f"unsupported input format {src.suffix!r}: {src.name}")
    if dst.suffix.lower() not in SUPPORTED:
        raise ValueError(f"unsupported output format {dst.suffix!r}: {dst.name}")

    xyz, intensity, semantic, tree_id, meta = _read_cloud(src)
    n_points = len(xyz)

    # Rebuilt in ForAINet's exact vertex layout rather than copied through, so a PLY
    # written here is byte-comparable with one Stage 2 wrote directly.
    array = np.empty(n_points, dtype=_FORAINET_DTYPE)
    array["x"] = xyz[:, 0]
    array["y"] = xyz[:, 1]
    array["z"] = xyz[:, 2]
    array["intensity"] = np.asarray(intensity, dtype=np.float64)
    array["semantic_seg"] = semantic
    array["treeID"] = np.asarray(tree_id, dtype=np.uint32)

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.suffix.lower() == _PLY:
        _write_ply(array, dst)
    else:
        # A PLY source records no scales or CRS, so an override is the only way to keep
        # them; otherwise _write_las falls back to 0.001 (mm) and writes no CRS.
        _write_las(array, dst, meta_override or meta)

    log.info("[%s] %s points converted -> %s", src.stem, f"{n_points:,}", dst.name)
    return n_points


def _offsets_for(stem: str, offsets_dir: Optional[Path]) -> Optional[Dict]:
    """Load ``<plot>_offsets.yml`` for a file stem, so LAS output keeps its scales/CRS.

    The stem may carry a split suffix (``plot_11_val``) while the offsets file is keyed
    by the plain plot name, so both are tried. Missing metadata is not an error — it
    only costs the source scales.
    """
    if offsets_dir is None:
        return None
    import yaml  # local: only needed on this path, keeps the import surface honest

    candidates = [stem]
    for split in ("train", "val", "test"):
        if stem.endswith(f"_{split}"):
            candidates.append(stem[: -len(split) - 1])
    for name in candidates:
        path = Path(offsets_dir) / f"{name}_offsets.yml"
        if path.is_file():
            with path.open() as fh:
                return yaml.safe_load(fh)
    return None


def _collect(input_dir: Path, patterns: List[str], to_suffix: str) -> List[Path]:
    """Match ``patterns`` in ``input_dir`` and one level down, minus the target format.

    Files already in the target format are excluded rather than "converted" onto
    themselves — running the tool twice on the same folder is a no-op, not a corruption.
    """
    found = set()
    for pattern in patterns:
        found.update(p.resolve() for p in input_dir.glob(pattern) if p.is_file())
        found.update(p.resolve() for p in input_dir.glob(f"*/{pattern}") if p.is_file())
    return sorted(p for p in found if p.suffix.lower() != to_suffix)


def convert_clouds(
    input_dir: str = ".",
    output_dir: Optional[str] = None,
    to: str = "ply",
    patterns: Optional[List[str]] = None,
    offsets_dir: Optional[str] = None,
    plots: Optional[List[str]] = None,
    overwrite: bool = True,
    dry_run: bool = False,
    continue_on_error: bool = True,
    workers: Optional[int] = None,
    memory_budget_frac: float = 0.7,
    bytes_per_point: int = 140,
    log_summary: bool = True,
) -> Dict[str, List[str]]:
    """Convert every matched cloud in ``input_dir`` into ``to`` format.

    Parameters
    ----------
    input_dir, output_dir:
        Source folder (matched flat and one level down) and destination; ``None``
        writes beside the inputs. Created if missing.
    to:
        Target format — ``"ply"`` (default), ``"laz"`` or ``"las"``.
    patterns:
        Globs to match (default ``["*.laz", "*.las", "*.ply"]``). Files already in the
        target format are skipped.
    offsets_dir:
        Folder of ``<plot>_offsets.yml``. Only consulted when writing LAS/LAZ **from a
        PLY**, to recover the source scales and CRS a PLY cannot carry.
    plots:
        Restrict to these file stems (split suffix included or not).
    overwrite, dry_run, continue_on_error, workers, memory_budget_frac, bytes_per_point:
        As in the other batch stages; see ``pipeline/parallel.py``.
    log_summary:
        ``False`` drops the run-level framing when called as part of a bigger run.

    Returns ``{"succeeded": [...], "failed": [...], "skipped": [...]}`` of file stems.
    """
    to_suffix = _target_suffix(to)
    in_dir = Path(input_dir)
    if not in_dir.is_dir():
        raise NotADirectoryError(f"input_dir does not exist: {in_dir}")
    out_dir = Path(output_dir) if output_dir else in_dir
    patterns = list(patterns) if patterns else list(DEFAULT_PATTERNS)
    off_dir = Path(offsets_dir) if offsets_dir else None

    succeeded: List[str] = []
    failed: List[str] = []
    skipped: List[str] = []

    src_files = _collect(in_dir, patterns, to_suffix)
    if plots:
        wanted = {str(p) for p in plots}
        src_files = [p for p in src_files if p.stem in wanted
                     or any(p.stem.startswith(f"{w}_") or p.stem == w for w in wanted)]
    if not src_files:
        log.warning("No clouds to convert to %s in %s — nothing to do.", to_suffix, in_dir)
        return {"succeeded": succeeded, "failed": failed, "skipped": skipped}
    if log_summary:
        log.info("Converting %d cloud(s) to %s: %s -> %s",
                 len(src_files), to_suffix, in_dir, out_dir)

    jobs: List[Job] = []
    for src in src_files:
        dst = out_dir / f"{src.stem}{to_suffix}"     # stem preserved: the split lives in it
        if not overwrite and dst.exists():
            log.info("[%s] %s already exists and overwrite=False — skipping", src.stem, dst.name)
            skipped.append(src.stem)
            continue
        if dry_run:
            log.info("[%s] DRY RUN: would convert %s -> %s", src.stem, src.name, dst)
            succeeded.append(src.stem)
            continue
        meta_override = (_offsets_for(src.stem, off_dir)
                         if (src.suffix.lower() == _PLY and to_suffix != _PLY) else None)
        jobs.append(Job(
            name=src.stem,
            fn=convert_one,
            args=(src, dst, meta_override),
            n_points=estimate_points(src),           # header-only; sizes its RAM share
        ))

    for result in run_jobs(jobs, workers=workers, memory_budget_frac=memory_budget_frac,
                           bytes_per_point=bytes_per_point,
                           continue_on_error=continue_on_error):
        replay(result, log)
        if result.ok:
            succeeded.append(result.name)
        else:
            log.error("[%s] FAILED\n%s", result.name, result.error or "")
            failed.append(result.name)
    succeeded.sort()
    failed.sort()

    if log_summary:
        log.info("=" * 60)
        if dry_run:
            log.info("DRY RUN — would convert %d file(s); %d skipped.",
                     len(succeeded), len(skipped))
        else:
            log.info("Converted %d file(s): %d ok, %d failed, %d skipped.",
                     len(succeeded) + len(failed), len(succeeded), len(failed), len(skipped))
        if failed:
            log.warning("Failed: %s", ", ".join(failed))
        if skipped:
            log.warning("Skipped: %s", ", ".join(skipped))

    return {"succeeded": succeeded, "failed": failed, "skipped": skipped}


def main(argv: Optional[List[str]] = None) -> int:
    """Standalone CLI — this is what runs inside the training container.

    Deliberately argparse rather than Hydra: the container needs ``laspy``, ``lazrs``,
    ``numpy``, ``plyfile`` and ``PyYAML`` and nothing else.
    """
    parser = argparse.ArgumentParser(
        prog="python -m pipeline.convert",
        description="Convert point clouds between LAZ/LAS and PLY, preserving every field.",
        epilog=(
            "The file stem is preserved, which matters: ForAINet reads the train/val/test "
            "split from the file NAME, so plot_11_val.laz must become plot_11_val.ply.\n\n"
            "Typical container use:\n"
            "  python -m pipeline.convert --to ply /data/export "
            "/workspace/PointCloudSegmentation/data_set1_5classes/treeinsfused/raw/SegmentedForests"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input_dir", help="folder of clouds to convert (also searched one level down)")
    parser.add_argument("output_dir", nargs="?", default=None,
                        help="destination folder (default: beside the inputs)")
    parser.add_argument("--to", default="ply", choices=["ply", "laz", "las"],
                        help="target format (default: ply)")
    parser.add_argument("--offsets-dir", default=None,
                        help="folder of <plot>_offsets.yml; only used when writing LAS/LAZ "
                             "from a PLY, to recover the source scales and CRS")
    parser.add_argument("--plots", nargs="+", default=None, help="restrict to these file stems")
    parser.add_argument("--no-overwrite", action="store_true", help="skip files that already exist")
    parser.add_argument("--dry-run", action="store_true", help="list what would be converted")
    parser.add_argument("--workers", type=int, default=None,
                        help="parallel workers (default: as many as RAM allows; 1 = inline)")
    parser.add_argument("-q", "--quiet", action="store_true", help="warnings and errors only")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    result = convert_clouds(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        to=args.to,
        offsets_dir=args.offsets_dir,
        plots=args.plots,
        overwrite=not args.no_overwrite,
        dry_run=args.dry_run,
        workers=args.workers,
    )
    return 1 if result["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
