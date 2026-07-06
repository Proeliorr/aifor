r"""Batch 3DFin CLI runner for the SegmentedForests plots.

For every point cloud (``*.laz``) in an input folder this finds the matching 3DFin
parameter file (``<stem>.ini``) and runs the 3DFin v0.6.0 command-line interface::

    3DFin cli <input.laz> <output_directory> <params.ini> [--normalize] [--denoise] [--export_txt]

Two quirks of the 3DFin 0.6.0 CLI (see ``three_d_fin/processing/__init__.py`` and
``configuration.py``) shape this runner:

* 3DFin pydantic-validates the *whole* ``.ini`` on load, including ``[misc] output_dir``,
  which is a ``DirectoryPath`` that **must already exist**. The dataset's ``.ini`` files
  hardcode a Windows path (``F:\classification``) that does not exist on Linux, so loading
  raises a ``ValidationError``. We therefore write a **patched copy** of each ``.ini`` whose
  ``output_dir`` points at the (existing) per-plot output folder. The originals are never
  modified.
* The CLI keeps only the ``basic``/``advanced``/``expert`` sections from the ``.ini`` and
  rebuilds ``[misc]`` from the command line, where ``is_normalized = not --normalize``,
  ``is_noisy = --denoise`` and ``export_txt = --export_txt``. To reproduce each plot's recorded
  settings we **derive the flags from its ``.ini`` ``[misc]`` section** by default (e.g.
  ``is_normalized=False`` -> pass ``--normalize``). The ``normalize``/``denoise``/``export_txt``
  arguments override this when set to ``True``/``False`` (``None`` = derive from the ``.ini``).

It is driven by ``main_pipeline.py``, which reads the ``threedfin:`` section of the shared
``conf/config.yaml`` and maps its keys onto ``run_pipeline``'s parameters.
"""
from __future__ import annotations

import configparser
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

log = logging.getLogger(__name__)


def _read_misc_flags(ini_path: Path) -> Tuple[bool, bool, bool]:
    """Return ``(is_normalized, is_noisy, export_txt)`` from a 3DFin ``.ini`` ``[misc]``.

    A missing ``[misc]`` section or missing keys fall back to the 3DFin defaults (``False``).
    """
    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str
    parser.read(ini_path)
    if not parser.has_section("misc"):
        return (False, False, False)
    misc = parser["misc"]
    return (
        misc.getboolean("is_normalized", fallback=False),
        misc.getboolean("is_noisy", fallback=False),
        misc.getboolean("export_txt", fallback=False),
    )


def _write_patched_ini(ini_path: Path, dest: Path, output_dir: Path) -> None:
    """Copy ``ini_path`` to ``dest`` with ``[misc] output_dir`` set to ``output_dir``.

    3DFin validates ``output_dir`` as an *existing* directory on load, so ``output_dir`` must
    already exist when 3DFin runs. Its value is otherwise unused: the CLI rebuilds the misc
    section from its own positional ``output_directory`` argument.
    """
    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str
    parser.read(ini_path)
    if not parser.has_section("misc"):
        parser.add_section("misc")
    parser["misc"]["output_dir"] = str(output_dir)
    with dest.open("w") as fh:
        parser.write(fh)


def _prune_plot_output(plot_out: Path, keep: set) -> List[str]:
    """Delete every file directly in ``plot_out`` whose name is not in ``keep``.

    3DFin writes ~10 flat files per plot but only ``<stem>_tree_ID_dist_axes.las`` is needed
    here. A whitelist-keep is future-proof: it removes every current/future 3DFin product (and
    the disposable patched ``.ini``) while preserving exactly the files we want. Subdirectories
    are left untouched, and each removal is guarded so one failure does not abort the batch.

    Returns the sorted list of removed file names (for logging).
    """
    removed: List[str] = []
    for entry in plot_out.iterdir():
        if not entry.is_file() or entry.name in keep:
            continue
        try:
            entry.unlink()
            removed.append(entry.name)
        except OSError as exc:
            log.warning("[%s] could not remove %s: %s", plot_out.name, entry.name, exc)
    return sorted(removed)


def _build_command(
    threedfin_bin: str,
    laz: Path,
    plot_out: Path,
    ini: Path,
    normalize: bool,
    denoise: bool,
    export_txt: bool,
) -> List[str]:
    """Assemble the ``3DFin cli ...`` argument vector for a single plot."""
    cmd = [threedfin_bin, "cli", str(laz), str(plot_out), str(ini)]
    if normalize:
        cmd.append("--normalize")
    if denoise:
        cmd.append("--denoise")
    if export_txt:
        cmd.append("--export_txt")
    return cmd


def run_pipeline(
    pointclouds_dir: str = "/home/jakub/Projects/04_AI_in_Forestry/SegmentedForests/pointclouds",
    ini_dir: str = "/home/jakub/Projects/04_AI_in_Forestry/SegmentedForests/3DFin_settings",
    output_dir: str = "/home/jakub/Projects/04_AI_in_Forestry/SegmentedForests/3DFin_output",
    threedfin_bin: str = "/home/jakub/miniforge3/envs/aifor/bin/3DFin",
    pattern: str = "*.laz",
    normalize: Optional[bool] = None,
    denoise: Optional[bool] = None,
    export_txt: Optional[bool] = None,
    plots: Optional[List[str]] = None,
    prune_outputs: bool = True,
    dry_run: bool = False,
    continue_on_error: bool = True,
) -> None:
    """Run the 3DFin CLI for every point cloud matched in ``pointclouds_dir``.

    Each ``<stem>.laz`` is paired with ``<ini_dir>/<stem>.ini`` and its results are written to
    ``<output_dir>/<stem>/`` (one subfolder per plot), alongside a patched ``3dfin_params.ini``
    and the captured ``3dfin_log.txt``. Plots whose ``.ini`` is missing are skipped with a warning.

    Parameters
    ----------
    pointclouds_dir, ini_dir, output_dir:
        Input point-cloud folder, 3DFin ``.ini`` folder, and output root.
    threedfin_bin:
        Path to the 3DFin executable (defaults to the ``aifor`` env binary).
    pattern:
        Glob used to select point clouds inside ``pointclouds_dir`` (default ``*.laz``).
    normalize, denoise, export_txt:
        Tri-state overrides for the 3DFin CLI flags. ``None`` (default) derives the flag from
        each plot's ``.ini`` ``[misc]`` section; ``True``/``False`` forces it for every plot.
        (``normalize=True`` builds a DTM and computes normalized heights; the SegmentedForests
        clouds are not height-normalized, so deriving from the ``.ini`` enables it.)
    plots:
        Optional list of plot stems (e.g. ``["plot_01", "plot_05"]``) to restrict the run.
        ``None`` processes every matched point cloud.
    prune_outputs:
        If ``True`` (default), after a successful run delete every 3DFin output for that plot
        except ``<stem>_tree_ID_dist_axes.las`` and the captured ``3dfin_log.txt`` — the only
        file this project needs plus its run log. ``False`` keeps the full 3DFin output set.
    dry_run:
        If ``True``, log the command for each plot without writing files or executing 3DFin.
    continue_on_error:
        If ``True`` (default), keep going after a plot fails; otherwise stop at the first
        non-zero exit.
    """
    pc_dir = Path(pointclouds_dir)
    ini_path = Path(ini_dir)
    out_dir = Path(output_dir)

    # --- resolve the 3DFin executable ------------------------------------
    resolved_bin = shutil.which(threedfin_bin)
    if resolved_bin is None and Path(threedfin_bin).is_file():
        resolved_bin = threedfin_bin
    if not dry_run and resolved_bin is None:
        raise FileNotFoundError(
            f"3DFin executable not found: {threedfin_bin!r}. Point `threedfin_bin` at "
            "the aifor env binary, e.g. /home/jakub/miniforge3/envs/aifor/bin/3DFin"
        )
    bin_path = resolved_bin or threedfin_bin

    if not pc_dir.is_dir():
        raise NotADirectoryError(f"pointclouds_dir does not exist: {pc_dir}")
    if not ini_path.is_dir():
        raise NotADirectoryError(f"ini_dir does not exist: {ini_path}")

    # --- collect the input point clouds ----------------------------------
    laz_files = sorted(pc_dir.glob(pattern))
    if plots:
        wanted = set(plots)
        selected = [p for p in laz_files if p.stem in wanted]
        for stem in sorted(wanted - {p.stem for p in selected}):
            log.warning("Requested plot %r has no %s in %s", stem, pattern, pc_dir)
        laz_files = selected

    if not laz_files:
        log.warning("No point clouds matched %r in %s — nothing to do.", pattern, pc_dir)
        return

    log.info("Found %d point cloud(s) to process in %s", len(laz_files), pc_dir)

    planned: List[str] = []
    succeeded: List[str] = []
    failed: List[str] = []
    skipped: List[str] = []

    for laz in laz_files:
        ini = ini_path / f"{laz.stem}.ini"
        if not ini.is_file():
            log.warning("[%s] no matching .ini at %s — skipping", laz.stem, ini)
            skipped.append(laz.stem)
            continue

        # Derive the 3DFin flags from the .ini's [misc], honouring any explicit override.
        ini_normalized, ini_noisy, ini_txt = _read_misc_flags(ini)
        do_normalize = (not ini_normalized) if normalize is None else normalize
        do_denoise = ini_noisy if denoise is None else denoise
        do_export_txt = ini_txt if export_txt is None else export_txt

        plot_out = out_dir / laz.stem
        patched_ini = plot_out / "3dfin_params.ini"
        cmd = _build_command(bin_path, laz, plot_out, patched_ini, do_normalize, do_denoise, do_export_txt)

        if dry_run:
            log.info("[%s] DRY RUN: %s", laz.stem, " ".join(cmd))
            planned.append(laz.stem)
            continue

        plot_out.mkdir(parents=True, exist_ok=True)
        # Patch output_dir (the .ini's hardcoded Windows path fails 3DFin's path validation).
        _write_patched_ini(ini, patched_ini, plot_out)

        log_file = plot_out / "3dfin_log.txt"
        log.info(
            "[%s] running 3DFin (normalize=%s denoise=%s export_txt=%s) -> %s",
            laz.stem, do_normalize, do_denoise, do_export_txt, plot_out,
        )
        # Force UTF-8 in the child: 3DFin's progress bar prints Unicode block chars, which
        # crash with a cp1252 UnicodeEncodeError when stdout is a redirected (non-console)
        # stream on Windows — i.e. exactly the headless/Docker case. PYTHONUTF8 makes the
        # child encode its stdout as UTF-8, matching the UTF-8 log file below.
        child_env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
        with open(log_file, "w", encoding="utf-8") as fh:
            fh.write("$ " + " ".join(cmd) + "\n\n")
            fh.flush()
            result = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=child_env)

        if result.returncode == 0:
            log.info("[%s] done (log: %s)", laz.stem, log_file)
            succeeded.append(laz.stem)
            if prune_outputs:
                wanted_las = f"{laz.stem}_tree_ID_dist_axes.las"
                if not (plot_out / wanted_las).is_file():
                    log.warning("[%s] expected %s not found after run — nothing kept but the log",
                                laz.stem, wanted_las)
                removed = _prune_plot_output(plot_out, {wanted_las, log_file.name})
                if removed:
                    log.info("[%s] pruned %d file(s): %s", laz.stem, len(removed), ", ".join(removed))
        else:
            log.error("[%s] FAILED (exit %d, log: %s)", laz.stem, result.returncode, log_file)
            failed.append(laz.stem)
            if not continue_on_error:
                log.error("continue_on_error=False — stopping after first failure.")
                break

    # --- summary ---------------------------------------------------------
    log.info("=" * 60)
    if dry_run:
        log.info("DRY RUN — would process %d plot(s); %d skipped (no .ini).",
                 len(planned), len(skipped))
    else:
        log.info("Processed %d plot(s): %d ok, %d failed, %d skipped.",
                 len(succeeded) + len(failed), len(succeeded), len(failed), len(skipped))
        if failed:
            log.warning("Failed: %s", ", ".join(failed))
    if skipped:
        log.warning("Skipped (no .ini): %s", ", ".join(skipped))
