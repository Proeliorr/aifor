r"""Standalone tool: split 3DFin-segmented plot clouds into one numpy file per tree.

.. note::
   This tool is **not part of the main pipeline**. The project's main approach is
   semantic segmentation at whole-plot level, so per-tree data is not needed there.
   It is kept in ``misc/`` for tree-level approaches that may want it later.

The 3DFin batch step (``pipeline/threedfin.py``) leaves one instance-segmented cloud
per plot at ``<input_dir>/<plot>/<plot>_tree_ID_dist_axes.las``, where every point
carries a ``tree_ID`` label. Those labels are sparse and non-sequential (e.g. plot_01
has 119 unique IDs spanning 0..630619; **id 0 is the first tree** — 0-based numbering,
not background), so this tool:

1. sorts all points by ``tree_ID`` ascending (stable sort),
2. assigns each distinct ``tree_ID`` a new ordinal ``2tree_ID`` = 1, 2, 3, ...
   (``tree_ID=0`` -> ``2tree_ID=1``),
3. splits the cloud on ``2tree_ID`` and saves each tree as
   ``<plot>_<NNN>.npy`` (zero-padded, e.g. ``plot_01_001.npy``) **next to the input
   file** in the plot's own folder.

Which columns land in the arrays is configurable (``columns``; default x, y, z, Class).
Besides any LAS dimension, two special names are understood: ``tree_ID`` (the original
label) and ``2tree_ID`` (the computed ordinal).

Run from the ``assignment/`` folder inside the ``aifor`` mamba environment::

    mamba run -n aifor python misc/tree_splitter.py                                # all plots
    mamba run -n aifor python misc/tree_splitter.py tree_splitter.dry_run=true    # print plan only
    mamba run -n aifor python misc/tree_splitter.py tree_splitter.plots=[plot_01] # a single plot

The config lives in ``misc/conf/config.yaml`` under the ``tree_splitter:`` section.
Override any key with that prefix, e.g.
``tree_splitter.columns=[x,y,z,Class,2tree_ID] tree_splitter.digits=4``.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

# This tool owns its config (misc/conf/config.yaml) but shares the pipeline's
# parallel scheduler rather than carrying a second copy of it. Python puts THIS
# file's folder on sys.path, not the repo root, so point at the parent explicitly
# — and do it before the import below.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.parallel import (  # noqa: E402  (needs the sys.path line above)
    DEFAULT_BYTES_PER_POINT,
    DEFAULT_MEMORY_BUDGET_FRAC,
    Job,
    estimate_points,
    replay,
    run_jobs,
)

import hydra
import laspy
import numpy as np
from hydra_zen import zen

log = logging.getLogger(__name__)

# Suffix stripped from the input file name to recover the plot name
# (plot_01_tree_ID_dist_axes.las -> plot_01).
_INPUT_SUFFIX = "_tree_ID_dist_axes"


def _plot_name(las_path: Path) -> str:
    """Derive the plot name from the input file, falling back to its directory name."""
    stem = las_path.stem
    if stem.endswith(_INPUT_SUFFIX):
        return stem[: -len(_INPUT_SUFFIX)]
    return las_path.parent.name


def _resolve_columns(las: laspy.LasData, columns: List[str]) -> List[str]:
    """Return the requested columns that exist on `las`, warning about unknown names.

    ``2tree_ID`` is always valid (computed here, not read from the file).
    """
    selected, missing = [], []
    for c in columns:
        if c == "2tree_ID" or hasattr(las, c):
            selected.append(c)
        else:
            missing.append(c)
    if missing:
        log.warning(
            "Skipping column(s) not in this file: %s (available: %s)",
            ", ".join(missing), ", ".join(las.point_format.dimension_names),
        )
    return selected


def _split_one_plot(las_path: Path, plot: str, columns: List[str], digits: int) -> int:
    """Split a single plot cloud into per-tree ``.npy`` files. Returns the tree count.

    All laspy accessors are materialised with ``np.asarray``: laspy exposes x/y/z as
    ScaledArrayView and bit fields as SubFieldView, which numpy/pandas do not treat as
    plain arrays.
    """
    las = laspy.read(str(las_path))
    tree_id = np.asarray(las.tree_ID)

    # Stable sort by the original tree_ID (ascending).
    order = np.argsort(tree_id, kind="stable")
    sorted_ids = tree_id[order]

    # One entry per distinct tree, in ascending tree_ID order. The new ordinal
    # 2tree_ID is simply 1..K in that order (tree_ID=0 -> 2tree_ID=1).
    uniques, counts = np.unique(sorted_ids, return_counts=True)
    new_ids = np.repeat(np.arange(1, len(uniques) + 1), counts)

    # Assemble the output matrix in the sorted order, one column per request.
    selected = _resolve_columns(las, columns)
    if not selected:
        raise ValueError(f"none of the requested columns {columns} exist in {las_path.name}")
    cols = []
    for c in selected:
        if c == "2tree_ID":
            cols.append(new_ids)
        else:
            cols.append(np.asarray(getattr(las, c))[order])
    matrix = np.column_stack(cols)

    # Split into per-tree chunks along the group boundaries and save each one.
    chunks = np.split(matrix, np.cumsum(counts)[:-1])
    for k, chunk in enumerate(chunks, start=1):
        np.save(las_path.parent / f"{plot}_{k:0{digits}d}.npy", chunk)

    log.info(
        "[%s] %s points -> %d trees (columns: %s), saved %s_%s.npy ... %s_%s.npy in %s",
        plot, f"{len(tree_id):,}", len(uniques), ", ".join(selected),
        plot, f"{1:0{digits}d}", plot, f"{len(uniques):0{digits}d}", las_path.parent,
    )
    return len(uniques)


def split_trees(
    input_dir: str = r"D:\Podyplomowe\04_AI_Intro\assignment\SegmentedForests\3DFin_output",
    pattern: str = "*_tree_ID_dist_axes.las",
    columns: Optional[List[str]] = None,
    plots: Optional[List[str]] = None,
    digits: int = 3,
    overwrite: bool = True,
    dry_run: bool = False,
    continue_on_error: bool = True,
    workers: Optional[int] = None,
    memory_budget_frac: float = DEFAULT_MEMORY_BUDGET_FRAC,
    bytes_per_point: int = DEFAULT_BYTES_PER_POINT,
) -> None:
    """Split every plot cloud under ``input_dir`` into per-tree ``.npy`` files.

    Each subdirectory of ``input_dir`` is treated as one plot; its input cloud is the
    file matching ``pattern`` inside it. The per-tree arrays are written back into the
    same plot folder as ``<plot>_001.npy``, ``<plot>_002.npy``, ...

    Parameters
    ----------
    input_dir:
        Root folder holding one subfolder per plot (the 3DFin output tree).
    pattern:
        Glob for the input cloud inside each plot folder.
    columns:
        Columns to store in each array (default ``["x", "y", "z", "Class"]``, where
        ``Class`` is the ground-truth semantic label used to train the downstream model).
        Any LAS dimension plus the special names ``tree_ID`` (original label) and
        ``2tree_ID`` (the new ordinal). Unknown names are skipped with a warning. Note
        ``np.column_stack`` upcasts everything to a common dtype (float64).
    plots:
        Optional list of plot names (e.g. ``["plot_01"]``) to restrict the run.
        ``None`` processes every plot folder.
    digits:
        Zero-padding width of the ordinal suffix (3 -> ``_001``).
    overwrite:
        If ``False``, a plot whose ``<plot>_001.npy`` already exists is skipped.
        ``True`` (default) recomputes and overwrites.
    dry_run:
        If ``True``, log what would be done without reading point data or writing files.
    continue_on_error:
        If ``True`` (default), keep going after a plot fails; otherwise stop at the
        first error.
    """
    root = Path(input_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"input_dir does not exist: {root}")
    columns = list(columns) if columns else ["x", "y", "z", "Class"]

    # --- collect the plot directories -------------------------------------
    plot_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if plots:
        wanted = set(plots)
        selected_dirs = [p for p in plot_dirs if p.name in wanted]
        for name in sorted(wanted - {p.name for p in selected_dirs}):
            log.warning("Requested plot %r has no directory in %s", name, root)
        plot_dirs = selected_dirs

    if not plot_dirs:
        log.warning("No plot directories to process in %s — nothing to do.", root)
        return

    log.info("Found %d plot directorie(s) in %s", len(plot_dirs), root)

    succeeded: List[str] = []
    failed: List[str] = []
    skipped: List[str] = []

    # Skips and dry runs are settled here in the parent: no work, nothing to
    # parallelise, and the log keeps plot order.
    jobs: List[Job] = []
    for plot_dir in plot_dirs:
        matches = sorted(plot_dir.glob(pattern))
        if not matches:
            log.warning("[%s] no file matching %r — skipping", plot_dir.name, pattern)
            skipped.append(plot_dir.name)
            continue
        if len(matches) > 1:
            log.warning("[%s] %d files match %r, using %s",
                        plot_dir.name, len(matches), pattern, matches[0].name)
        las_path = matches[0]
        plot = _plot_name(las_path)

        first_out = plot_dir / f"{plot}_{1:0{digits}d}.npy"
        if not overwrite and first_out.exists():
            log.info("[%s] %s already exists and overwrite=False — skipping", plot, first_out.name)
            skipped.append(plot)
            continue

        if dry_run:
            log.info("[%s] DRY RUN: would split %s into <%s_NNN>.npy files (columns: %s) in %s",
                     plot, las_path.name, plot, ", ".join(columns), plot_dir)
            succeeded.append(plot)
            continue

        jobs.append(Job(
            name=plot,
            fn=_split_one_plot,
            args=(las_path, plot, columns, digits),
            n_points=estimate_points(las_path),   # header-only; sizes its RAM share
        ))

    # Plots are independent; how many run at once is decided by memory, not cores.
    for result in run_jobs(jobs, workers=workers,
                           memory_budget_frac=memory_budget_frac,
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

    # --- summary -----------------------------------------------------------
    log.info("=" * 60)
    if dry_run:
        log.info("DRY RUN — would process %d plot(s); %d skipped.", len(succeeded), len(skipped))
    else:
        log.info("Processed %d plot(s): %d ok, %d failed, %d skipped.",
                 len(succeeded) + len(failed), len(succeeded), len(failed), len(skipped))
        if failed:
            log.warning("Failed: %s", ", ".join(failed))
    if skipped:
        log.warning("Skipped: %s", ", ".join(skipped))


# --------------------------------------------------------------------------
# Hydra entry point
# --------------------------------------------------------------------------

# Absolute path to the conf/ directory next to this file, so the script can be run
# from anywhere (Hydra also accepts an absolute config_path).
CONF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conf")


@hydra.main(config_path=CONF_DIR, config_name="config", version_base="1.3")
def main(cfg):
    # misc/conf/config.yaml keeps the splitter's keys under a `tree_splitter:` section
    # (same override prefix as when it was a pipeline stage). zen() maps the section's
    # keys onto split_trees's parameters (and converts OmegaConf containers to plain
    # Python), falling back to defaults for anything absent.
    zen(split_trees)(cfg.tree_splitter)


if __name__ == "__main__":
    main()
