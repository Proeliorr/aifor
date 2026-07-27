r"""Class unifier — remap semantic labels into a new column and export ``.npy``.

Different datasets encode their ground truth differently: the label column has a
different *name* (``Class`` / ``semantic_seg`` / ``classification`` / ...) and the
class *values* mean different things. Before such data can be compared or fed to
ForAINet, the labels must be brought into one common scheme. This stage does exactly
that, for every point cloud in a folder:

1. **read** the cloud (``.las`` 1.2–1.4, ``.laz`` or ``.ply`` — the format does not
   matter, and inputs may sit flat in ``input_dir`` or one level down in per-plot
   subfolders, i.e. the 3DFin output tree from Stage 1);
2. **find the label column** (``semantic_field``; ``None`` auto-detects the usual
   candidates, a configured name missing from a file fails that plot loudly);
3. **create a new column** (``unified_field``, default ``semantic_seg``) whose values
   are the source labels sent through ``class_map`` — a plain ``{source: unified}``
   dict. **Merges** are expressed by giving several sources the same target, e.g.
   ``{3: 2, 10: 2}`` folds classes 3 and 10 into class 2. The map must be **total**:
   a source value it does not mention fails that plot loudly (see
   ``pipeline/classes.py`` for why guessing is worse than failing);
4. **drop** the points whose mapped label is ``drop_value`` (``-1`` by convention) —
   how classes with no counterpart in the target scheme are discarded. The mask is
   applied to every column, so the export never holds a coordinate whose label was
   removed;
5. **export** the same seven columns —
   ``x, y, z, intensity, <source label>, <unified label>, <tree id>`` — in the format
   ``output_format`` asks for, plus a ``<plot>.json`` sidecar (same stem) recording the
   column names/dtypes and the full provenance (source file, class_map, how many points
   were dropped, per-class counts before/after):

   * ``ply`` (default) — a binary PLY keeping each column's **own dtype**. This is the
     one to open in ``misc/view_split_point_cloud.ipynb``: the viewer only draws class
     colours + a legend (and unique counts, and one histogram bar per class) for
     INTEGER columns, so a float64 label is shown as a continuous ramp instead.
     Also ~32 bytes/point rather than 56.
   * ``npy`` — the plain ``N x 7`` float64 matrix ``np.load`` reads anywhere; the
     column names travel in the sidecar (``{"columns": [...]}``, the convention the
     viewer reads, every other key being metadata).
   * ``both`` — writes the pair.

The mapping is **not** defined here: it comes from the shared ``classes:`` block of
``conf/config.yaml``, which ``forainet_prep`` interpolates as well, so the ``.npy``
exports and the PLYs ForAINet trains on can never disagree. The default block maps
the SegmentedForests dataset onto ForAINet's classes; another dataset is a different
``class_map`` and field names, not a code change.

It is driven by ``class_unifier.py`` (assignment root), which reads the
``class_unifier:`` section of the shared ``conf/config.yaml`` — the same pattern as
``pipeline/threedfin.py`` + ``main_pipeline.py``.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import laspy
import numpy as np
from plyfile import PlyData

# The remapping rule itself lives in a neutral module so Stage 2 can apply the
# very same one (see pipeline/classes.py for why it is not defined in either
# stage).
from pipeline.classes import remap_and_filter

# Shared plumbing from Stage 2: input collection, plot-name derivation, the
# format-agnostic reader (with the laspy np.asarray gotcha handled) and the
# field-name candidate lists used for auto-detection.
from pipeline.forainet_prep import (
    _SEMANTIC_CANDIDATES,
    _TREE_ID_CANDIDATES,
    _collect_inputs,
    _plot_name,
    _read_cloud,
    _write_ply,
)

log = logging.getLogger(__name__)


def _available_fields(path: Path) -> Tuple[str, ...]:
    """Return the field names a cloud offers (header-only for LAS/LAZ).

    Used to record in the sidecar *which* source column the auto-detection
    actually picked (``_read_cloud`` returns values, not the matched name).
    """
    suffix = path.suffix.lower()
    if suffix in (".las", ".laz"):
        with laspy.open(str(path)) as reader:  # header only — no point data read
            return tuple(reader.header.point_format.dimension_names)
    if suffix == ".ply":
        return tuple(PlyData.read(str(path))["vertex"].data.dtype.names)
    raise ValueError(f"unsupported input format {suffix!r}: {path.name}")


def _detect_field_name(
    explicit: Optional[str],
    candidates: Tuple[str, ...],
    available: Tuple[str, ...],
    fallback: str,
) -> str:
    """The column name that ``_read_cloud``'s resolution logic will have matched.

    ``explicit`` wins when set; otherwise the first candidate present in the file;
    ``fallback`` names the zero-filled column when nothing matched at all.
    """
    if explicit:
        return explicit
    for name in candidates:
        if name in available:
            return name
    return fallback


def _write_npy_matrix(path: Path, columns: List[np.ndarray]) -> np.ndarray:
    """Write the columns as a plain ``N x K`` float64 matrix and return it.

    Everything is cast to float64 because a bare ``.npy`` holds ONE dtype: this is
    the "load it anywhere with np.load" format, with the column names travelling in
    the sidecar. The cost is that label columns stop looking like integers — see
    :func:`_write_ply_cloud` for why that matters when viewing.
    """
    matrix = np.column_stack([c.astype(np.float64) for c in columns])
    np.save(path, matrix)
    return matrix


def _write_ply_cloud(path: Path, names: List[str], columns: List[np.ndarray]) -> None:
    """Write the columns as a binary PLY, keeping each one's own dtype.

    This is the format to inspect in ``misc/view_split_point_cloud.ipynb``. The viewer
    only treats a column as a set of classes — discrete colours plus a legend, unique
    counts in the statistics table, one histogram bar per value — when its dtype is an
    INTEGER one. A PLY stores per-property dtypes, so ``semantic_seg`` stays ``uint8``
    and ``tree_ID`` stays an int; the float64 ``.npy`` cannot express that and every
    label there is drawn as a continuous ramp.

    Keeping the native dtypes also makes the file roughly 32 bytes/point instead of 56.
    """
    dtype = [(name, col.dtype.str) for name, col in zip(names, columns)]
    array = np.empty(len(columns[0]), dtype=dtype)
    for name, col in zip(names, columns):
        array[name] = col
    _write_ply(array, path)


def _value_counts(labels: np.ndarray) -> Dict[str, int]:
    """``{value: count}`` histogram with JSON-safe (string) keys."""
    values, counts = np.unique(np.asarray(labels), return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(values, counts)}


def _short_dtype(arr: np.ndarray) -> str:
    """numpy dtype as its short code ('f8', 'u1', ...) for the sidecar."""
    return arr.dtype.str.lstrip("<>|=")


def _unify_one(
    src_path: Path,
    plot: str,
    out_dir: Path,
    semantic_field: Optional[str],
    tree_id_field: Optional[str],
    intensity_field: Optional[str],
    unified_field: str,
    class_map: Dict[int, int],
    drop_value: Optional[int],
    class_names: Dict[int, str],
    output_format: str = "ply",
) -> None:
    """Read one cloud, remap its labels, write the cloud + ``<plot>.json``."""
    xyz, intensity, semantic, tree_id, meta = _read_cloud(
        src_path,
        semantic_field=semantic_field,
        tree_id_field=tree_id_field,
        intensity_field=intensity_field,
    )

    # Names of the source columns as they will appear in the sidecar. When the
    # source label column is itself called like `unified_field` (e.g. a PLY that
    # already has `semantic_seg`), suffix it so the two columns stay distinct.
    available = _available_fields(src_path)
    semantic_name = _detect_field_name(semantic_field, _SEMANTIC_CANDIDATES,
                                       available, "source_label")
    tree_id_name = _detect_field_name(tree_id_field, _TREE_ID_CANDIDATES,
                                      available, "tree_ID")
    if semantic_name == unified_field:
        semantic_name = f"{semantic_name}_src"
        log.info("[%s] source label column shares the unified name — recorded as %r",
                 plot, semantic_name)

    counts_before = _value_counts(semantic)

    # Remap, then drop the points whose class has no counterpart in the target
    # scheme. The mask subsets EVERY column, so the exported matrix stays
    # internally consistent (a coordinate never outlives its label).
    unified, keep, n_dropped = remap_and_filter(semantic, class_map, drop_value, plot)
    dropped_values = ([int(v) for v in np.unique(np.asarray(semantic)[~keep])]
                      if n_dropped else [])
    xyz, intensity = xyz[keep], intensity[keep]
    semantic, tree_id = np.asarray(semantic)[keep], tree_id[keep]

    counts_after = _value_counts(unified)
    log.info("[%s] class counts before: %s", plot, counts_before)
    log.info("[%s] class counts after:  %s", plot, counts_after)
    if n_dropped:
        log.info("[%s] dropped %s point(s) from source class(es) %s (-> %s)",
                 plot, f"{n_dropped:,}", dropped_values, drop_value)

    # One column per field, in the order the sidecar advertises. The source label sits
    # next to the unified one on purpose: that is what lets the split viewer show the
    # before/after of a class merge from a single file.
    columns = ["x", "y", "z", "intensity", semantic_name, unified_field, tree_id_name]
    values = [xyz[:, 0], xyz[:, 1], xyz[:, 2], intensity, semantic, unified, tree_id]
    n_points = len(xyz)

    written = []
    if output_format in ("ply", "both"):
        ply_path = out_dir / f"{plot}.ply"
        _write_ply_cloud(ply_path, columns, values)
        written.append(ply_path)
    if output_format in ("npy", "both"):
        npy_path = out_dir / f"{plot}.npy"
        _write_npy_matrix(npy_path, values)
        written.append(npy_path)

    # Sidecar: `columns` is what readers (the viewer, numpy users) need to restore
    # the column names of a bare .npy matrix; `dtypes` are the ORIGINAL dtypes, which
    # the .ply keeps natively and a .npy flattens to float64; the rest is provenance.
    sidecar = {
        "columns": columns,
        "dtypes": ["f8", "f8", "f8", _short_dtype(intensity),
                   _short_dtype(semantic), _short_dtype(unified), _short_dtype(tree_id)],
        "output_format": output_format,
        "files": [p.name for p in written],
        "source_file": meta.get("source_file"),
        "source_format": meta.get("source_format"),
        "crs_wkt": meta.get("crs_wkt"),
        "num_points": int(n_points),
        "num_points_source": int(keep.size),
        "semantic_field": semantic_name,
        "unified_field": unified_field,
        "class_map": {str(int(k)): int(v) for k, v in class_map.items()},
        "drop_value": drop_value,
        "dropped_points": n_dropped,
        "dropped_source_values": dropped_values,
        "class_names": {str(int(k)): str(v) for k, v in class_names.items()},
        # NOTE: `before` counts the source cloud, `after` only the points kept —
        # with points dropped the two do NOT sum to the same total.
        "class_counts_before": counts_before,
        "class_counts_after": counts_after,
    }
    json_path = out_dir / f"{plot}.json"
    json_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

    # A leftover file from a previous run in the OTHER format still sits next to the
    # shared sidecar, and the viewer would happily open it as if it were current.
    for stale in (out_dir / f"{plot}.npy", out_dir / f"{plot}.ply"):
        if stale not in written and stale.exists():
            log.warning("[%s] %s is left over from an earlier run in another format "
                        "and is now out of date — delete it or re-run with "
                        "output_format=both", plot, stale.name)

    log.info("[%s] %s of %s points kept -> %s", plot, f"{n_points:,}", f"{keep.size:,}",
             ", ".join(p.name for p in written + [json_path]))


def unify_classes(
    input_dir: str = r"D:\Podyplomowe\04_AI_Intro\assignment\SegmentedForests\3DFin_output",
    patterns: Optional[List[str]] = None,
    output_dir: str = r"D:\Podyplomowe\04_AI_Intro\assignment\SegmentedForests\ClassUnifier_output",
    semantic_field: Optional[str] = None,
    tree_id_field: Optional[str] = None,
    intensity_field: Optional[str] = None,
    unified_field: str = "semantic_seg",
    class_map: Optional[Dict[int, int]] = None,
    drop_value: Optional[int] = -1,
    class_names: Optional[Dict[int, str]] = None,
    output_format: str = "ply",
    plots: Optional[List[str]] = None,
    overwrite: bool = True,
    dry_run: bool = False,
    continue_on_error: bool = True,
) -> None:
    """Remap the semantic labels of every matched cloud and export ``.npy`` + ``.json``.

    Parameters
    ----------
    input_dir:
        Folder of input clouds. ``patterns`` (default ``["*.las", "*.laz", "*.ply"]``)
        are matched directly inside it AND one level down in per-plot subfolders, so
        both a flat folder and the 3DFin output tree work unchanged.
    output_dir:
        Folder receiving ``<plot>.npy`` + ``<plot>.json`` (created if missing).
    semantic_field, tree_id_field, intensity_field:
        Source column names. ``None`` (default) auto-detects from the usual
        candidates (semantic: ``Class``/``semantic_seg``/``classification``; tree id:
        ``tree_ID``/``treeID``; intensity: ``intensity``/``scalar_Intensity``/...).
        A configured name missing from a file fails that plot loudly instead of
        silently writing zeros.
    unified_field:
        Name of the NEW column holding the remapped labels (default
        ``semantic_seg`` — what ForAINet expects).
    class_map:
        ``{source: unified}`` value mapping. Several sources may share one target —
        that merges classes (``{3: 2, 10: 2}`` folds 3 and 10 into 2). The map must
        be **total**: a source value it does not mention fails that plot loudly
        rather than being guessed at (see ``pipeline/classes.py``).
    drop_value:
        Points whose mapped label equals this are **removed** from the export —
        the way classes with no counterpart in the target scheme are discarded.
        ``None`` keeps every point.
    class_names:
        ``{value: name}`` documentation of the unified scheme, echoed into each
        sidecar (defaults to ForAINet's classes in ``conf/config.yaml``).
    output_format:
        ``ply`` (default), ``npy`` or ``both``; the ``.json`` sidecar is written
        either way. Prefer ``ply`` for inspection in
        ``misc/view_split_point_cloud.ipynb``: it keeps each column's own dtype, and
        the viewer only draws class colours/legends and unique counts for INTEGER
        columns — in the float64 ``.npy`` every label looks continuous. ``npy`` is
        the load-anywhere matrix.
    plots:
        Optional list of plot stems (e.g. ``["plot_01"]``) to restrict the run.
    overwrite:
        ``False`` skips plots whose ``<plot>.npy`` already exists.
    dry_run:
        Log what would be processed without reading or writing anything.
    continue_on_error:
        Keep going after a plot fails (default) or stop at the first failure.
    """
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    patterns = list(patterns) if patterns else ["*.las", "*.laz", "*.ply"]
    if output_format not in ("ply", "npy", "both"):
        raise ValueError(
            f"output_format must be 'ply', 'npy' or 'both', got {output_format!r}")
    # The files a plot is considered "already done" by (see overwrite below).
    suffixes = {"ply": [".ply"], "npy": [".npy"], "both": [".ply", ".npy"]}[output_format]
    class_map = {int(k): int(v) for k, v in (class_map or {}).items()}
    class_names = {int(k): str(v) for k, v in (class_names or {}).items()}

    if not in_dir.is_dir():
        raise NotADirectoryError(f"input_dir does not exist: {in_dir}")

    inputs = _collect_inputs(in_dir, patterns)
    if plots:
        wanted = set(plots)
        selected = [p for p in inputs if _plot_name(p) in wanted]
        for stem in sorted(wanted - {_plot_name(p) for p in selected}):
            log.warning("Requested plot %r matched no input in %s", stem, in_dir)
        inputs = selected

    if not inputs:
        log.warning("No inputs matched %s in %s — nothing to do.", patterns, in_dir)
        return

    log.info("Found %d input cloud(s) in %s", len(inputs), in_dir)
    if not class_map:
        raise ValueError(
            "class_map is empty — every source class needs an explicit target. "
            "Set the `classes.class_map` block in conf/config.yaml."
        )
    log.info("Mapping %d source class(es) -> %s%s", len(class_map),
             sorted({int(v) for v in class_map.values()}),
             "" if drop_value is None else f"  (points mapping to {drop_value} are dropped)")

    succeeded: List[str] = []
    failed: List[str] = []
    skipped: List[str] = []

    for src in inputs:
        plot = _plot_name(src)
        out_paths = [out_dir / f"{plot}{suf}" for suf in suffixes]

        # For output_format=both a plot only counts as done when BOTH files are there,
        # so an interrupted run finishes rather than being skipped half-written.
        if not overwrite and all(p.exists() for p in out_paths):
            log.info("[%s] output exists (%s) and overwrite=false — skipping",
                     plot, ", ".join(p.name for p in out_paths))
            skipped.append(plot)
            continue

        if dry_run:
            log.info("[%s] DRY RUN: %s -> %s (map: %s, drop: %s)",
                     plot, src.name,
                     ", ".join([p.name for p in out_paths] + [f"{plot}.json"]),
                     class_map, drop_value)
            skipped.append(plot)
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            _unify_one(
                src, plot, out_dir,
                semantic_field, tree_id_field, intensity_field,
                unified_field, class_map, drop_value, class_names, output_format,
            )
            succeeded.append(plot)
        except Exception as exc:
            log.error("[%s] FAILED: %s", plot, exc)
            failed.append(plot)
            if not continue_on_error:
                log.error("continue_on_error=False — stopping after first failure.")
                break

    # --- summary ---------------------------------------------------------
    log.info("=" * 60)
    if dry_run:
        log.info("DRY RUN — would process %d plot(s).", len(skipped))
    else:
        log.info("Processed %d plot(s): %d ok, %d failed, %d skipped.",
                 len(succeeded) + len(failed), len(succeeded), len(failed), len(skipped))
        if failed:
            log.warning("Failed: %s", ", ".join(failed))
