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
   ``{1: 2, 4: 2}`` folds classes 1 and 4 into class 2. Source values absent from the
   map follow ``unmapped_value`` (``None`` = keep as-is with a warning, an int =
   force that value);
4. **export** ``<output_dir>/<plot>.npy`` — a plain ``N x 7`` float64 matrix with the
   columns ``x, y, z, intensity, <source label>, <unified label>, <tree id>`` — plus
   a ``<plot>.json`` sidecar (same stem) recording the column names/dtypes and the
   full provenance (source file, class_map, per-class point counts before/after).
   The sidecar is the same convention ``misc/view_split_point_cloud.ipynb`` reads:
   ``{"columns": [...]}`` restores the column names, every other key is metadata.

The default ``class_map`` in ``conf/config.yaml`` is the **identity** ({0:0 ... 4:4})
and the default ``class_names`` document ForAINet's five classes — edit both when the
real correspondence between your dataset and ForAINet is known.

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

# Shared plumbing from Stage 2: input collection, plot-name derivation, the
# format-agnostic reader (with the laspy np.asarray gotcha handled) and the
# field-name candidate lists used for auto-detection.
from pipeline.forainet_prep import (
    _SEMANTIC_CANDIDATES,
    _TREE_ID_CANDIDATES,
    _collect_inputs,
    _plot_name,
    _read_cloud,
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


def _apply_class_map(
    labels: np.ndarray,
    class_map: Dict[int, int],
    unmapped_value: Optional[int],
    plot: str,
) -> Tuple[np.ndarray, List[int]]:
    """Send ``labels`` through ``class_map`` and return ``(unified, unmapped_values)``.

    The remap is vectorised: one boolean mask per map entry, applied to a copy —
    the source column itself is never modified. Because several source values may
    share one target, class merges need no special handling. Values not covered by
    the map are kept (``unmapped_value=None``) or forced to ``unmapped_value``.
    """
    src = np.asarray(labels)
    unified = src.astype(np.int64, copy=True)
    covered = np.zeros(src.shape, dtype=bool)
    for old, new in class_map.items():
        hit = src == int(old)
        unified[hit] = int(new)
        covered |= hit

    unmapped = [int(v) for v in np.unique(src[~covered])]
    if unmapped:
        if unmapped_value is None:
            log.warning("[%s] source value(s) %s not in class_map — kept as-is",
                        plot, unmapped)
        else:
            unified[~covered] = int(unmapped_value)
            log.warning("[%s] source value(s) %s not in class_map — set to %d",
                        plot, unmapped, int(unmapped_value))

    # ForAINet stores labels as uint8; keep that when the values fit.
    if unified.min() >= 0 and unified.max() <= 255:
        unified = unified.astype(np.uint8)
    else:
        log.warning("[%s] unified labels exceed uint8 range — keeping int64", plot)
    return unified, unmapped


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
    unmapped_value: Optional[int],
    class_names: Dict[int, str],
) -> None:
    """Read one cloud, remap its labels, write ``<plot>.npy`` + ``<plot>.json``."""
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

    unified, unmapped = _apply_class_map(semantic, class_map, unmapped_value, plot)
    counts_before = _value_counts(semantic)
    counts_after = _value_counts(unified)
    log.info("[%s] class counts before: %s", plot, counts_before)
    log.info("[%s] class counts after:  %s", plot, counts_after)

    # The minimal ForAINet set, one column per field, everything as float64 —
    # a plain matrix np.load can read anywhere; names travel in the sidecar.
    columns = ["x", "y", "z", "intensity", semantic_name, unified_field, tree_id_name]
    matrix = np.column_stack([
        xyz[:, 0], xyz[:, 1], xyz[:, 2],
        intensity.astype(np.float64),
        semantic.astype(np.float64),
        unified.astype(np.float64),
        tree_id.astype(np.float64),
    ])

    npy_path = out_dir / f"{plot}.npy"
    np.save(npy_path, matrix)

    # Sidecar: `columns` is what readers (the viewer, numpy users) need to restore
    # the column names; `dtypes` are the ORIGINAL dtypes should a faithful
    # per-column cast be wanted; the rest is provenance for later diagnosis.
    sidecar = {
        "columns": columns,
        "dtypes": ["f8", "f8", "f8", _short_dtype(intensity),
                   _short_dtype(semantic), _short_dtype(unified), _short_dtype(tree_id)],
        "source_file": meta.get("source_file"),
        "source_format": meta.get("source_format"),
        "crs_wkt": meta.get("crs_wkt"),
        "num_points": int(len(matrix)),
        "semantic_field": semantic_name,
        "unified_field": unified_field,
        "class_map": {str(int(k)): int(v) for k, v in class_map.items()},
        "unmapped_values": unmapped,
        "unmapped_value": unmapped_value,
        "class_names": {str(int(k)): str(v) for k, v in class_names.items()},
        "class_counts_before": counts_before,
        "class_counts_after": counts_after,
    }
    json_path = out_dir / f"{plot}.json"
    json_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

    log.info("[%s] %s points -> %s + %s", plot, f"{len(matrix):,}",
             npy_path.name, json_path.name)


def unify_classes(
    input_dir: str = r"D:\Podyplomowe\04_AI_Intro\assignment\SegmentedForests\3DFin_output",
    patterns: Optional[List[str]] = None,
    output_dir: str = r"D:\Podyplomowe\04_AI_Intro\assignment\SegmentedForests\ClassUnifier_output",
    semantic_field: Optional[str] = None,
    tree_id_field: Optional[str] = None,
    intensity_field: Optional[str] = None,
    unified_field: str = "semantic_seg",
    class_map: Optional[Dict[int, int]] = None,
    unmapped_value: Optional[int] = None,
    class_names: Optional[Dict[int, str]] = None,
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
        that merges classes (``{1: 2, 4: 2}`` folds 1 and 4 into 2). ``None``/empty
        keeps every value unchanged (the unified column is then a plain copy).
    unmapped_value:
        What happens to source values absent from ``class_map``: ``None`` (default)
        keeps them as-is with a warning, an integer forces that value.
    class_names:
        ``{value: name}`` documentation of the unified scheme, echoed into each
        sidecar (defaults to ForAINet's five classes in ``conf/config.yaml``).
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
        log.info("class_map is empty — the unified column will be a plain copy.")

    succeeded: List[str] = []
    failed: List[str] = []
    skipped: List[str] = []

    for src in inputs:
        plot = _plot_name(src)
        npy_path = out_dir / f"{plot}.npy"

        if not overwrite and npy_path.exists():
            log.info("[%s] output exists (%s) and overwrite=false — skipping",
                     plot, npy_path.name)
            skipped.append(plot)
            continue

        if dry_run:
            log.info("[%s] DRY RUN: %s -> %s + %s (map: %s)",
                     plot, src.name, npy_path.name, f"{plot}.json",
                     class_map or "copy")
            skipped.append(plot)
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            _unify_one(
                src, plot, out_dir,
                semantic_field, tree_id_field, intensity_field,
                unified_field, class_map, unmapped_value, class_names,
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
