r"""Stage 2 — prepare plot clouds for ForAINet (and restore its results).

ForAINet consumes binary PLY files with **non-negative coordinates** and the vertex
fields ``x, y, z, intensity, semantic_seg, treeID``. This stage bridges arbitrary
input clouds and ForAINet, in one of two modes. **The input format does not matter**:
``.las`` (any version 1.2–1.4), ``.laz`` (compressed) and ``.ply`` are all read the
same way, whether they sit directly in ``input_dir`` (flat layout) or one level down
in per-plot subfolders (the 3DFin output tree from Stage 1).

``mode="preprocess"`` (before classification)
    For every matched cloud: read it, compute the per-plot coordinate shifts
    ``offset_x/y/z = min(x/y/z)``, subtract them (centering — every coordinate becomes
    >= 0), and write

    * ``<output_dir>/<plot>.ply`` — the ForAINet-ready cloud
      (``x, y, z, intensity, semantic_seg, treeID``), and
    * ``<output_dir>/<plot>_offsets.yml`` — the shifts, validation statistics
      (original mins/ranges, point count) **and the source metadata needed for a
      faithful restoration**: source format/file, LAS version, point format, scales,
      and the CRS as WKT (``null`` for PLY sources or clouds without a CRS).

``mode="restore"`` (after classification)
    For every ``*.ply`` in ``restore_dir``: recover the plot name from the file stem,
    load its ``<plot>_offsets.yml``, **add the shifts back** to x/y/z (validating the
    restored minima/ranges against the recorded statistics), and write
    ``restored_<plot>.<restore_format>`` next to the input. The default output is
    **compressed LAZ (LAS 1.4)** with the source scales and CRS re-applied and every
    non-standard vertex field (``semantic_seg``, ``treeID``, predictions, ...) kept
    as LAS extra-bytes dimensions — nothing is lost.

The label -> ``semantic_seg`` assignment is isolated in :func:`_semantic_labels`;
a future ``class_map`` config key (merging classes / renumbering) will plug in there.

It is driven by ``forainet_prep.py`` (assignment root), which reads the
``forainet_prep:`` section of the shared ``conf/config.yaml`` — the same pattern as
``pipeline/threedfin.py`` + ``main_pipeline.py``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import laspy
import numpy as np
import yaml
from laspy.vlrs.known import WktCoordinateSystemVlr
from plyfile import PlyData, PlyElement

log = logging.getLogger(__name__)

# Suffix stripped from the input file name to recover the plot name
# (plot_01_tree_ID_dist_axes.las -> plot_01).
_INPUT_SUFFIX = "_tree_ID_dist_axes"

# Prefixes that classification tools may prepend to a plot's file name; stripped when
# recovering the plot name in restore mode (dummy_plot_01.ply -> plot_01).
_RESTORE_PREFIXES = ("restored_", "dummy_")

# Vertex layout ForAINet expects (see ForAINet's treeins datasets): coordinates and
# intensity as float64, the semantic label as uint8, the instance id as uint32.
_FORAINET_DTYPE = [
    ("x", "f8"),
    ("y", "f8"),
    ("z", "f8"),
    ("intensity", "f8"),
    ("semantic_seg", "u1"),
    ("treeID", "u4"),
]

# Field-name candidates tried in order when reading inputs. LAS: `Class` is the
# 3DFin/dataset extra dimension, `classification` the standard LAS field. PLY:
# `scalar_*` are CloudCompare export variants.
_SEMANTIC_CANDIDATES = ("Class", "semantic_seg", "classification")
_TREE_ID_CANDIDATES = ("tree_ID", "treeID")
_INTENSITY_CANDIDATES = ("intensity", "scalar_Intensity", "scalar_intensity", "Intensity")

# Tolerance for the restore-mode validation of minima/ranges (matches the original
# Points2ForAINet.py checks).
_TOL = 1e-6


def _plot_name(path: Path) -> str:
    """Derive the plot name from the input file stem (stripping the 3DFin suffix)."""
    stem = path.stem
    if stem.endswith(_INPUT_SUFFIX):
        return stem[: -len(_INPUT_SUFFIX)]
    return stem


def _semantic_labels(labels: np.ndarray) -> np.ndarray:
    """Return the per-point semantic label (``semantic_seg``) as uint8.

    Currently a straight copy of the label array found in the source cloud. This
    helper is the designated hook for the planned **class reclassification** step:
    a future ``class_map: {old: new}`` config key will be applied here (merging
    classes and assigning new class numbers) before the labels are written out.
    """
    return labels.astype(np.uint8)


def _crs_wkt(header: laspy.LasHeader) -> Optional[str]:
    """Extract the CRS of a LAS/LAZ header as a WKT string (``None`` if absent).

    A WKT VLR is read directly (no pyproj needed); GeoTIFF-key CRSs go through
    laspy's ``parse_crs`` (which requires pyproj) and are converted to WKT.
    """
    for vlr in header.vlrs:
        if isinstance(vlr, WktCoordinateSystemVlr):
            return vlr.string
    try:
        crs = header.parse_crs()
        return crs.to_wkt() if crs is not None else None
    except Exception as exc:  # e.g. pyproj missing for GeoTIFF-key CRS
        log.warning("could not parse CRS (%s) — storing null", exc)
        return None


def _first_field(names: Tuple[str, ...], available, getter) -> Optional[np.ndarray]:
    """Return the first candidate field found via `getter`, or None."""
    for name in names:
        if name in available:
            return np.asarray(getter(name))
    return None


def _resolve_field(
    explicit: Optional[str],
    candidates: Tuple[str, ...],
    available,
    getter,
    path: Path,
    kind: str,
) -> Optional[np.ndarray]:
    """Fetch a field either by its explicitly configured name or by auto-detection.

    An explicitly configured name that is missing from the file is an error (the
    plot fails loudly instead of silently training on zeros); ``explicit=None``
    falls back to trying the ``candidates`` in order, returning ``None`` when no
    candidate exists.
    """
    if explicit:
        if explicit in available:
            return np.asarray(getter(explicit))
        raise ValueError(
            f"configured {kind} field {explicit!r} not present in {path.name} "
            f"(available: {', '.join(sorted(available))})"
        )
    return _first_field(candidates, available, getter)


def _read_cloud(
    path: Path,
    semantic_field: Optional[str] = None,
    tree_id_field: Optional[str] = None,
    intensity_field: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Read any supported cloud into ``(xyz, intensity, semantic, tree_id, meta)``.

    ``xyz`` is an ``[N, 3]`` float64 array; the other three are length-N arrays
    (zeros + warning when a field is missing from the source). The three
    ``*_field`` names override the built-in candidate lists (``None`` =
    auto-detect); a configured name missing from the file raises. ``meta`` records
    what a faithful restoration needs: source format/file and — for LAS/LAZ —
    version, point format id, scales and the CRS as WKT (``None`` where not
    applicable). All laspy accessors are materialised with ``np.asarray``: laspy
    exposes x/y/z as ScaledArrayView and bit fields as SubFieldView, which numpy
    does not treat as plain arrays.
    """
    suffix = path.suffix.lower()

    if suffix in (".las", ".laz"):
        las = laspy.read(str(path))
        dims = set(las.point_format.dimension_names)
        xyz = np.column_stack([
            np.asarray(las.x, dtype=np.float64),
            np.asarray(las.y, dtype=np.float64),
            np.asarray(las.z, dtype=np.float64),
        ])
        getter = lambda name: getattr(las, name)
        intensity = _resolve_field(intensity_field, _INTENSITY_CANDIDATES, dims, getter, path, "intensity")
        semantic = _resolve_field(semantic_field, _SEMANTIC_CANDIDATES, dims, getter, path, "semantic label")
        tree_id = _resolve_field(tree_id_field, _TREE_ID_CANDIDATES, dims, getter, path, "tree id")
        meta = {
            "source_format": suffix.lstrip("."),
            "source_file": path.name,
            "las_version": str(las.header.version),
            "point_format": int(las.point_format.id),
            "scales": [float(s) for s in las.header.scales],
            "crs_wkt": _crs_wkt(las.header),
        }

    elif suffix == ".ply":
        vertex = PlyData.read(str(path))["vertex"].data
        names = vertex.dtype.names
        xyz = np.column_stack([
            np.asarray(vertex["x"], dtype=np.float64),
            np.asarray(vertex["y"], dtype=np.float64),
            np.asarray(vertex["z"], dtype=np.float64),
        ])
        getter = lambda name: vertex[name]
        intensity = _resolve_field(intensity_field, _INTENSITY_CANDIDATES, names, getter, path, "intensity")
        # PLY sources most likely already use the ForAINet names, so try those first.
        semantic = _resolve_field(semantic_field, ("semantic_seg",) + _SEMANTIC_CANDIDATES,
                                  names, getter, path, "semantic label")
        tree_id = _resolve_field(tree_id_field, ("treeID",) + _TREE_ID_CANDIDATES,
                                 names, getter, path, "tree id")
        meta = {
            "source_format": "ply",
            "source_file": path.name,
            "las_version": None,
            "point_format": None,
            "scales": None,
            "crs_wkt": None,
        }

    else:
        raise ValueError(f"unsupported input format {suffix!r}: {path.name}")

    n = len(xyz)
    if intensity is None:
        intensity = np.zeros(n, dtype=np.float64)
    if semantic is None:
        log.warning("[%s] no semantic label field found (tried %s) — writing zeros",
                    path.name, ", ".join(_SEMANTIC_CANDIDATES))
        semantic = np.zeros(n, dtype=np.uint8)
    if tree_id is None:
        log.warning("[%s] no tree id field found (tried %s) — writing zeros",
                    path.name, ", ".join(_TREE_ID_CANDIDATES))
        tree_id = np.zeros(n, dtype=np.uint32)
    return xyz, intensity, semantic, tree_id, meta


def _write_ply(array: np.ndarray, path: Path) -> None:
    """Write a structured numpy array as a binary PLY 'vertex' element."""
    element = PlyElement.describe(array, "vertex")
    PlyData([element], text=False).write(str(path))


def _read_ply(path: Path) -> np.ndarray:
    """Read a PLY's 'vertex' element into a writable structured numpy array."""
    data = PlyData.read(str(path))
    # np.array(...) copies: plyfile hands back a memory-mapped/read-only view.
    return np.array(data["vertex"].data)


def _write_las(array: np.ndarray, path: Path, meta: Dict) -> None:
    """Write a structured array to LAS 1.4 / LAZ, restoring the source metadata.

    * ``.laz`` suffix -> compressed via the lazrs backend, ``.las`` -> uncompressed.
    * Scales come from the recorded source metadata (fallback 0.001 = mm precision);
      offsets are the per-axis minima.
    * The source CRS (``crs_wkt``) is re-attached as a WKT VLR.
    * Every field beyond x/y/z/intensity (``semantic_seg``, ``treeID``, predicted
      labels, ...) is stored as a LAS extra-bytes dimension — nothing is dropped.
    """
    header = laspy.LasHeader(point_format=6, version="1.4")
    scales = meta.get("scales") or [0.001, 0.001, 0.001]
    header.scales = np.asarray(scales, dtype=np.float64)
    header.offsets = np.array([array["x"].min(), array["y"].min(), array["z"].min()])

    crs_wkt = meta.get("crs_wkt")
    if crs_wkt:
        header.vlrs.append(WktCoordinateSystemVlr(crs_wkt))
        header.global_encoding.wkt = True

    standard = set(header.point_format.dimension_names)
    extra = [n for n in array.dtype.names
             if n not in ("x", "y", "z", "intensity") and n not in standard]
    for name in extra:
        header.add_extra_dim(laspy.ExtraBytesParams(name=name, type=array.dtype.fields[name][0]))

    las = laspy.LasData(header)
    las.x = array["x"]
    las.y = array["y"]
    las.z = array["z"]
    if "intensity" in array.dtype.names:
        las.intensity = np.clip(array["intensity"], 0, 65535).astype(np.uint16)
    for name in extra:
        las[name] = array[name]
    # Fields that happen to match a standard LAS dimension (e.g. `classification`)
    # are written into that dimension instead of an extra-bytes one.
    for name in array.dtype.names:
        if name in standard and name not in ("x", "y", "z", "intensity"):
            las[name] = array[name]

    las.write(str(path))


# ---------------------------------------------------------------------------
# mode="preprocess"
# ---------------------------------------------------------------------------

def _collect_inputs(root: Path, patterns: List[str]) -> List[Path]:
    """Match ``patterns`` directly in ``root`` and one level down (plot subfolders).

    Covers both supported layouts — a flat folder of user-supplied clouds and the
    3DFin output tree (one subfolder per plot) — deduplicated and sorted.
    """
    found = set()
    for pattern in patterns:
        found.update(p.resolve() for p in root.glob(pattern) if p.is_file())
        found.update(p.resolve() for p in root.glob(f"*/{pattern}") if p.is_file())
    return sorted(found)


def _preprocess_one(
    src_path: Path,
    plot: str,
    out_dir: Path,
    semantic_field: Optional[str] = None,
    tree_id_field: Optional[str] = None,
    intensity_field: Optional[str] = None,
) -> int:
    """Center one cloud, write ``<plot>.ply`` + ``<plot>_offsets.yml``.

    Returns the point count.
    """
    xyz, intensity, semantic, tree_id, meta = _read_cloud(
        src_path, semantic_field, tree_id_field, intensity_field)
    n_points = len(xyz)

    # The shifts are the per-axis minima: subtracting them guarantees every
    # coordinate is >= 0 (ForAINet's requirement), with 0 at the plot corner.
    offset_x = float(xyz[:, 0].min())
    offset_y = float(xyz[:, 1].min())
    offset_z = float(xyz[:, 2].min())

    array = np.empty(n_points, dtype=_FORAINET_DTYPE)
    array["x"] = xyz[:, 0] - offset_x
    array["y"] = xyz[:, 1] - offset_y
    array["z"] = xyz[:, 2] - offset_z
    array["intensity"] = intensity.astype(np.float64)
    array["semantic_seg"] = _semantic_labels(semantic)
    array["treeID"] = tree_id.astype(np.uint32)

    ply_path = out_dir / f"{plot}.ply"
    _write_ply(array, ply_path)

    # The shifts plus statistics for validating a later restoration, and the source
    # metadata a faithful restore needs (format, LAS header info, CRS). Plain
    # floats/ints (not numpy scalars) so the YAML stays human-readable.
    offsets = {
        "plot": plot,
        "offset_x": offset_x,
        "offset_y": offset_y,
        "offset_z": offset_z,
        "original_min_x": offset_x,
        "original_min_y": offset_y,
        "original_min_z": offset_z,
        "original_range_x": float(np.ptp(xyz[:, 0])),
        "original_range_y": float(np.ptp(xyz[:, 1])),
        "original_range_z": float(np.ptp(xyz[:, 2])),
        "n_points": int(n_points),
        **meta,
    }
    offsets_path = out_dir / f"{plot}_offsets.yml"
    with offsets_path.open("w") as fh:
        yaml.dump(offsets, fh, default_flow_style=False, sort_keys=False)

    log.info(
        "[%s] %s points centered from %s (shifts: x=%.3f y=%.3f z=%.3f, crs=%s) -> %s + %s",
        plot, f"{n_points:,}", meta["source_file"], offset_x, offset_y, offset_z,
        "yes" if meta["crs_wkt"] else "none", ply_path.name, offsets_path.name,
    )
    return n_points


# ---------------------------------------------------------------------------
# mode="restore"
# ---------------------------------------------------------------------------

def _restore_plot_name(ply_path: Path) -> str:
    """Recover the plot name from a classified PLY's stem, stripping known prefixes."""
    stem = ply_path.stem
    for prefix in _RESTORE_PREFIXES:
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
    return stem


def _restore_one(ply_path: Path, plot: str, offsets: Dict, out_path: Path) -> int:
    """Add the recorded shifts back to one classified PLY and save it.

    Validates the restored minima and ranges against the statistics recorded at
    preprocessing time (mismatch -> error/warning log, the file is still written).
    Returns the point count.
    """
    array = _read_ply(ply_path)
    for coord in ("x", "y", "z"):
        if coord not in array.dtype.names:
            raise ValueError(f"cannot restore {ply_path.name}: missing vertex field {coord!r}")

    array["x"] = array["x"] + offsets["offset_x"]
    array["y"] = array["y"] + offsets["offset_y"]
    array["z"] = array["z"] + offsets["offset_z"]

    # --- validation against the recorded statistics ------------------------
    for axis in ("x", "y", "z"):
        restored_min = float(array[axis].min())
        expected_min = offsets.get(f"original_min_{axis}")
        if expected_min is not None and abs(restored_min - expected_min) > _TOL:
            log.error("[%s] %s minimum mismatch after restoration: expected %.6f, got %.6f",
                      plot, axis, expected_min, restored_min)
        restored_range = float(np.ptp(array[axis]))
        expected_range = offsets.get(f"original_range_{axis}")
        if expected_range is not None and abs(restored_range - expected_range) > _TOL:
            log.warning("[%s] %s range mismatch after restoration: expected %.6f, got %.6f",
                        plot, axis, expected_range, restored_range)
    if "n_points" in offsets and len(array) != offsets["n_points"]:
        log.warning("[%s] point count changed: preprocessed %d, restoring %d",
                    plot, offsets["n_points"], len(array))

    if out_path.suffix.lower() == ".ply":
        _write_ply(array, out_path)
    else:
        _write_las(array, out_path, offsets)

    log.info("[%s] %s points restored (shifts added back: x=%.3f y=%.3f z=%.3f) -> %s",
             plot, f"{len(array):,}", offsets["offset_x"], offsets["offset_y"],
             offsets["offset_z"], out_path.name)
    return len(array)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def prep_forainet(
    mode: str = "preprocess",
    input_dir: str = r"D:\Podyplomowe\04_AI_Intro\assignment\SegmentedForests\3DFin_output",
    patterns: Optional[List[str]] = None,
    output_dir: str = r"D:\Podyplomowe\04_AI_Intro\assignment\SegmentedForests\ForAINet_input",
    semantic_field: Optional[str] = None,
    tree_id_field: Optional[str] = None,
    intensity_field: Optional[str] = None,
    restore_dir: Optional[str] = None,
    offsets_dir: Optional[str] = None,
    restore_format: str = "laz",
    plots: Optional[List[str]] = None,
    overwrite: bool = True,
    dry_run: bool = False,
    continue_on_error: bool = True,
) -> None:
    """Center plot clouds for ForAINet, or restore its results to original coordinates.

    Parameters
    ----------
    mode:
        ``"preprocess"`` = input clouds -> centered ForAINet PLYs + per-plot offset
        files; ``"restore"`` = classified PLYs -> original coordinates.
    input_dir:
        (preprocess) Folder holding the input clouds — either directly (flat layout)
        or one level down in per-plot subfolders (the 3DFin output tree). Any mix of
        ``.las`` (1.2–1.4), ``.laz`` and ``.ply`` is accepted.
    patterns:
        (preprocess) Globs matched in both layouts (default
        ``["*.las", "*.laz", "*.ply"]``). Narrow this (e.g.
        ``["*_tree_ID_dist_axes.las"]``) when folders hold more matching files than
        the one cloud per plot.
    output_dir:
        (preprocess) Folder receiving ``<plot>.ply`` + ``<plot>_offsets.yml``; created
        if missing. Also the default place restore mode looks for the offset files.
    semantic_field, tree_id_field, intensity_field:
        (preprocess) Names of the source fields holding the ground-truth semantic
        label, the tree/instance id, and the intensity. ``None`` (default) auto-detects
        from the built-in candidates (semantic: ``Class``/``semantic_seg``/
        ``classification``; tree id: ``tree_ID``/``treeID``; intensity:
        ``intensity``/``scalar_Intensity``/...). Set explicitly for data whose
        ground truth lives in a differently named field; a configured name missing
        from a file fails that plot loudly (instead of silently writing zeros).
    restore_dir:
        (restore) Folder of classified ``*.ply`` files to restore. **Required** in
        restore mode. Files already named ``restored_*`` are ignored.
    offsets_dir:
        (restore) Where the ``<plot>_offsets.yml`` files live. ``None`` = ``output_dir``.
    restore_format:
        (restore) ``"laz"`` (default) = compressed LAS 1.4 with the source scales and
        CRS re-applied and all extra fields (labels, predictions) kept as extra-bytes
        dimensions; ``"las"`` = the same, uncompressed; ``"ply"`` = plain PLY.
    plots:
        Optional list of plot names (e.g. ``["plot_01"]``) to restrict the run.
        ``None`` processes everything found.
    overwrite:
        If ``False``, a plot whose output file already exists is skipped.
        ``True`` (default) recomputes and overwrites.
    dry_run:
        If ``True``, log what would be done without reading point data or writing files.
    continue_on_error:
        If ``True`` (default), keep going after a plot fails; otherwise stop at the
        first error.
    """
    if mode not in ("preprocess", "restore"):
        raise ValueError(f"mode must be 'preprocess' or 'restore', got {mode!r}")

    wanted = set(plots) if plots else None
    succeeded: List[str] = []
    failed: List[str] = []
    skipped: List[str] = []

    if mode == "preprocess":
        root = Path(input_dir)
        if not root.is_dir():
            raise NotADirectoryError(f"input_dir does not exist: {root}")
        out_dir = Path(output_dir)
        patterns = list(patterns) if patterns else ["*.las", "*.laz", "*.ply"]

        src_files = _collect_inputs(root, patterns)
        if wanted is not None:
            selected = [p for p in src_files if _plot_name(p) in wanted]
            for name in sorted(wanted - {_plot_name(p) for p in selected}):
                log.warning("Requested plot %r matched no file in %s", name, root)
            src_files = selected
        if not src_files:
            log.warning("No clouds matching %s in %s (flat or per-plot) — nothing to do.",
                        patterns, root)
            return
        log.info("Found %d cloud(s) matching %s in %s", len(src_files), patterns, root)

        seen: Dict[str, Path] = {}
        for src_path in src_files:
            plot = _plot_name(src_path)
            if plot in seen:
                log.warning("[%s] duplicate plot name: %s already processed from %s — skipping %s",
                            plot, plot, seen[plot].name, src_path)
                skipped.append(plot)
                continue
            seen[plot] = src_path

            out_ply = out_dir / f"{plot}.ply"
            if not overwrite and out_ply.exists():
                log.info("[%s] %s already exists and overwrite=False — skipping", plot, out_ply.name)
                skipped.append(plot)
                continue
            if dry_run:
                log.info("[%s] DRY RUN: would center %s -> %s + %s_offsets.yml in %s",
                         plot, src_path.name, out_ply.name, plot, out_dir)
                succeeded.append(plot)
                continue

            out_dir.mkdir(parents=True, exist_ok=True)
            try:
                _preprocess_one(src_path, plot, out_dir,
                                semantic_field, tree_id_field, intensity_field)
                succeeded.append(plot)
            except Exception:
                log.exception("[%s] FAILED", plot)
                failed.append(plot)
                if not continue_on_error:
                    log.error("continue_on_error=False — stopping after first failure.")
                    break

    else:  # mode == "restore"
        if not restore_dir:
            raise ValueError("restore_dir is required when mode='restore'")
        rdir = Path(restore_dir)
        if not rdir.is_dir():
            raise NotADirectoryError(f"restore_dir does not exist: {rdir}")
        off_dir = Path(offsets_dir) if offsets_dir else Path(output_dir)
        if restore_format not in ("ply", "las", "laz"):
            raise ValueError(f"restore_format must be 'ply', 'las' or 'laz', got {restore_format!r}")

        ply_files = [p for p in sorted(rdir.glob("*.ply"))
                     if not p.stem.startswith("restored_")]
        if wanted is not None:
            ply_files = [p for p in ply_files if _restore_plot_name(p) in wanted]
        if not ply_files:
            log.warning("No PLY files to restore in %s — nothing to do.", rdir)
            return
        log.info("Found %d PLY file(s) to restore in %s (offsets from %s)",
                 len(ply_files), rdir, off_dir)

        for ply_path in ply_files:
            plot = _restore_plot_name(ply_path)
            offsets_path = off_dir / f"{plot}_offsets.yml"
            if not offsets_path.is_file():
                log.warning("[%s] no offset file at %s — skipping", plot, offsets_path)
                skipped.append(plot)
                continue

            out_path = rdir / f"restored_{plot}.{restore_format}"
            if not overwrite and out_path.exists():
                log.info("[%s] %s already exists and overwrite=False — skipping", plot, out_path.name)
                skipped.append(plot)
                continue
            if dry_run:
                log.info("[%s] DRY RUN: would restore %s (+ offsets from %s) -> %s",
                         plot, ply_path.name, offsets_path.name, out_path.name)
                succeeded.append(plot)
                continue

            try:
                with offsets_path.open() as fh:
                    offsets = yaml.safe_load(fh)
                _restore_one(ply_path, plot, offsets, out_path)
                succeeded.append(plot)
            except Exception:
                log.exception("[%s] FAILED", plot)
                failed.append(plot)
                if not continue_on_error:
                    log.error("continue_on_error=False — stopping after first failure.")
                    break

    # --- summary -----------------------------------------------------------
    log.info("=" * 60)
    if dry_run:
        log.info("DRY RUN (%s) — would process %d plot(s); %d skipped.",
                 mode, len(succeeded), len(skipped))
    else:
        log.info("%s: processed %d plot(s): %d ok, %d failed, %d skipped.",
                 mode, len(succeeded) + len(failed), len(succeeded), len(failed), len(skipped))
        if failed:
            log.warning("Failed: %s", ", ".join(failed))
    if skipped:
        log.warning("Skipped: %s", ", ".join(skipped))
