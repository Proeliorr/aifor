# Development history

Chronological log of the pipeline's functionality — what was added when, and why.
Dates follow the Hydra run logs in `outputs/<date>/<time>/`.

---

## 2026-07-04 — Stage 1: 3DFin batch runner

- **`main_pipeline.py` + `pipeline/threedfin.py`**: batch driver for the 3DFin v0.6.0
  CLI over a folder of forest LiDAR plots (`plot_01.laz` … `plot_14.laz`, each paired
  with its `3DFin_settings/<plot>.ini`). Produces one instance-segmented cloud per plot
  (`<plot>_tree_ID_dist_axes.las`, per-point `tree_ID` + `dist_axes`).
- Works around two 3DFin quirks: a **patched `.ini` copy** per plot (3DFin validates
  `[misc] output_dir` as an existing directory) and **CLI flags derived from each
  plot's `.ini`** (`--normalize`/`--denoise`/`--export_txt`).
- `prune_outputs` keeps only the needed `.las` + log out of ~10 files 3DFin emits.
- hydra-zen configuration pattern established: one shared `conf/config.yaml`, one
  section per stage, one entry script per stage, per-run logs under `outputs/`.

## 2026-07-05 — Tree splitter (per-tree numpy files)

- **`tree_splitter.py` + `pipeline/tree_splitter.py`** added as Step 2: sorts each
  plot cloud by `tree_ID`, assigns the sequential ordinal `2tree_ID` = 1…K, and saves
  one `.npy` per tree (`plot_01_001.npy` …) next to the input.
- Run over all 14 plots: 1,357 per-tree files (columns `x, y, z, Class`).

## 2026-07-06 — Pivot to plot-level semantic segmentation; tree splitter → `misc/`

- **Project scope fixed**: the main approach is semantic segmentation resolved at
  whole-plot level — per-tree data is not needed in the pipeline.
- Tree splitter moved out of the pipeline into **`misc/`** as a self-contained tool
  (own entry + engine merged into `misc/tree_splitter.py`, own `misc/conf/config.yaml`,
  same `tree_splitter.` override prefix). Pipeline package and shared config carry no
  per-tree code anymore.

## 2026-07-06 — Stage 2: ForAINet preparation (`forainet_prep`)

- **`forainet_prep.py` + `pipeline/forainet_prep.py`**: port of the standalone
  `Points2ForAINet.py` logic into the pipeline pattern, with two modes:
  - `preprocess` — **centers coordinates** (subtracts per-plot min x/y/z so all
    coordinates are ≥ 0, ForAINet's requirement) and writes ForAINet-ready binary PLYs
    (`x, y, z, intensity, semantic_seg, treeID`; labels filled from the real
    `Class`/`tree_ID` dimensions) plus a per-plot **`<plot>_offsets.yml`** holding the
    shifts and validation statistics (original mins/ranges, point count).
  - `restore` — adds the shifts back to classified PLYs after segmentation,
    validating restored minima/ranges against the recorded statistics.
- Per-plot offset files keyed by plot name replaced the original script's single
  `offset.yml` keyed by absolute paths (fragile when files move).
- The old `Points2ForAINet.py` moved to `misc/` as a generic ad-hoc converter.
- Verified on plot_02 (23.8M points): mins exactly 0 after centering, 20 unique tree
  IDs, restore round-trip exact.

## 2026-07-06 — Stage 2 generalization: format-agnostic input, faithful restore

- **Any input format**: `.las` (versions 1.2–1.4), `.laz` and `.ply` are processed
  identically; inputs may sit directly in `input_dir` (flat folder) or one level down
  in per-plot subfolders (the 3DFin tree). `patterns` config key (list) replaced the
  single `pattern`.
- **Source metadata recorded** in `<plot>_offsets.yml` for faithful restoration:
  source format/file, LAS version, point format, **scales**, and the **CRS as WKT**
  (closing the gap that PLY files carry no coordinate-system information).
- **Restore rewritten**: default output is **compressed LAZ (LAS 1.4)** with the source
  scales and CRS re-applied, and *every* extra vertex field (`semantic_seg`, `treeID`,
  predictions, …) preserved as LAS extra-bytes dimensions — nothing is dropped anymore.
- Verified: bit-exact LAS→PLY→LAZ round trip on plot_02 (scales 1e-7 restored); CRS
  WKT (EPSG:2180) survived byte-identical through a synthetic preprocess→restore cycle.

## 2026-07-06 — Adjustable ground-truth field names

- New config keys **`semantic_field` / `tree_id_field` / `intensity_field`**:
  `null` (default) auto-detects from candidate lists (`Class`/`semantic_seg`/
  `classification`; `tree_ID`/`treeID`; `intensity`/`scalar_Intensity`/…); a string
  forces that exact source field — for datasets whose ground truth lives under a
  different name. A configured name missing from a file **fails that plot loudly**
  (listing the available fields) instead of silently writing zero labels.

## 2026-07-06 — Housekeeping

- `.gitignore`: excludes the large binary data (input clouds, Stage 1 outputs,
  Stage 2 `.ply`/`.las`/`.laz`) while keeping the small `<plot>_offsets.yml`
  metadata files trackable.
- `requirements.txt` extended with `lazrs` (LAZ backend), `plyfile`, `PyYAML`.

## 2026-07-13 — Split point-cloud viewer (diagnostic tool, `misc/`)

- **`misc/view_split_point_cloud.ipynb`**: tkinter GUI (launched from a notebook cell)
  comparing **two point clouds side by side** — five spatial views each (XY top,
  XZ front, YZ side, ISO 45°, ISO 135°), per-field statistics (dtype/min/max/unique),
  colouring by any field (≤ 20 unique ints → discrete legend, else viridis colourbar),
  and per-cloud filtering by **value set** (`Class in {2,3}`) or **min/max range**.
  Built to diagnose which fields/labels should be merged — groundwork for the planned
  `class_map` reclassifier.
- Reads `.las`/`.laz`/`.ply`/`.npy`; bare `.npy` matrices get column names from a
  **JSON sidecar** with the same stem (`plot_01_007.json`, `{"columns": [...]}` +
  optional metadata keys shown in the info box; missing/mismatched sidecar → fallback
  names `x, y, z, col_3…` with a visible warning). Sidecars will be produced by a
  separate export script (later stage).
- Random downsample after filtering (default 100k points, fixed seed) keeps
  multi-million-point plots responsive; matplotlib toolbar gives zoom/pan/save-PNG.
- `matplotlib` added to the `aifor` env (pip user-install; conda cache in
  `C:\ProgramData` not writable). Verified: 27 headless loader/filter/projection
  tests + GUI smoke test against plot_01 (18M points, 119 tree IDs).

## 2026-07-13 — Class unifier stage (`class_unifier`)

- **`class_unifier.py` + `pipeline/class_unifier.py`**: new stage in the standard
  hydra-zen pattern (`class_unifier:` section in `conf/config.yaml`). For every input
  cloud (`.las`/`.laz`/`.ply`, flat or one level down — same collection as Stage 2):
  finds the semantic-label column (`semantic_field`, `null` = auto-detect), creates a
  **new column** (`unified_field`, default `semantic_seg`) with the labels sent through
  **`class_map: {source: unified}`** — several sources sharing one target **merges**
  classes (`{1: 2, 4: 2}` folds 1 and 4 into 2); `unmapped_value` controls values
  absent from the map (keep-with-warning or force). Default map = identity ({0..4}),
  default `class_names` = ForAINet's five classes.
- Exports `<plot>.npy` (plain `N×7` float64: `x, y, z, intensity, <source label>,
  <unified label>, <tree id>`) + **`<plot>.json` sidecar** (same stem): `columns` +
  `dtypes` (numpy-restorable names), source file/format, the map used, per-class
  point counts before/after — directly loadable by `misc/view_split_point_cloud.ipynb`.
- Reuses Stage 2 plumbing (`_read_cloud`, `_collect_inputs`, `_plot_name`) — no
  duplicated readers.
- Hydra gotcha found while testing: CLI dict overrides **merge** with the config map
  (`class_unifier.class_map={1: 2}` keeps the other keys); full replacement needs the
  keys spelled out.
- Verified: identity run on plot_01 (20,831,953 pts, counts unchanged
  {0: 1647848, 1: 4770862, 2: 8430699, 3: 5974506, 4: 8038}); merge run
  1,2→2 gives 13,201,561; synthetic PLY covers both `unmapped_value` branches;
  viewer loads the npy+json with names restored and zero warnings.

## 2026-07-24 — Viewer diagnostics: histogram, unique-values, wheel-zoom, field picker

- **`misc/view_split_point_cloud.ipynb`** gained five inspection features (motivated by
  wide id fields like `tree_ID`, 0…~60 000, where the existing stats gave no way to see
  which ids are real):
  - **Histogram** pop-up and a sortable **Unique values** table (value / count / %) for
    the current *Colour by* field; each has a *respect active filter* toggle
    (off = whole cloud). The unique-values table is capped at 100k rows (suggests a
    range filter beyond that).
  - **Mouse-wheel zoom-to-cursor** on any of the five spatial views (toolbar Home resets).
  - **Fields shown…** picker choosing which fields the statistics box lists (all shown
    by default, untick the noise).
  - Info-box header line now carries the **LAS version + point-format id**
    (`LAS/LAZ  1.4 (PF 6)`), read via `str(las.header.version)` / `las.point_format.id`
    — same idiom as `pipeline/forainet_prep.py`.
- **Window-resize fix**: the canvas is hosted in a `pack_propagate(False)` frame so it
  can shrink after the first `draw()`. Previously the canvas pinned the window to the
  figure's pixel size, so resizing "stopped working" once a cloud was loaded;
  `root.minsize(900, 600)` sets a floor.
- `summarize_fields` / `cloud_info_text` gained an optional `fields` subset argument
  (`None` = all, previous behaviour); loaders now carry a `header` dict.

## 2026-07-24 — Viewer 3-D view (PyVista, separate process)

- **`misc/view_cloud_3d.py`** (new) + a **3D view** button per panel in
  `misc/view_split_point_cloud.ipynb`: opens the **currently filtered selection** in a
  real OpenGL window — left-drag rotate, scroll zoom, middle-drag pan, `r` reset,
  `d` depth-shading toggle, `q` close. 2-D remains the default view.
- **Open3D could not be used**: the `aifor` env is Python 3.13 and Open3D publishes no
  cp313 wheels (`pip install open3d` → *no matching distribution*). **PyVista 0.48.4 +
  VTK 9.6.2** (cp313 wheels, `requires_python >=3.10`) replace it.
- **Runs as a separate process** (temp `.npz` handed over, child deletes it): VTK and
  tkinter each want the event loop, so nesting them freezes the 2-D window. As its own
  process the 2-D viewer stays live and Cloud A + Cloud B can both have a 3-D window
  open. The child's stdout/stderr go to a `.log` next to the payload so a GPU/VTK
  failure is diagnosable instead of silent.
- Own point budget (**3D max pts**, default 500 000) separate from the 2-D
  `Max points`; both go through the new shared `_selected_indices()` with the same
  fixed seed, so the 3-D view is a superset of the 2-D subsample.
- Colour logic extracted into **`color_spec()`** (cell-3) and now used by *both* the
  2-D scatter and the 3-D export, so the two can't drift: tab20 + legend for ≤ 20
  integer labels, viridis + scalar bar otherwise.
- **Eye-dome lighting defaults OFF**: measured on the off-screen renders it crushed
  mean point brightness from ~104 to ~6 (sparse) / ~125 to ~33 (dense), i.e. a nearly
  black cloud. Kept as the `d` toggle instead (not `e` — PyVista binds that to exit).
- Verified headlessly: notebook cells exec; `color_spec` discrete/continuous/wide-id
  branches; `_selected_indices` mask + budget; 2-D `redraw()` still fine after the
  refactor; and `view_cloud_3d.py --smoke` off-screen renders for **both** colouring
  branches (PNGs inspected, payload cleanup and a missing-payload exit code checked).

---

## Planned

- **In-PLY reclassification (optional)**: the `_semantic_labels()` hook in
  `pipeline/forainet_prep.py` is still a straight copy; if remapped labels should be
  written directly into the Stage 2 PLYs (instead of / in addition to the standalone
  `class_unifier` stage), a `class_map` key can plug in there the same way.
- Optional: `pyproj` in the env would let `crs_wkt` capture also GeoTIFF-key CRSs
  (LAS 1.2/1.3 style); WKT-based CRSs (mandatory in LAS 1.4) already work without it.
