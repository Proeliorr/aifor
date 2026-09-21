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

## 2026-07-27 — SegmentedForests → ForAINet class scheme (4 classes, strict, shared)

- **The mapping is live in both stages.** A new shared `classes:` block in
  `conf/config.yaml` defines the scheme once; `class_unifier:` and `forainet_prep:`
  interpolate it, so the `.npy` inspection exports and the PLYs ForAINet actually trains
  on cannot drift. The rule lives in the new **`pipeline/classes.py`** — a neutral module
  because `class_unifier` already imports the reader from `forainet_prep`, so the reverse
  import would be circular.
- **Map** (`Class` → `semantic_seg`): `8,9,11 → -1` (dropped) · `0,4,5,6,7,12,13,22,23 → 0`
  · `1 → 1` · `3,10 → 2` · `2 → 3`. Verified before implementing: the union of `Class`
  over all 14 plots is exactly those 16 values, no gaps or overlaps. Per-plot vocabularies
  differ a lot (`plot_01`: `0–4`; `plot_10`: `0–5,11,12,13,22,23`), which is why an early
  identity run on `plot_01` had looked like a tidy 5-class dataset.
- **`-1` removes points**, it is not an ignore label. In Stage 2 the drop happens *before*
  the centering minima are computed, so `<plot>_offsets.yml` describes the points actually
  in the PLY — `restore` validates against those, and would otherwise fail. New keys
  `n_points_source` / `n_points_dropped` record the difference; the class-unifier sidecar
  gains `dropped_points` / `dropped_source_values`.
- **Strict by design**: `unmapped_value` is gone. A source class missing from `class_map`
  raises and fails that plot, naming the offending values. Guessing (keep-as-is, or force
  to a default) would put a bogus class into training.
- **ForAINet is now a 4-class problem.** Class 4 `branches` has no SegmentedForests
  counterpart, so `Treeins_NUM_CLASSES 5→4`, `INV_OBJECT_LABEL`/`CLASSES_INV`/`OBJECT_COLOR`
  lose it, `VALID_CLASS_IDS → [0,1,2,3]`, and `SemIDforInstance [2,3,4] → [2,3]` (tree
  instances now come from stem_points + live_branches only). The panoptic eval counters are
  1-based and shifted, so `NUM_CLASSES_sem 6→5`, `sem_classcount → [1,2,3,4]`,
  `thing_classes → [3,4]`. Deliberately left alone: `NUM_CLASSES = 3` (it counts the binary
  stuff/thing scheme, independent of the fine class count) and the dead, never-referenced
  `NUM_CLASSES_count`. Edits captured in `patches/forainet-local.patch` (now 5 files) and
  the patch was proven to reapply onto a pristine submodule checkout.
- **Hydra gotcha corrected by measurement.** The old note ("CLI dict overrides merge")
  was only half right. With integer keys: `classes.class_map.5=1` and
  `classes.class_map={99: 0}` both fail with *"Key ... is not in struct"*;
  `++classes.class_map={99: 0}` merges; and `class_unifier.class_map={...}` **replaces**
  that stage's map, severing the link to the shared block so Stage 2 quietly keeps the old
  one. Change the scheme by editing the file.
- Verified on `plot_10` (richest vocabulary, 42,511,997 pts): strict failure on an
  incomplete map writes nothing; exact conservation (every source count lands in the right
  bucket, e.g. 7 sources → class 0 = 15,228,253); 269,800 points of class 11 dropped;
  `.npy`, `.ply` and `offsets.yml` all agree at 42,242,197 points with `semantic_seg`
  `{0,1,2,3}` as `uint8`, coordinates ≥ 0 with minima ≈ 0; restore round-trips with no
  warning.

## 2026-07-27 — class_unifier `output_format` (.ply / .npy / both), default .ply

- **`output_format: ply | npy | both`** added to the `class_unifier:` section; the
  `.json` sidecar is written either way and now records `output_format` + `files`.
- **Why the default changed to `ply`.** The old `N×7` float64 `.npy` was actively hard to
  inspect in `misc/view_split_point_cloud.ipynb`: the viewer decides colouring with
  `discrete = np.issubdtype(dtype, np.integer) and n_unique <= 20`, which a float64
  column can never satisfy — so `Class`/`semantic_seg`/`tree_ID` drew as a viridis ramp
  with no legend, `summarize_fields` printed `-` in the *Unique* column, and the
  histogram fell back to generic bins. The stage exists to check a class merge, and its
  own export format defeated exactly that.
- The PLY carries the **same 7 columns** (source label kept next to the unified one, so
  one file supports the before/after comparison) with each column's **native dtype** —
  verified on plot_10: `intensity` u2, `Class` i1, `semantic_seg` u1, `tree_ID` i4.
  `color_spec` now returns `discrete=True` for both label columns (4 and 10 classes) and
  the statistics table shows real unique counts. Also smaller: **1289 MB vs 2256 MB**
  (57%), ~32 vs 56 bytes/point.
- Coordinates stay **original** (no centering): that is Stage 2's job, together with the
  `<plot>_offsets.yml` that makes it reversible. This export is diagnostic, not a second
  ForAINet producer.
- Reuses Stage 2's `_write_ply`; the two writers are split into `_write_ply_cloud` /
  `_write_npy_matrix`. `overwrite=false` now tests the files for the *selected* format
  (and for `both` only skips when both exist, so a half-finished run completes), and a
  leftover file from a previous run in the other format is warned about — the shared
  sidecar would otherwise let the viewer open stale data.
- Verified: invalid format rejected; plot_10 as `.ply` (42,242,197 pts, `semantic_seg`
  ⊆ {0,1,2,3}, `Class` still holding pre-merge values); `both` writes the pair; `npy`
  reproduces the legacy float64 layout; skip and half-finished-skip behave.

## 2026-07-27 — 3-D view: reset-view button

- **`misc/view_cloud_3d.py`** gained a **reset view** button in the bottom-left corner
  (x=220 px, clearing the orientation-axes gizmo), plus the same action on the `r` key.
  PyVista has no plain push-button, so the checkbox widget is used as one with identical
  on/off colours; the click is what matters, not the state.
- Deliberately **stronger than VTK's built-in `r`**, which refits the bounds but keeps
  whatever orientation you rotated to. Re-applying the isometric camera *before*
  `reset_camera()` makes reset land on the same view every time, so that key is rebound.
- `_add_reset_view` returns the reset callable — the very one the button and key invoke
  — so tests exercise the real closure instead of a copy.
- Widgets need a live interactor, so the button is skipped (not attempted) under
  `--smoke` / off-screen rendering; the `r` binding still applies.
- Verified: off-screen smoke unaffected; one button widget attaches; after an
  orbit+zoom the camera returns **exactly** to the opening position
  (`[125.88 128.06 127.14]` → moved → identical again); `r` confirmed bound via
  `clear_events_for_key(raise_on_missing=True)`; placement eyeballed on a screenshot.

## 2026-07-29 — Stage 2: tree ids cleared on non-tree points (instances were all being lost)

- **The defect.** ForAINet's convention is `treeID = 0` for anything that is not part
  of a tree; its own files follow it exactly (ground and low_vegetation are id 0 for
  100% of their points). 3DFin does not — it stamps every point with the *nearest
  stem's* id, so ground and undergrowth inherit the tree above them. One measured
  example on plot_10: id `114189` covered 3,046,870 points, of which 1,073,013 were
  low_vegetation and 585,295 ground.
- **Why that is fatal.** `set_extra_labels()` skips any instance whose id also appears
  on a non-thing point — a rule meant to discard the single "not a tree" bucket (id 0).
  With 3DFin ids every id looks contaminated, so **0 of 42 tree instances survived** on
  plot_10. Training would have run normally and learned instance segmentation from
  nothing.
- **The fix.** New `classes.instance_classes` (`[3, 4]` = stem_points, live_branches, in
  the same numbering as `class_names`); Stage 2 zeroes the tree id of every point
  outside those classes, via `zero_stuff_tree_ids()` in `pipeline/classes.py`.
  `null` disables it. `<plot>_offsets.yml` records `n_tree_ids_zeroed`.
- Verified on plot_10 against `NIBIO2_plot12_annotated_train.ply` as the reference,
  simulating ForAINet's own read (`semantic_seg - 1`, `treeID + 1`) and its instance
  filter: stuff classes now 100% id 0, thing classes untouched, and **42 of 42
  instances survive** (was 0 of 42). 19,568,564 ids cleared.
- Note the reference file itself reports 35 of 48 under our *4-class* thing set, because
  upstream's `branches` becomes "stuff" and takes its ids with it. That is an artifact of
  scoring 5-class data with the patched framework, not a defect in the original.

## 2026-07-29 — Parallel plot processing, bounded by memory rather than cores

- **`pipeline/parallel.py`** (new): plots are independent, so several now run at once in
  worker **processes** (the work is numpy + LAZ decompression, so threads would serialise
  on the GIL). Used by `class_unifier`, `forainet_prep` (both modes) and
  `misc/tree_splitter.py`. **Stage 1 (3DFin) stays sequential** — it only waits on an
  external binary whose own threading is unknown.
- **Cores are not the limit, RAM is.** The clouds span 19 M to 280 M points, and Stage 2
  costs a **measured ~134 bytes/point** (plot_09: 72.6 M pts → 9.1 GB peak), so plot_08
  alone needs ~37 GB. A worker per core would exhaust the machine. The scheduler reads
  each plot's point count from the LAS header (instant), sorts biggest-first, and admits
  work only while it still fits a budget (70% of RAM) — so a huge plot runs nearly alone
  while small ones pack several deep. A plot larger than the whole budget still runs, on
  its own, rather than deadlocking.
- Worker log lines are captured and replayed by the parent, so
  `outputs/<date>/<time>/*.log` keeps the same per-plot detail. Workers also get
  `OMP_NUM_THREADS=1` etc., or each would start its own BLAS pool and fight for cores.
- `workers: 1` runs everything inline — byte-for-byte the old behaviour, and the escape
  hatch if anything looks wrong.
- **Verified**: outputs are byte-identical between sequential and parallel (sha256 over
  `.ply` + `.json`); every per-plot log line survives the worker boundary; a failing plot
  is isolated with its real error text intact and the summary still reports it.
- **Measured on the full 14-plot Stage 2 run: 3.3 min → 2.4 min (~1.4×), peak RSS 39.4 GB
  against a 44.7 GB budget, 14/14 ok.** The modest speedup is honest and expected:
  plot_08 is 33% of all points and needs ~37 GB, so it runs essentially alone and sets the
  floor (Amdahl). Raising `memory_budget_frac` above 0.7 buys a little more at the cost of
  headroom.
- Two things worth recording from the debugging, because both looked like code defects and
  were not: an early "worker OOM" was actually **the system drive filling up** with my own
  benchmark output (a 14-plot run is ~25 GB, and the default temp dir is on C:); and the
  first `bytes_per_point` guess of 60 counted only the obvious arrays, missing that
  `plyfile.write()` does `data.astype(...).tobytes()` — two more full copies.

## 2026-07-29 — Stage 2 assigns train / val / test via the file name

- **Why it was needed.** ForAINet has no split manifest — it reads the set from the FILE
  NAME (`name[-7:-4] == "val"`, then `name[-8:-4] == "test"`, everything else training).
  Stage 2 wrote plain `<plot>.ply`, so **every plot landed in training** and there was no
  validation or test set at all. Output is now `<plot>_train.ply` / `_val.ply` /
  `_test.ply`, matching the reference dataset (`old_training_dataset/NIBIO2`: 29 train,
  15 test, 6 val — and `_val`, never `_eval`).
- **The rule** (`assign_splits()`): plots ranked by point count **descending**, the
  smallest held out — most points stay in training. New `forainet_prep.split` block takes
  `n_val`/`n_test` as either a count (`int`) or a fraction of the plots (`float` < 1,
  rounded, min 1), so the same config fits a dataset of any size. On the 14 plots the
  defaults `1`/`2` give **plot_11 → val, plot_14 + plot_01 → test, 11 train**.
  Ties break on plot name, so equal-sized plots never swap roles between runs; a split
  leaving zero training plots raises rather than producing an empty training set.
- **Computed over every matched cloud, before the `plots` filter** — otherwise
  `plots=[plot_01]` would make plot_01 "the smallest" and mark it validation. Verified:
  running plot_01 alone still writes `plot_01_test.ply`.
- **Stale siblings are removed.** A plot can change split (a new plot shifts the ranking,
  or `n_test` changes), and ForAINet globs `raw/**/*.ply` — leaving the old file would
  load that plot **twice, in two different splits**, training on its own test data. The
  old file is deleted with a warning.
- `<plot>_offsets.yml` keeps the plain plot name (it describes geometry, not a split) and
  gains `split` + `ply_file`. `_restore_plot_name` now strips the suffix too, so
  `plot_11_val.ply` still resolves to `plot_11_offsets.yml` — verified by a restore
  round-trip.
- `split: null` reproduces the previous `<plot>.ply` naming exactly.
- Verified end to end: every generated name fed back through **ForAINet's own**
  `[-7:-4]`/`[-8:-4]` test returns the split we intended; fractions `0.07`/`0.14` on 14
  plots reproduce the `1`/`2` assignment, and `0.1`/`0.2` on 50 plots give 5 val / 10
  test / 35 train.

---

## 2026-07-30 — The chain: stages run per plot, intermediates deleted as they are consumed

- **Why it was needed.** Run as separate stages, each handed the next a *complete*
  folder, so the disk cost was the sum of every stage at once. Measured for a **4.4 GB
  input**: `3DFin_output/` 46.3 GB + `ClassUnifier_output/` 25.3 GB + a stale
  `ForAINet_input/` 29.2 GB + `raw/` + `processed_0.2/` = **108.6 GB**. 3DFin wrote all
  fourteen `.las` files before Stage 2 read the first one, and each became dead weight
  the moment Stage 2 consumed it.
- **`pipeline/chain.py` + `main_pipeline.py` now drive the whole pipeline per plot**:
  3DFin → Stage 2 → delete that plot's `.las`. Transient cost falls from all fourteen
  intermediates (46.3 GB) to the largest single one (**15.2 GB**, plot_08). 3DFin is an
  external CLI and can only hand over a file, so consuming it immediately is as close to
  "not exporting between stages" as the tool allows.
- **The chain adds no processing.** It calls `run_pipeline()` and `prep_forainet()` —
  the same functions the standalone entry scripts call — reading the same `threedfin:`
  and `forainet_prep:` sections, so a chained run and a stage-by-stage run cannot drift
  apart. Verified: the chain's `plot_11_val.ply` is **byte-identical** (SHA-256) to the
  one the old two-step route produced, offsets file included.
- **The split had to be decided up front.** Stage 2 ranks plots by point count and holds
  out the smallest, but the chain feeds it one plot at a time — which could not be
  ranked against anything. So the ranking is taken from the **source** `.laz` headers
  before 3DFin runs and passed down as the new `split_of` parameter. Sound because 3DFin
  preserves point counts exactly (checked across all 14 plots, zero delta); the chain
  reproduces `plot_11 → val, plot_14 + plot_01 → test`.
- **Deletion is conservative.** Only after Stage 2 reports that plot as *succeeded*;
  only point-cloud files (`3dfin_log.txt` survives, so the folder still records the run);
  and never when Stage 2 is not in `stages`, since then the `.las` *is* the output.
  Verified by forcing a Stage 2 failure — the 15-minute-to-regenerate file stayed put and
  the log said why.
- **New `chain:` config section**: `stages` (either stage alone still works),
  `keep_intermediates` (`false` / `true` / **a list of plot names**, which is how the
  opt-in `class_unifier` diagnostics stay practical — it needs the `tree_ID` only 3DFin
  produces), `plots`, `overwrite` (defaults to `false`, so an interrupted run resumes
  instead of repeating hours of 3DFin), `dry_run`, `continue_on_error`.
- Both stage engines now **return** their `{succeeded, failed, skipped}` tallies (they
  already built them for the summary line) and take `log_summary=False`, which drops the
  run-level framing when they are called once per plot — otherwise a 14-plot chained run
  logged 28 summary blocks.
- **Breaking change**: `python main_pipeline.py` used to mean "3DFin only". That is now
  `chain.stages=[threedfin]`; the bare command runs the whole chain.
- Cleanup after the change: the 46.3 GB of `3DFin_output` clouds were consumed by a
  `chain.stages=[forainet_prep]` run that regenerated all 14 training PLYs, and the stale
  `ClassUnifier_output/` and `ForAINet_input/*.ply` were removed (the tracked
  `*_offsets.yml` kept).

## 2026-07-30 — `conf/config.yaml` slimmed to a settings file; keys documented in the README

- **Why.** The config had reached **366 lines for ~70 settings** — every rationale,
  measurement and gotcha found while building the pipeline had accumulated in it. Changing
  `n_val` meant scrolling past a 20-line essay on ForAINet's `ignore_index` semantics.
- **Now 120 lines**: values, a one-line header per section, and a short inline hint only
  where a value is not self-explanatory (`null` meanings, legal enum values). Verified as a
  comment-only change — the resolved config (`OmegaConf.to_container(..., resolve=True)`)
  is identical before and after, key for key.
- **README gains a `## Configuration reference` section**: one subsection per config block
  with a `Key | Default | What it does` table, plus the run-control keys
  (`plots`/`overwrite`/`dry_run`/`continue_on_error`) and parallelism keys
  (`workers`/`memory_budget_frac`/`bytes_per_point`) documented **once** instead of in four
  places. The three scattered key tables it replaces were deleted, so there is exactly one
  description of each key. A scratch cross-check asserts all 47 config keys appear there
  and nothing is documented that does not exist.
- The rationale worth keeping moved with it: why `forainet_raw` is not a free choice, the
  Hydra integer-key gotcha, why `drop_value` is inert, why `instance_classes` exists, and
  the RAM-not-cores concurrency model.
- **Stale documentation fixed while reading** (each already contradicted the code):
  the class-mapping table was still 0-based and claimed classes 8/9/11 were *removed*
  (they map to `0` = unclassified and nothing is dropped — `n_points_dropped` is 0);
  Stage 2 was described as writing to `SegmentedForests/ForAINet_input/<plot>.ply`
  (it writes `${paths.forainet_raw}/<plot>_<split>.ply`); a "**Planned:** reclassification
  via a `class_map` key" note survived long after that shipped; and two links pointed at
  `progress.md` and a `ForAINet/ARCHITECTURE_ANALYSIS.md` that no longer exist.

## 2026-07-30 — Stage 2 exports LAZ; conversion to PLY moves into the training container

- **Why.** Training moves to a rented GPU, so the clouds have to be uploaded — and a
  binary PLY is an uncompressed memory dump. Measured: `plot_11_val.ply` is **668.5 MB =
  exactly 37.00 bytes/point** (`f8×3 + f8 + u1 + u4`), and the 14 plots come to
  **29.21 GB from 4.41 GB of source**. Every one of those bytes would have crossed the
  network.
- **`forainet_prep.output_format`** (`laz` | `las` | `ply`, default **`laz`**) reuses the
  `_write_las` the restore path has always used, so the labels travel as LAS extra-bytes
  dimensions and the source scales and CRS are re-applied. New `paths.export`
  (`SegmentedForests/ForAINet_export`) is the new `output_dir` default; `output_format=ply`
  paired with `${paths.forainet_raw}` reproduces the old behaviour exactly.
- **Measured over all 14 plots: 29.21 GB → 4.32 GB, 6.8×, saving 24.9 GB of upload.**
  Per-plot ratios run 4.0× to 11.9× and track the source coordinate scale, because LAZ
  compresses integer deltas: the 1e-07 plots reach ~4×, 1e-06 ~8×, and plot_13 at 1e-05
  hits 11.9×. Kept the source scales rather than quantising, so the export stays bit-exact
  against the source coordinates.
- **`pipeline/convert.py`** — new, and deliberately dependency-light: `laspy`, `lazrs`,
  `numpy`, `plyfile`, `yaml` and stdlib, **no Hydra**, verified by asserting nothing from
  `hydra`/`omegaconf`/`torch` enters `sys.modules` on import. Runs as
  `python -m pipeline.convert --to ply <in> <out>` in the container, as `python convert.py`
  through the new `convert:` config section, or via `misc/Points2ForAINet.py`.
- **It is a pure format conversion** — no centring, no class remapping, no tree-id zeroing.
  All of that stays in Stage 2, whose output already carries centred coordinates (LAS
  header offset 0), the unified `semantic_seg` and the zeroed `treeID`. The **file stem is
  preserved**, which is load-bearing: ForAINet reads the split from the file name, so
  `plot_11_val.laz → plot_11_val.ply` and a prefix would silently move a plot between sets.
- **Round trip verified on plot_11** (18.9 M points): `semantic_seg`, `treeID` and
  `intensity` **bit-identical**; largest coordinate deviation **3.6e-15 m** — float64
  rounding, not quantisation, because the centred coordinates already sat exactly on the
  1e-07 grid. The `output_format=ply` path still produces a byte-identical file (SHA-256).
- **`misc/Points2ForAINet.py` rewritten as a thin CLI** over `pipeline.convert`. Its own
  `array_to_las` wrote only `x/y/z/intensity`, so it would have **silently discarded
  `semantic_seg` and `treeID`** — it could never have done this job. The `dummy_`/
  `restored_` prefixes and the path-keyed `offset.yml` retire with it.
- Knock-on fixes: the `overwrite` skip, the stale-sibling sweep (now covering every split
  **and** every format, so switching format cannot strand a `.ply` beside a new `.laz`) and
  the chain's resume check are all extension-aware; `<plot>_offsets.yml` gains
  `output_file` alongside the legacy `ply_file`; and Stage 2 warns if a non-PLY export is
  written into a ForAINet `raw/` tree, which globs `*.ply` only and would train on nothing.

## 2026-07-31 — Finished the 4-class conversion; documented the evaluation path

- **Why now.** `docs/train_logs.md` is from 2026-07-21 and shows
  `iou_per_class = {0,1,2,3,4}` — **five** classes. The 5→4 patch landed on 2026-07-27, so
  the patched pipeline had never actually been run. Finding that out on a rented GPU would
  be expensive.
- **Audit result: the conversion was functionally complete but had two loose ends.**
  `dataset.num_classes` is the single source (`segmentation/treeins_set1.py:211`), read by
  the model head (`PointGroup3heads.py:80`) and the tracker (`self._num_classes`), so
  nothing structural was wrong. The two artefacts still saying "5":
  - `NUM_CLASSES_count = 5` in `panoptic/treeins_set1.py` — dead in *that* copy, but the
    sibling copies do `mIoU = sum(iou_list) / NUM_CLASSES_count` (`treeins.py:187`,
    `npm3d_4class.py:186`). A wrong value in a variable that elsewhere divides the headline
    metric is a copy-paste landmine. Now 4.
  - `path_pretrained` in `FORpartseg_3heads.yaml` pointed at a **5-class** checkpoint on the
    author's cluster. Verified from `train_logs.md:11` that it only ever logged *"The path
    does not exist, it will not load any model"* — every run trained from scratch by
    accident. And had it resolved, `load_state_dict_with_same_shape(strict=False)`
    (`base_model.py`) would have **silently skipped** the mismatched 4-vs-5 semantic head.
    Now `null`, so "from scratch" is a decision.
- **New `misc/check_forainet_classes.py`** — re-derives all 14 class constants from the
  `classes:` block of `conf/config.yaml` and checks the submodule against them, by
  *parsing* the source (no torch needed, runs anywhere). It exists because a submodule
  stores only a SHA: `git submodule update` reverts the patch, and nothing crashes — the
  model just grows a fifth output no label can reach while `final_eval` divides by the
  wrong number. Verified by stashing the patch: the checker reported **12 mismatches**,
  including `path_pretrained`; re-applying restored a byte-identical tree and it passed.
- **`patches/forainet-local.patch` regenerated** — 6 files now (the model config joined).
  `patches/README.md` was claiming three; corrected, with the `.pyc` exclusion the
  regenerate command needs and a note that untracked files in the submodule are not
  captured at all.
- **New `docs/eval_process.md`** — traces `eval.py` (18 lines) through `Trainer.eval` →
  `_test_epoch` → `tracker.finalise` → `dataset.final_eval`, and defines all 44 statistics
  the run emits. The points worth knowing: there are **two tiers** of metrics and only
  `Evaluation_<i>.txt` is reportable (the live tracker numbers are running averages over
  subsampled cylinders); instance matching is IoU ≥ 0.5; `MUCov` vs `MWCov` is unweighted
  vs size-weighted, so the gap between them is the small-tree/large-tree story;
  `PQ = SQ × RQ`, so PQ alone cannot say which half failed; and `final_eval` uses **two
  numbering schemes twelve lines apart** (1-based with 0 = ignore, plus a binary
  unclassified/stuff/thing collapse where `NUM_CLASSES = 3`).
- Also found and documented: `conf/eval.yaml` cannot run as shipped (eleven absolute paths
  under `/cluster/work/igp_psr/binbin/…`, empty `checkpoint_dir`), and `conf/config.yaml`
  defaults to `models: panoptic/area4_ablation_2` — **a file that does not exist**, so
  `models=` must always be overridden.
- README: restored the pointer to `docs/ARCHITECTURE_ANALYSIS.md`, which a previous
  link fix had redirected to the submodule's own readme.

## 2026-08-12 — TorchSparse 1.4 available as a second backbone; MinkowskiEngine still default

- **Why.** The framework prints *"Minkowski API is deprecated in favor of the SparseConv3d
  API"* three times per run. `docs/backbone_torchsparse_1.4.md` had already worked out the
  swap; every claim in it was re-verified against the code before implementing.
- **Two model blocks, one knob.** `model_name=PointGroup-PAPER` → MinkowskiEngine (the
  default, `conf/config.yaml` untouched); `model_name=PointGroup-PAPER-TS` → TorchSparse
  1.4. Same switch for `train.py` and `eval.py`.
- **New `PointGroup3heads_ts.py`**, a copy of `PointGroup3heads.py` differing in four
  lines (the import plus `backend=` on the three `SparseConv3d(...)` constructions).
  **A copy, not an edit**, because the two applications resolve `ResNetDown`/`ResNetUp`
  against *different* modules — `applications/minkowski.py:9` imports
  `modules/MinkowskiEngine/api_modules`, `applications/sparseconv3d.py:11` imports
  `modules/SparseConv3d/modules`, and both then set `modules_lib = sys.modules[__name__]`.
  Editing in place would have changed the Minkowski path too and invalidated the baseline
  in `docs/train_logs.md`. Verified: `diff` between the two files shows only those four
  changes.
- **The yaml block is four lines**, inheriting all ~115 lines of `PointGroup-PAPER`
  through a YAML merge key (`<<: *paper`) so the two cannot drift. Verified that OmegaConf
  honours merge keys, and asserted mechanically that the two blocks differ in exactly
  `class` (value) and `backend` (key) — i.e. **`PointGroup-PAPER` is provably untouched**.
- **Hazards checked, not assumed.** The coordinate-order difference (minkowski
  `[batch,x,y,z]` vs torchsparse `[x,y,z,batch]`) bites only
  `SparseConv3dEncoder.forward`, which reads `data.C[:, 0]`; that path is unreachable at
  `scorer_type: "unet"`. `SparseConv3dUnet.forward` returns no `batch` field, which is
  safe because the model reads `.x` only and takes `batch` from the *input*. Both noted in
  the new file's comments so the next person does not have to re-derive them.
- **No image rebuild**: `torchsparse@v1.4.0` is already installed (`Dockerfile:77-78`), and
  `nn/torchsparse.py` uses the 1.4 API (`TS.SparseTensor`).
- **Deleted `ForAINet_new_backbone/`** (639 MB). It was a copy whose `.git` was a *file*
  reading `gitdir: ../.git/modules/ForAINet` — `git rev-parse --show-toplevel` inside it
  returned the **original**, so every git command there operated on the wrong tree and no
  patch could ever be produced from it. 627 MB of its bulk was copied `outputs/`. Confirmed
  byte-identical to the original before removing.
- Patch regenerated (7 files). `patches/README.md` gained the `git add -N` step the new
  file requires — without it `git diff` silently drops the file from the patch — and the
  non-destructive `git apply --check --reverse` verification, since `git stash` refuses to
  move intent-to-add entries and therefore proves nothing.

## 2026-08-12 — `docs/gpu_training_runbook.md`: local smoke test → R2 → Vast.ai → two models

- **Why.** Everything built since 2026-07-27 — the 4-class conversion, the LAZ export, the
  converter, the second backbone — was verified statically or on a single plot. Nothing has
  run end to end. `docs/train_logs.md` predates all of it and shows five classes. Finding a
  broken path on a rented A100 is an expensive way to find out.
- The document goes: **Part A** local smoke test (2 epochs per backbone on a 3-plot subset),
  **B** push the image, **C** upload to R2, **D** rent the instance, **E** two 150-epoch
  runs, **F** download the results, **G** troubleshooting, **H** time and cost.
- **Facts established while writing it**, several of which change what the reader should do:
  - The local **GTX 1660 Ti is sm_75**, inside the image's arch list `6.0;7.0;7.5;8.0;8.6` —
    so Part A is a genuine CUDA test. The **A100 is sm_80**, also supported; a **4090/L40S
    (8.9) or H100 (9.0) would be unusable**, since the list carries no `+PTX` fallback.
  - The **repo-root `Dockerfile` cannot be used** — a VSCode template on `python:3-slim`
    with no CUDA and a Windows backslash in its `CMD`. `container_export/Dockerfile.train`
    is the real one. `docker-compose.yml` *is* usable, but only locally: its
    `./ForAINet:/workspace` bind mount has nothing to bind to on a rented host.
  - **The checkpoint is 308 MB**, and `training.wandb.public: True` makes `trainer.py:192`
    copy it into the wandb directory **every epoch**, where wandb's live policy uploads it —
    ~46 GB per run over 150 epochs. `training.wandb.public=False` stops that without
    disabling wandb (`Wandb.launch` simply fires from `trainer.py:141` instead of `:76`).
    A finished run directory reports ~620 MB because `wandb/` holds a second copy.
  - `dataset_factory.py:15` resolves `dataroot` with `hydra.utils.to_absolute_path`, i.e.
    against the launch directory rather than Hydra's output directory — which is what makes
    the container data path predictable.
  - The `debugpy.breakpoint()` calls in `train.py` are already commented out, so unattended
    training will not hang.
- **Verified mechanically** (`scratchpad/check_runbook.py`): every file the reader is told
  to use exists, all four Hydra config groups resolve, both `model_name` blocks resolve to
  the right classes, the "`models=` is required" claim holds (the default really does point
  at a missing file), the data manifest is 14 `.laz` + 14 `_offsets.yml`, and every
  cross-document link resolves.

## 2026-08-24 — Cheap-GPU pre-flight gate

- **`container_export/preflight.sh` + `docs/preflight_cheap_gpu.md`**: a one-command
  PASS/FAIL gate that validates the whole container chain on a **cheap, supported GPU**
  (T4/3090, ~$0.10–0.40) before renting the A100. Motivated by the dev machine no longer
  having a local NVIDIA GPU, so `gpu_training_runbook.md` Part A ("test locally first")
  can't run there — the cheapest equivalent is a short cheap-instance rental.
- Env-var driven to match how the instance is provisioned: reads `DATASET_TRAIN_URL`
  (dataset archive), `DATASET_PATCH` (patch + converter archive), `WANDB_API_KEY`
  (optional step F). Auto-detects `.zip`/`.tar.gz`/`.tar`/`.7z` (sniffs content when the
  URL carries no extension).
- Covers exactly the two verification scopes chosen: **converter + smoke test**, and
  **training startup for both backbones**. Steps: GPU-arch check against the image's
  `6.0;7.0;7.5;8.0;8.6` kernel set (fails loudly on a 4090/L40S/H100 — no `+PTX`
  fallback), `unset SPARSE_BACKEND`, fetch+apply the 4-class patch if absent, run
  `smoke_test.py`, fetch+convert `.laz`→`.ply` via the `python3.8 -m pipeline.convert`
  CLI (note: `--plots` is **space**-separated here, unlike the Hydra `convert.plots=[...]`),
  clear `processed_0.2`, then 2-epoch runs of `PointGroup-PAPER` and `PointGroup-PAPER-TS`.
- `PREFLIGHT_DRYRUN=1` prints every command without executing (verified on the GPU-less
  dev box); `PREFLIGHT_PLOTS`, `PREFLIGHT_EPOCHS`, `PREFLIGHT_SKIP_TRAIN`, `PREFLIGHT_WANDB`
  tune scope/cost. Registered in `container_export/README.md`.

## 2026-08-27 — The pre-flight meets a real rented box: five fixes, one of them serious

First live run of `preflight.sh` on a Vast.ai **RTX 3090** ("forainet" template). It ended
`4 × FAIL` — and, far worse, **2 × OK that were false**. Every fix below is in the script;
`docs/preflight_cheap_gpu.md` gained a §2b describing the self-healing and three
troubleshooting entries.

- **The false pass (the one that mattered).** `run_sh` executed pipelines through
  `bash -c "$1"`, and a fresh `bash` does **not** inherit `set -o pipefail` from line 47. So
  `train.py … | tee log` reported *tee's* exit status: both training runs died on
  `python3.8: can't open file 'train.py'` and both were summarised as
  `[OK] … finished 2 epoch(s)`. A gate that green-lights a $16–30 A100 run after the training
  step never started is worse than no gate. Now `bash -o pipefail -c`, **and** a run only
  counts if its log contains the trainer's own `EPOCH n / m` line (`trainer.py:154`) — an
  exit status alone is not evidence.
- **The image is environment-only** — no `/workspace/PointCloudSegmentation/train.py`, because
  locally the code arrives through the `./ForAINet:/workspace` bind mount that a rented host
  cannot have. The old advice ("rebuild the image with `COPY ForAINet/ /workspace/`") is
  useless while the meter is running, so step B now *locates* the tree (6 candidate paths, then
  a bounded `find`) and, failing that, **clones `prs-eth/ForAINet` at the pinned `5fe600a`**.
  Measured: a shallow fetch of that exact SHA is **16 MB in ~1 s**, GitHub serves it, and
  `git apply` of `patches/forainet-local.patch` then succeeds on the fresh tree
  (`Treeins_NUM_CLASSES = 4`, `PointGroup3heads_ts.py` present) — all verified end-to-end
  locally. Because a clone lands at `/workspace/ForAINet/`, one level deeper than the bind
  mount, `RAW`/`CACHE`/`SEG_DS`/`TS_MODEL` are no longer constants: `set_paths()` re-derives
  them from wherever the tree was found, and the summary prints all three resolved paths.
- **Archive layout.** The uploaded `container_export.tar.gz` nests everything under
  `container_export/` (packed as `tar -czf x.tgz container_export`, not `-C container_export .`),
  so the converter, the patch and `smoke_test.py` were all "missing" at `/opt/prep/…` — three
  of the four FAILs. Step D still converted 14 clouds, purely because the operator had `cd`-ed
  into the nested folder and cwd rescued the import. `resolve_prep_root()` now searches for
  `*/pipeline/convert.py`, so both layouts work.
- **`import debugpy`.** The patch adds it to `train.py` for local VSCode debugging (the
  `breakpoint()` calls are already commented out, but the import is not). Applying the patch on
  a box without debugpy would have turned "no train.py" into an `ImportError` at startup —
  the exact failure `patches/README.md` warns about. `strip_debugpy()` deletes that one line
  when debugpy is not importable, and leaves it alone when it is.
- **laspy predates `header.parse_crs`** on this image — one `could not parse CRS` warning per
  plot during conversion. Harmless (our clouds carry no CRS, and `_crs_wkt` tries the WKT VLR
  first) but `smoke_test.py:60` *asserts* it, so step C would have hard-failed on a working
  box. New step **B2** probes the LAZ backend + `parse_crs`, installs `laspy[lazrs]` when
  either is missing — deliberately **without** `Dockerfile.train`'s `numpy==1.24.4` pin, which
  is only correct for the known base image — then re-checks that numpy/torch still import. If
  the repair is off or fails, a smoke-test failure whose only cause is `parse_crs` is recorded
  as WARN, not FAIL.
- New opt-outs, all self-heal on by default: `PREFLIGHT_NO_CLONE`, `PREFLIGHT_NO_PIP`,
  `PREFLIGHT_FORAINET_DIR` / `_REF` / `_URL`.
- Verified before shipping: `bash -n`; a sandbox harness exercising `resolve_prep_root`
  (nested/flat/absent), `locate_forainet` (explicit dir, `NO_CLONE` refusal) and
  `strip_debugpy` (removes, idempotent, keeps when debugpy exists); a real clone + patch +
  reverse-check + strip against a fresh `5fe600a` tree; and `PREFLIGHT_DRYRUN=1` still exits 0.

## 2026-08-28 — Pre-flight: a pinned laspy, and the dataset is fetched once

Two changes to `container_export/preflight.sh`, both aimed at the cost of *re-running* the
gate (which is the normal case — a cheap box exists to be iterated on).

- **laspy is pinned to `2.5.3`.** Step B2 installed `laspy[lazrs]` with `--upgrade`, so two
  runs of the same gate could end up on two different readers — the one thing a
  reproducibility gate must not do. `$LASPY_SPEC` (`PREFLIGHT_LASPY_SPEC`, default
  `laspy[lazrs]==2.5.3`) now drives the install, and the *version* joins the LAZ-backend and
  `header.parse_crs` probes in deciding whether to install at all: a floating 2.6 and a
  2.0.3-with-lazrs-bolted-on both converge on 2.5.3. If the pin can't be satisfied (an
  unexpected Python), it retries unpinned and WARNs rather than failing — a working laspy
  still beats none — and a version mismatch is likewise a WARN, since the gate is about
  capabilities. Dropping the `==` from the spec disables the version check entirely.
  `Dockerfile.train` still installs `laspy[lazrs]` unpinned; if the image is ever rebuilt,
  pinning it there too would stop B2 reinstalling laspy on every run of a fresh image.
- **`$DATASET_TRAIN_URL` is downloaded once, then checksum-verified.** 4.3 GB was being
  re-pulled on every run. After a good extraction, step D writes
  `.preflight_manifest.sha256` (sha256 per extracted file, paths relative to `$DATA`) and
  `.preflight_source` (the URL) *inside* `/data`; the next run verifies the manifest and
  skips the download. Checksums, not sizes or timestamps: a truncated `.laz` from an
  interrupted extraction has a plausible size and would otherwise reach the converter. A
  changed URL, a missing file, one bad checksum, or `PREFLIGHT_FORCE_FETCH=1` re-downloads.
  A `/data` with clouds but no manifest is adopted once with a WARN and then manifested
  (`PREFLIGHT_ADOPT_EXISTING=0` to refuse). `$DATASET_PATCH` is deliberately **not** cached —
  ~50 KB, and it is the archive you iterate on; a stale copy would defeat the re-run.
- Two supporting changes: `$DATA`/`$PREP` are overridable (`PREFLIGHT_DATA_DIR`,
  `PREFLIGHT_PREP_DIR`) so the gate can run outside a container's root filesystem — which is
  also what made the cache testable on the dev box — and an **empty** backend list from the
  laspy probe (interpreter missing, import crashed) is now a FAIL like `NONE`, instead of
  passing as "LAZ backend available ()".
- Verified on the dev box with a fake dataset archive over `file://`: fresh fetch writes 4
  checksums; re-run skips the download; appending one byte to a `.laz` prints
  `./plot_02.laz: FAILED` and re-downloads; `PREFLIGHT_FORCE_FETCH=1` ignores a good cache;
  deleting the manifest adopts-with-WARN (and refuses under `ADOPT_EXISTING=0`); a changed
  URL re-downloads; `PREFLIGHT_DRYRUN=1` still exits 0. Plus `bash -n`.

## 2026-08-29 — The TorchSparse block never parsed in the image (a YAML merge key)

First full pre-flight on the local container got through clone, patch, laspy, the smoke test
and the conversion, then **both** training runs died identically:

```
yaml.constructor.ConstructorError: could not determine a constructor for the tag
'tag:yaml.org,2002:merge' in ".../conf/models/panoptic/FORpartseg_3heads.yaml", line 210
```

- **Cause, ours not the environment's.** The TorchSparse block added in `cc5e0d9` was
  `PointGroup-PAPER-TS: {<<: *paper, class: …, backend: torchsparse}`. A merge key cannot be
  loaded by omegaconf 2.0.x, which `hydra-core==1.0.7` pins in the image: its loader calls
  `construct_object()` on every key node and registers nothing for
  `tag:yaml.org,2002:merge`. The failure is at *parse* time, so hydra never reads
  `model_name` and **both** backbones die with an error that names neither — which is why the
  Minkowski run failed too.
- **Why it shipped.** omegaconf **2.3** (the `aifor` env) replaced that loader with a
  `construct_mapping` override that skips non-scalar keys and lets PyYAML expand merges. The
  file is valid on the dev machine and invalid in the container; every local check used the
  dev machine. Reproduced offline by re-implementing 2.0's loader
  (`_loads_under_omegaconf_20`), which fails on the old file and passes on the new one.
- **Fix:** `PointGroup-PAPER-TS` is now a full copy of `PointGroup-PAPER` (the `&paper`
  anchor is gone), differing only in `class` and `backend`. Its three
  `${models.PointGroup-PAPER.feat_size}` interpolations were retargeted to its own block, so
  the copy is self-consistent if the two ever diverge.
- **Guards, because duplication drifts.** `misc/check_forainet_classes.py` gained two checks:
  the model config parses under 2.0 semantics, and the TS block equals the PAPER block except
  `class`/`backend` (own-block interpolations normalised before comparing). Both were
  negative-tested against a re-introduced merge key and a hand-drifted `prepare_epoch`.
  `preflight.sh` now loads the three patched configs with the image's own omegaconf
  immediately after applying the patch — a second, instead of discovering it after the 4.3 GB
  download and a 2.5-minute conversion.
- **A gate bug the same run exposed:** the post-run log greps ran unconditionally, so two runs
  that never built a model still produced `[OK] no Minkowski deprecation warning on run 2
  (TorchSparse active)` — a log with no deprecation warning because there is no log. Both grep
  blocks are now gated on `trained <log>`, the same evidence rule the `EPOCH n / m` check uses.
- `patches/forainet-local.patch` regenerated (reverses cleanly against the working tree),
  copied into `container_export/patches/`, and `container_export.tar.gz` repacked — the R2
  copy must be re-uploaded before the next run on a box.

## 2026-08-29 — Epoch 31 on the A100: `np.float` was removed in numpy 1.24

The first real A100 run died at **epoch 31 / 100** with wandb showing `failed`. Cause, from
the traceback:

```
File ".../metrics/panoptic_tracker_pointgroup_treeins_partseg.py", line 1061, in _compute_eval
    tp = np.asarray(tpsins[i_sem]).astype(np.float)
AttributeError: module 'numpy' has no attribute 'float'
```

- **Why numpy.** `np.float` / `np.int` / `np.bool` were deprecated in numpy 1.20 and
  **removed in 1.24**. The image ships **1.24.4** (`Dockerfile.train` pins it, and the base
  already had it), so these lines cannot run there at all. Nothing to do with memory, the
  GPU, or Vast.
- **Why epoch 31, and why that is the expensive part.** `models: prepare_epoch: 30` gates
  every instance-segmentation path: `forward` only clusters when `epoch > prepare_epoch`,
  and the tracker's `_compute_eval` — where the alias lives — only runs once there are
  clusters to score. So epochs 1–30 execute a strict subset of the code, and a one-word
  API removal survives ~7 hours of paid A100 time before killing the run. Same shape as the
  YAML merge key: a defect that startup checks cannot see.
- **Fix:** 8 lines, all mechanical (`astype(np.float)` → `astype(float)`,
  `astype(np.int)` → `astype(int)`), in the two files on this config's code path —
  `metrics/panoptic_tracker_pointgroup_treeins_partseg.py` (2) and
  `datasets/panoptic/treeins_set1.py` (6). Behaviour is identical; numpy's own error
  message says so. The tree carries ~90 more occurrences in datasets and trackers this
  config never loads (npm3d, s3dis, stpls3d, treeins set2/3) — deliberately left alone.
- **Guard:** `preflight.sh` step B now greps those three code-path files for removed
  aliases whenever the installed numpy is ≥ 1.24, and FAILs with the epoch number the
  crash would land on. Milliseconds, before anything is rented. Negative-tested against the
  unpatched upstream files: it flags all 8 lines, including 1061.
- Patch regenerated (reverses cleanly), `container_export/patches/` synced,
  `container_export.tar.gz` repacked (112 KB) — **must be re-uploaded to R2**.
- **Resume, do not restart:** `trainer.py:153` runs from `self._checkpoint.start_epoch` and
  `resume` is just `bool(cfg.training.checkpoint_dir)`, so pointing
  `training.checkpoint_dir` at the failed run's directory picks up at epoch 31 from
  `PointGroup-PAPER.pt` instead of repeating the 30 good epochs.

---

## 2026-09-01 — Epoch 31 again, TorchSparse this time: `KeyError: (1, 1, 1)`

With the numpy aliases fixed, the **MinkowskiEngine** run cleared epoch 31 and kept going.
The **TorchSparse** run then died at the same epoch, on an unrelated defect:

```
File ".../models/panoptic/PointGroup3heads_ts.py", line 552, in _compute_score
    score_backbone_out = self.ScorerUnet(batch_cluster)
File ".../torchsparse/nn/functional/conv.py", line 140, in conv3d
    output = SparseTensor(coords=input.cmaps[tensor_stride],
KeyError: (1, 1, 1)
```

- **How TorchSparse 1.4 tracks coordinates.** Every `conv3d` ends with
  `output.cmaps.setdefault(output.stride, output.coords)` — it records the stride it
  *produced*, never the one it consumed. A transposed conv recovers the finer resolution
  with `input.cmaps[tensor_stride]`, so a decoder can only un-stride back to resolutions
  the encoder actually wrote. `TS.SparseTensor(feats, coords)` starts with `cmaps == {}`.
- **Why only the scorer.** It is decided entirely by the first conv's stride:

  | module | `down_conv.stride` | first conv | `cmaps` after the encoder | last `ResNetUp` needs |
  |---|---|---|---|---|
  | `backbone` | `[1,2,2,2,2,2,2]` | stride 1, so `output.stride == input.stride` | `(1,1,1)` … `(64,64,64)` | `(1,1,1)` ✓ |
  | `scorer_unet` | `2` | stride 2, jumps straight past the input resolution | `(2,2,2)`, `(4,4,4)` | `(1,1,1)` ✗ |

  MinkowskiEngine never hits this — its coordinate manager regenerates parent coordinates
  on demand — which is why `model_name=PointGroup-PAPER` runs fine on the identical config.
- **Why epoch 31 *again*, for a completely different reason.** `prepare_epoch: 30` gates
  `_compute_score`, so epoch 31 is simply the first time `ScorerUnet` executes at all. Two
  independent defects (numpy alias, TorchSparse cmaps) both hid behind the same gate. Any
  bug in the scorer or the instance tracker costs 30 epochs of paid GPU time to reach.
- **Fix:** one line in `modules/SparseConv3d/nn/torchsparse.py` — the `SparseTensor()`
  factory now does `x.cmaps.setdefault(x.stride, x.coords)` before returning. It is a no-op
  for the backbone (its stride-1 first conv would `setdefault` the identical tensor) and
  propagates for free, since `output.cmaps = input.cmaps` shares one dict across the whole
  forward pass. Nothing about the computation changes.
- Patch regenerated — 8 → **9 files**, reverses cleanly and applies to a pristine `5fe600a`.
  `container_export/patches/` synced, `container_export.tar.gz` repacked (114 KB) —
  **must be re-uploaded to R2.** Both `patches/README.md` tables also gained the missing
  row for the numpy-alias file, which was never listed.
- **Verify without renting 30 epochs:** `models.PointGroup-PAPER-TS.prepare_epoch=0` forces
  the scorer to run on epoch 1, so a 2-epoch local run exercises the exact path that was
  crashing. Worth doing before any paid run — and worth considering as a permanent
  pre-flight step, the way `preflight.sh` now greps for the numpy aliases.

---

## 2026-09-01 — The same TorchSparse asymmetry, one level deeper: the kernel map

The `cmaps` seed worked — and the crash moved four lines down, from `modules.py:139`
(`conv_in`) to `modules.py:141` (`self.blocks`), and from a coordinate-map lookup to a
kernel-map one:

```
torchsparse/nn/functional/conv.py:134
    kmap = input.kmaps[(tensor_stride, kernel_size, stride, dilation)]
KeyError: ((1, 1, 1), (3, 3, 3), (1, 1, 1), (1, 1, 1))
```

- **Why there were two.** `ScorerUnet`'s encoder starts at stride 2, so nothing in it ever
  operates at the input resolution — and TorchSparse 1.4 records only what a conv
  *produces*. Its decoder needs two things at stride (1,1,1) that were therefore never
  built: the coordinate map (fixed by the seed) and the **kernel** map. The backbone's
  stride-1 stem builds both, which is why only the scorer breaks.
- **Where the stride-1 transpose comes from.** `ResNetUp` inherits `ResNetDown.__init__`
  and reuses a single `CONVOLUTION` attribute for *both* `conv_in` and its inner
  `ResBlock`s, so the blocks are built as transposed convs at stride 1
  (`modules.py:29`). TorchSparse's transposed branch looks the kernel map up with `[]`, not
  `.get()` — it assumes a forward conv built it. The forward branch builds on demand.
- **Fix:** `Conv3dTranspose` in the shim now passes `transposed=(stride != 1)`. At stride 1
  the output lives on the input's own coordinates either way; forward is
  `out[p] = Σ W[o]·in[p+o]`, transposed is `out[p] = Σ W[o]·in[p−o]`, and the offset set is
  symmetric under negation — so they differ only by a 180° flip of a **learned** kernel.
  Same shapes, same capacity, and `path_pretrained: null` means no checkpoint exists whose
  weights would care about the orientation.
- **What this costs.** It is *not* bitwise identical to MinkowskiEngine, whose coordinate
  manager builds kernel maps on demand and so keeps the transpose at every stride. It also
  changes the TS backbone's decoder blocks, which worked before. Both accepted deliberately;
  the alternative was stateful conditional logic in the shim. The TorchSparse run therefore
  **restarts clean** rather than resuming from its epoch-30 checkpoint.
- Patch regenerated (9 files, reverses cleanly, applies to a pristine `5fe600a`),
  `container_export/` synced, tarball repacked — **must be re-uploaded to R2.**
- **Two discrepancies spotted in the failed run's log, still unresolved:**
  `training=default` was used instead of `training=treeins_set1`, and it logged
  `EPOCH 1 / 100` with 188 iterations — 188 implies `batch_size: 4` (our patched value) but
  100 epochs implies the *unpatched* `epochs`, so the box's `conf/training/default.yaml`
  matches neither. And `Model size = 11872109` for `PointGroup-PAPER-TS` against the
  runbook's documented `11872126` for `PointGroup-PAPER`: a 17-parameter gap that should not
  exist between two backends of the same model. Diff the box's tree against the patch, and
  the two config blocks against each other, before the next long run.
  **→ Model size resolved 2026-09-17: not config drift.** 17 is one output of the
  `Linear(16→N)` semantic head, so `11872126` was the old 5-class model and `11872109` is
  the correct 4-class size — both backbones log it. See the 2026-09-17 entry.

---

## 2026-09-01 — The backbone A/B has its answer, and a pool that was rebuilt 196× an epoch

With both TorchSparse fixes in, the TS run cleared epoch 31 and kept going — so for the
first time there are comparable timings on **identical** code (`_compute_score` is
byte-identical between the two model files, and both hard-code `cluster_voxel_size = False`,
so the sparse backend is the only variable):

| phase | MinkowskiEngine | TorchSparse |
|---|---|---|
| epochs 1–30 (backbone + heads) | 1.04 s/it · 3:16/epoch | 1.26 s/it · 3:57/epoch |
| epoch 31+ (scorer active) | 13.70 s/it · ~44 min/epoch | 27.8–29.6 s/it · ~1h32m/epoch |

- **TorchSparse is slower here — 1.21× on the backbone, 2.1× with the scorer.** That is a
  result, not a defect, and it closes the backbone question. The "TorchSparse is faster"
  claim benchmarks against MinkowskiEngine **v0.4** on large, spatially coherent scenes;
  this image has ME from git master (0.5.x), and the hot path is the opposite shape —
  `_compute_score` hands `ScorerUnet` hundreds of small, spatially *disjoint* clusters at
  full resolution, packed into a fresh `SparseTensor` (empty `cmaps`/`kmaps`) every
  iteration. Fragmented coordinates are the worst case for hash-based kernel-map
  construction, and the per-cluster GEMMs are too small to amortise launch overhead.
  Not a build problem: `TORCH_CUDA_ARCH_LIST` includes `8.0`, so the A100 has native
  kernels for both.
- **Subtracting the pre-scorer baseline**, the scorer stage costs 12.66 s/it on Minkowski
  and 26.57 s/it on TorchSparse. Mean-shift and `region_grow` are backend-independent, so
  essentially all of that ~14 s/it delta is `ScorerUnet`.
- **The cost common to both**: `cluster_single` built and tore down a whole
  `multiprocessing.Pool` on *every* forward pass — ~196 lifecycles per epoch (188 train +
  2 val + 6 test), each forking up to `batch_size` copies of an ~8 GB process holding a live
  CUDA context, to do a few hundred ms of sklearn per child. Now one process-wide pool,
  created lazily, released via `atexit`, with an explicit `fork` context (falling back where
  fork does not exist, so importing the module off the training box stays harmless).
- **Semantics unchanged, and checked rather than asserted.** `pool.map` is order-preserving
  and `MeanShift(bandwidth, bin_seeding=True)` is deterministic. Verified standalone
  (`sklearn` only — torch is not in the `aifor` env): 48 maps / 19,200 labels
  byte-identical between fresh-pool-per-call and one shared pool.
- `cluster_loop` got the same treatment for consistency, with a comment recording that it is
  **dead code** — every `_cluster3`…`_cluster7` in both model files calls `cluster_single`.
- Patch regenerated (9 → **10 files**, reverses cleanly, applies to a pristine `5fe600a`),
  `container_export/` synced, tarball repacked — **must be re-uploaded to R2.**
- **This does not speed up the run in flight.** Python had already imported the module; the
  TorchSparse job was left alone to finish. The fix applies to the next run. Worth
  re-checking once there is a measured number: continuing costs ~103 h, and a restart
  carrying the fix might finish sooner.

---

## 2026-09-16 — Training done; local inference wired up for both backbones

Both 99-epoch runs finished. The checkpoints live in `ForAINet/pre-trained_models/`
(`PointGroup-PAPER.pt`, `PointGroup-PAPER-TS.pt`, 761 MB each) — inside the submodule, so
the existing `./ForAINet:/workspace` mount exposes them to the debug container with no
compose change. The folder stays **untracked** in the submodule so it can never enter the
patch.

- **Evaluation plots re-created.** The PLYs had been deleted. `convert.py
  convert.plots=[plot_11_val,plot_01_test,plot_14_test]` with the config's *default*
  paths (`ForAINet_export/` → `raw/SegmentedForests/`) — a pure format change, no
  restore, coordinates stay centred. The `raw/` copy of `plot_11_val.laz` is byte-identical
  to the export's (SHA-256), so this is the same conversion. Verified per plot: vertex count
  equals `n_points` in `<plot>_offsets.yml`, the ForAINet field layout, and the body is
  exactly `n_points × 37` bytes.
- **`conf/eval.yaml` made runnable** (container paths — MinkowskiEngine and TorchSparse are
  Linux-only, so a `D:\…` path cannot work): `checkpoint_dir: /workspace/pre-trained_models`,
  `fold` = plot_01_test, plot_14_test, plot_11_val (outputs `Evaluation_0/1/2`), and a run
  dir with `${model_name}` in it, since `Evaluation_*.txt` is opened in append mode. Composed
  under the container's real Hydra 1.0.7: the list parses, and the run dir resolves
  separately per model.
- **What a checkpoint contains** (read straight from the pickle, stdlib only, no torch):
  22 weight sets — `latest` plus 21 `best_<metric>` — with 99 epochs of train/val/test stats.
  Every `best_*` was selected on **val** (`base_dataset.py:519-521`), i.e. on plot_11.

  | training metrics (subsampled) | Minkowski latest | Minkowski best | TorchSparse latest | TorchSparse best |
  |---|---|---|---|---|
  | val mIoU | 65.09 | 69.08 (ep 66) | 64.99 | 68.29 (ep 22) |
  | test mIoU | 70.03 | 70.42 (ep 78) | 69.71 | 70.41 (ep 92) |
  | test F1 | 0.486 | 0.513 (ep 91) | 0.468 | 0.526 (ep 95) |

- **Two traps documented** (`docs/eval_process.md` §9). A mistyped `weight_name` silently
  loads `latest` (bare `except:` in `get_state_dict`). And TorchSparse's `best_miou` is from
  **epoch 22, before `prepare_epoch: 30`** — its ScoreNet was untrained, so `latest` stays
  the default.
- **Myth corrected: `eval.py` needs no `_val`/`_eval` file.** The suffix matters only to
  `train.py` (split at `segmentation/treeins_set1.py:381-386`, val evaluated every epoch,
  `best_*` selected on it). With `fold` set to paths, every split goes through
  `process_test()`.
- Patch 10 → **11 files** (reverses cleanly, applies to a pristine `5fe600a`, identical
  once CRLF is normalised), `container_export/` synced, tarball repacked.
- `docs/learning.md` gained *Model selection & "best" checkpoints*. Its older "In this
  project" links are still repo-root-relative from before the file moved to `docs/`, and
  no longer resolve — not fixed here.

---

## 2026-09-17 — Both evaluations verified against the 4-class scheme; results recorded

The question: does `eval.py` respect the 5 → 4 class change, and are the two evaluation runs
correct? Yes to both — checked, not assumed.

- **Where the class count lives.** Not in `train.py` — the patch only adds `import debugpy`
  there. It lives in `datasets/{segmentation,panoptic}/treeins_set1.py` and `final_eval`,
  which both entry points reach through the same `Trainer`. Eval cannot disagree with
  training unless the patch is reverted, and then nothing crashes.
- **Five checks** (now a reusable checklist, `docs/eval_process.md` §10):
  1. `misc/check_forainet_classes.py --verbose` passes 16/16.
  2. Both eval logs end the semantic head with `Linear(in_features=16, out_features=4)`,
     and **both** backbones print `Model size = 11872109`.
  3. Both load `…/<model>.pt:latest`.
  4. `Semantic Segmentation IoU` has 5 slots, `[ignore, 4 classes]`, and the mIoU
     recomputes over the 4 real ones. Minkowski plot_01:
     (0.4914+0.8160+0.6621+0.8193)/4 = 0.6972, as reported.
  5. One report block per `Evaluation_<i>.txt`.
- **The 17-parameter "gap" was the fifth class.** 11872126 − 11872109 = 17 = one output of
  `Linear(16→N)` (16 weights + 1 bias). The documented sanity value came from the July
  5-class runs. The runbook, pre-flight and backbone docs now expect `11872109`. The
  2026-09-01 suspicion of config drift was wrong, and the check script confirms the two
  model blocks match.
- **Uneven per-class IoU is class share, not swapped labels.** Heights above the lowest
  point of each 1 m cell (plot_14 / plot_01 medians) match the class names: ground
  0.05 / 0.22 m, low_vegetation 0.59 / 0.42 m, stem_points 5.4 / 2.3 m, live_branches
  11.9 / 6.8 m. On plot_14, live_branches is 68.8% of the points (IoU 0.97) and ground
  4.3% (IoU 0.65); on plot_01 ground is 22.9% and scores 0.82. New glossary entry:
  *Class imbalance*.

Full-resolution results (`weight_name: latest`, one `Evaluation_<i>.txt` per plot):

| model | plot | oAcc | mIoU | low_veg | ground | stem | live_br | inst P | inst R | PQ things |
|---|---|---|---|---|---|---|---|---|---|---|
| Minkowski | plot_01_test | 0.853 | 0.697 | 0.491 | 0.816 | 0.662 | 0.819 | 0.342 | 0.110 | 0.111 |
| Minkowski | plot_14_test | 0.956 | 0.822 | 0.843 | 0.654 | 0.825 | 0.967 | 0.544 | 0.413 | 0.349 |
| Minkowski | plot_11_val | 0.846 | 0.629 | 0.342 | 0.543 | 0.693 | 0.940 | 0.278 | 0.167 | 0.129 |
| TorchSparse | plot_01_test | 0.848 | 0.689 | 0.471 | 0.814 | 0.653 | 0.819 | 0.261 | 0.050 | 0.059 |
| TorchSparse | plot_14_test | 0.952 | 0.816 | 0.825 | 0.658 | 0.819 | 0.962 | 0.593 | 0.427 | 0.380 |
| TorchSparse | plot_11_val | 0.847 | 0.631 | 0.339 | 0.554 | 0.694 | 0.937 | 0.389 | 0.233 | 0.165 |

- **Semantics are solid on the test plots** (mIoU 0.69–0.82), and the two backbones are
  within ~0.01 on every plot. There is no measurable quality difference between them, only
  the training-speed difference recorded on 2026-09-01.
- **Tree instances are the weak part:** recall 0.05–0.43 and PQ(things) 0.06–0.38, worst
  on plot_01. The live tracker showed test F1 ≈ 0.47–0.48 over cylinders; the drop comes
  with full-resolution block merging. With two test plots, report per plot, not averaged.
- **Eval wall time is not a backbone comparison.** Minkowski took 1 h 52 min and
  TorchSparse 34 min, but between the runs `to_eval_ply` (`panoptic/treeins_set1.py:83`)
  was switched to `text=False`. The `_forEval` PLYs went from ASCII (1.32 GB for plot_01)
  to binary (333 MB), and ASCII formatting through the Windows bind mount was most of the
  first run. `to_ply` and `to_ins_ply` still write ASCII. **That edit is not in
  `patches/forainet-local.patch` yet:** `git apply --check --reverse` still passes only
  because line 83 lies outside every hunk's context, so the check cannot see it.
- **→ Resolved the same day: all three writers are binary, and the patch carries them.**
  - `to_ply` and `to_ins_ply` got `text=False` too, with one LOCAL PATCH comment covering all
    three.
  - Tested on the exact code: the three functions were extracted from the file with `ast`,
    since Docker was down and the host has no torch. All three write binary little-endian;
    xyz, labels, colours and the `-1` in `gt` round-trip identically; and every file opens
    with KPConv's `read_ply`, which raised *"The file is not binary"* on the old ASCII.
  - Nothing downstream is lost: `tree_metrics/`, `merge_tiles.py` and our restore read
    through plyfile, which takes either format.
  - **Correction (2026-09-18):** this entry originally also claimed the switch was what let
    upstream's `evaluation_stats*.py` open these files, because they use KPConv's `read_ply`.
    They **import** it and never call it — all 22 variants read through plyfile. The claim
    was wrong and has been removed from the patch READMEs, `eval_process.md` and the
    `LOCAL PATCH` comment. The switch stands on size and speed alone (1.32 GB → 333 MB per
    plot; a full 3-plot eval 1 h 52 min → ~12 min).
- **The patch-verification recipe had a blind spot, now closed.** The *old* patch was shown to
  pass `git apply --check --reverse` while a tree rebuilt from it differed by 18 lines in
  `panoptic/treeins_set1.py`. `patches/README.md` now calls the reverse check "necessary,
  not sufficient" and adds the real test: rebuild a pristine `5fe600a` from the patch and
  diff every file it touches (CRLF-normalised). The regenerated patch (11 files) passes
  both, and the documented block was run verbatim.

---

## 2026-09-17 — Viewer: one colour per value (palettes, hex, picker), and a colour bug fixed

**`misc/view_split_point_cloud.ipynb`** can now set the colour of every value of the
current *Colour by* field. Tracing the request first turned up a bug that had to be
fixed for the feature to mean anything.

- **The bug.** `color_spec()` handed out palette colours by a value's **rank among the
  values currently drawn**, so any change to the value set recoloured everything.
  Measured on the notebook's own code, class 3 came out **light orange** over
  `{0,1,2,3,4}`, **blue** after filtering to `{3,4}`, and **light blue** in a cloud
  without classes 0 and 2. In a two-cloud comparison tool the same class could
  therefore look different on each side.
- **The fix, and the feature, are one mechanism.** A new `ColourScheme` (cell 3, tk-free)
  holds `{field: {value: "#rrggbb"}}`. A value's colour is **pinned the first time it is
  drawn** and then only the user moves it. `PointCloudCompareApp` creates **one** scheme
  and gives it to both panels, so a value matches across Cloud A / Cloud B; every change
  redraws both.
- **`Colours…`** (new button, on the *Fields shown… / 3D view* row) lists each value in
  view with a swatch, a hex box, **Pick…** (the OS colour chooser) and its point count,
  plus a palette dropdown and *Apply palette*. `parse_colour` accepts `#E9E56B`, `#abc`,
  `e9e56b` and matplotlib names (`red`, `tab:blue`) but **rejects bare numbers** —
  matplotlib reads `"1"` as white and `"0.5"` as grey, never what a hex box means.
- **Palettes**: tab20 (default, so first-draw colours are unchanged), tab10, Set1-3,
  Paired, Dark2, Accent, Pastel1, plus **ForAINet classes** — the `OBJECT_COLOR` rows
  from `panoptic/treeins_set1.py`, re-indexed to our on-disk `semantic_seg`
  (0 unclassified black, 1 low_veg yellow, 2 ground blue, 3 stem brown, 4 branches
  salmon).
- **Save… / Load…** write one JSON per field (default `misc/conf/colours/<field>.json`).
  Loading validates every entry and **reports** the bad ones instead of dropping them.
- **The discrete rule is deliberately unchanged**: ≤ 20 distinct integers *in what is
  drawn*. That is what keeps the documented `tree_ID` workflow working (filter to a few
  ids → each tree gets its own colour → they appear as rows in `Colours…`). On a float
  or wide field the button explains how to get there. `color_spec()` still serves both
  the 2-D scatter and the 3-D export, so custom colours reach the 3-D window for free
  and `misc/view_cloud_3d.py` needed **no change**. `color_spec(cvals)` without a scheme
  still behaves exactly as before.
- Also: `load_file()` split into the dialog plus `load_path(path)`, so a test can fill a
  panel without clicking.
- **Verified.** 31 headless checks on the real notebook cells: the three-colour bug
  scenario now gives one colour; `parse_colour` accept/reject table; preset hexes exact;
  `apply_palette` order-independent; JSON round-trip with int keys and malformed-file
  reporting; legacy `color_spec` untouched. 12 GUI checks with a real tk window —
  `plot_11_val.ply` (18.9 M pts, loaded in 1.5 s) in panel A and a synthetic cloud
  *missing classes 0 and 2* in panel B: a custom magenta appears in **both** panels and
  B's colours are a subset of A's; the ForAINet preset reaches the canvas; the editor
  opens, and explains itself on a float field. 3-D: the payload carries the preset blue
  and the custom magenta per point and in the legend, and
  `view_cloud_3d.py --smoke` rendered it — the PNG shows ground blue at the bottom,
  crowns salmon on top and the overridden stems magenta as vertical trunks.

---

## 2026-09-18 — The pooled report made 4-class-aware; the backbone comparison settled

`evaluation_stats_FOR.py` pools every plot of an eval run into one set of numbers — the
figure to quote as *the* result. It was **not** aware of the 5 → 4 class change, and could
not run at all. Five separate problems, all now fixed and carried in the patch (12 files):

| Problem | Was | Now |
|---|---|---|
| class constants | `NUM_CLASSES_sem 6`, `sem_classcount [1..5]`, `..._remove_ground [1,3,4,5]`, `thing_classes [3,4,5]` | `5`, `[1,2,3,4]`, `[1,3,4]`, `[3,4]` — same values `final_eval` already carries |
| `np.float` ×2 | `AttributeError` on numpy ≥ 1.24 | `float` |
| the hardcoded path | plain string, so `\04`→`\x04` and `\a`→BEL: glob matched 0 files and the script died later on an undefined variable | `r""` plus `<run_dir> [index …]` on the command line |
| binary counters | re-zeroed **inside** the per-plot loop, so every "Binary Semantic Segmentation" figure and the stuff RQ/SQ/PQ described only the *last* plot | initialised once; pooled binary mIoU now lands between the per-plot values (0.9573 / 0.9633 → 0.9603) instead of equalling the last |
| `positive_classes[[…]]` | doubled brackets → `(1,K)` array → `float()` raised `TypeError`; with >1 class this line could never run | single brackets |

Also: the unused `torch_points3d` import is gone, so **the pooled report runs on the host**
in `aifor` — no container, no GPU. `misc/check_forainet_classes.py` now guards its four
class constants too (21 checks; negative-tested by reverting `thing_classes`, which it
catches by name).

**Pooled result over the two `_test` plots** (41,534,130 points; `weight_name: latest`):

| | MinkowskiEngine | TorchSparse |
|---|---|---|
| oAcc | **0.9048** | 0.8996 |
| mIoU | **0.7768** | 0.7662 |
| mIoU without ground | **0.7734** | 0.7602 |
| binary mIoU (tree / non-tree) | **0.9603** | 0.9484 |
| mMUCov / mMWCov | **0.357 / 0.419** | 0.298 / 0.369 |
| mPrecision / mRecall | **0.505 / 0.238** | 0.494 / 0.196 |
| instance F1 | **0.324** | 0.280 |
| meanSQ / meanPQ (things) | 0.732 / **0.237** | **0.755** / 0.212 |

MinkowskiEngine is ahead on every figure except `SQ (things)` — when TorchSparse did match
a tree it outlined it slightly better, but it found fewer.

**Verified, not assumed.** An independent numpy recomputation of the pooled confusion matrix
straight from the same PLYs reproduces the script's `oAcc`, `mIoU` and `mIoU without ground`
to 13 decimals, over exactly 20,831,953 + 20,702,177 points. That is the check that would
catch a class-index mistake.

**Eval is deterministic — the earlier "run-to-run variance" was wrong.** Three runs per
backbone produced **byte-identical** `Evaluation_<i>.txt` (zero spread on mIoU, oAcc,
MUCov, precision, recall, F1, PQ). So the backbone differences above are real and not
noise, and my earlier claim that sparse-conv atomics were moving the instance numbers was
unfounded.

**What did move them: whether the eval cache already existed.** The 2026-09-16 run that
disagreed was the one that *built* `processed_0.2_test/`; every run since has loaded it.
Tested by moving the cache aside and re-running — the result reproduced that first run to
every printed digit:

| | cache **built** by the run | cache **loaded** |
|---|---|---|
| plot_01 mIoU / recall | 0.6972 / 0.1102 | 0.6996 / 0.0932 |
| plot_14 mIoU / recall | 0.8223 / 0.4133 | 0.8216 / 0.4667 |
| plot_11 mIoU / recall | 0.6292 / 0.1667 | 0.6320 / 0.2000 |

Two regimes, each perfectly reproducible, differing by up to 0.053 in instance recall. The
mechanism inside `process_test` is **not** isolated — the operational rule is what matters:
compare only runs in the same regime, and after deleting the cache (required whenever the
raw PLYs change) re-run every model being compared. Every number in this entry is from
cache-loading runs for both backbones, so the comparison is sound. Recorded in
`eval_process.md` §11.

Housekeeping: the five extra runs made for this test keep their `Evaluation_*.txt` and
`eval.log`; their PLYs and checkpoint copies were deleted (eval/ 30.0 → 9.9 GB). The two
runs the tables quote are untouched.

**Correction carried out.** The claim that binary PLYs were needed for upstream's
`evaluation_stats*.py` (KPConv `read_ply` rejecting ASCII) was wrong — those scripts import
`read_ply` and never call it; all 22 read through plyfile. Removed from the patch READMEs,
`eval_process.md` §7, the `LOCAL PATCH` comment and the 2026-09-17 entry. Binary still wins
on size and speed (1.32 GB → 333 MB per plot; a 3-plot eval 1 h 52 min → ~12 min).

---

## 2026-09-21 — The report chapters, the comparison with the publication, and a cleanup

The pipeline work is done; this entry closes the project out.

- **Two report chapters** (`docs/report_methodology.md`, `docs/report_results.md`) written
  for a reader rather than a maintainer — the opposite register to the rest of `docs/`.
  Methodology covers the dataset and its four unification steps, the technology stack and
  the container encapsulation, and the twelve-file patch grouped by purpose; Results covers
  training, the backbone comparison, the comparison with the publication, and six themed
  challenges. No new measurements: every figure is sourced from `evaluation_total.txt`,
  this file, or the offsets sidecars, and a check asserts that each one rounds from a value
  present in those sources.
- **Compared against the paper's Table 4, basic setting** — the row our training arguments
  correspond to. The metric *names* differ but the quantities are identical, and that was
  verified rather than assumed: the paper's completeness 79.3 % with commission error
  21.2 % implies precision 78.8 %, and those recombine to its own reported F-score of 79.0.
  So *completeness* = recall, *commission error* = 1 − precision, *F-score* = F1.

  | | paper (basic) | Minkowski | TorchSparse |
  |---|---|---|---|
  | semantic mIoU | 73.0 | **77.7** | 76.6 |
  | semantic mAcc | 81.2 | **85.9** | 85.4 |
  | instance F-score | **79.0** | 32.4 | 28.0 |
  | coverage | **77.0** | 35.7 / 41.9 | 29.8 / 36.9 |

  **Semantic segmentation transferred to terrestrial LiDAR; instance segmentation did not.**
  Caveat on the semantic half: ours averages over 4 classes, theirs over 5.
- **A caveat that had been missed entirely, and materially changes the reading.**
  SegmentedForests carries semantic labels only — the source `.laz` dimensions are the
  standard LAS set plus `Class` and `Split`, with **no tree id**. Every instance label in
  this project comes from 3DFin (Stage 1), not from manual annotation, while the paper's
  instance figures are measured against hand-delineated trees. So label provenance is a
  second candidate explanation for the instance gap alongside the airborne-vs-terrestrial
  sensor difference, and the two cannot be separated without a manually delineated subset.
  Recorded in both chapters.
- **`misc/make_backbone_comparison.py`** regenerates `docs/wandb_backbone_comparison.{md,html}`
  from the two `evaluation_total.txt` files, so those tables are reproducible rather than
  hand-maintained. Verified: a regeneration is byte-identical to the committed files.
  **`misc/log_html_to_wandb.py`** uploads an HTML file as a W&B media panel — the only way
  to get styled tables into a report, since W&B's markdown blocks strip raw HTML.
- **Cleanup for the final commit.** Deleted the repo-root `Dockerfile` and `.dockerignore`
  (a VSCode `python:3-slim` template with no CUDA that `gpu_training_runbook.md` already
  warned against — the warning is rewritten, not just orphaned), `docs/errors/` (raw
  epoch-31 crash logs, all three narrated above), and `misc/view_pointcloud.ipynb`
  (superseded, referenced nowhere). **`.gitattributes` is finally tracked**: it is the fix
  for `core.autocrlf=true` shipping CRLF `preflight.sh` into the container, where bash dies
  on the first line. `docs/training_forainet.pdf` is ignored by name rather than by `*.pdf`,
  which would have swallowed the report itself.

---

## Planned

- **In-PLY reclassification (optional)**: the `_semantic_labels()` hook in
  `pipeline/forainet_prep.py` is still a straight copy; if remapped labels should be
  written directly into the Stage 2 PLYs (instead of / in addition to the standalone
  `class_unifier` stage), a `class_map` key can plug in there the same way.
- Optional: `pyproj` in the env would let `crs_wkt` capture also GeoTIFF-key CRSs
  (LAS 1.2/1.3 style); WKT-based CRSs (mandatory in LAS 1.4) already work without it.
