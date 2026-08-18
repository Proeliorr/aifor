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

---

## Planned

- **In-PLY reclassification (optional)**: the `_semantic_labels()` hook in
  `pipeline/forainet_prep.py` is still a straight copy; if remapped labels should be
  written directly into the Stage 2 PLYs (instead of / in addition to the standalone
  `class_unifier` stage), a `class_map` key can plug in there the same way.
- Optional: `pyproj` in the env would let `crs_wkt` capture also GeoTIFF-key CRSs
  (LAS 1.2/1.3 style); WKT-based CRSs (mandatory in LAS 1.4) already work without it.
