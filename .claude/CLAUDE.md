# CLAUDE.md

Project context for Claude Code sessions. Read `README.md` for full documentation and
`progress.md` for the development history — keep the latter updated when adding
functionality.

## What this is

Forest LiDAR point-cloud pipeline preparing plot clouds for **ForAINet semantic
segmentation** (resolved at whole-plot level — per-tree data is intentionally out of
scope; tools for that live in `misc/`).

- **The chain** `main_pipeline.py` → `pipeline/chain.py`: runs the stages **per plot**
  (3DFin → Stage 2 → delete that plot's `.las`) so only one intermediate is alive at a
  time — 15.2 GB instead of 46.3 GB. Adds no processing: it calls the same two engines
  with the same config sections. `chain.stages=[threedfin]` is Stage 1 alone (what
  `main_pipeline.py` used to mean).
- **Stage 1** `pipeline/threedfin.py`: batch-runs the 3DFin v0.6.0 CLI per plot
  (instance segmentation, `<plot>_tree_ID_dist_axes.las`).
- **Stage 2** `forainet_prep.py` → `pipeline/forainet_prep.py`: centers coordinates
  (min-subtraction → all ≥ 0), writes ForAINet PLYs + `<plot>_offsets.yml`
  (shifts + source metadata: format, LAS version/point format/scales, CRS WKT);
  `mode=restore` brings classified PLYs back to original coordinates as LAZ 1.4
  (scales + CRS re-applied, all fields kept as extra-bytes dims).
- **Class unifier** `class_unifier.py` → `pipeline/class_unifier.py`: remaps the
  semantic-label column through `class_map: {source: unified}` (many-to-one entries
  merge classes) into a new column (`unified_field`), **drops** the points mapped to
  `drop_value` (-1), exports the 7 columns `x,y,z,intensity,<source label>,
  <unified label>,<tree id>` as `output_format: ply | npy | both` plus a `<plot>.json`
  sidecar (provenance + drop counts) — readable by `misc/view_split_point_cloud.ipynb`.
  Reuses Stage 2's `_read_cloud`/`_collect_inputs`/`_write_ply`.
  **Use `ply` (the default) for viewing**: the viewer only draws class colours + legend,
  unique counts and per-value histogram bars for INTEGER columns, and a bare `.npy` is
  one float64 matrix, so every label there renders as a continuous ramp. The PLY keeps
  each column's dtype (`semantic_seg` u1, `tree_ID` i4) and is ~32 vs 56 bytes/point.
  Coordinates stay **original** — centering belongs to Stage 2, which also writes the
  offsets file that makes it reversible.

## The semantic class scheme (SegmentedForests → ForAINet)

One shared `classes:` block in `conf/config.yaml` is interpolated by **both**
`class_unifier:` and `forainet_prep:`, so the `.npy` exports and the PLYs ForAINet
trains on cannot drift. The rule itself lives in `pipeline/classes.py` — a neutral
module because `class_unifier` imports the reader from `forainet_prep`, so
`forainet_prep` cannot import back without a cycle.

- Map (verified against all 14 plots, whose `Class` union is exactly these 16 values):
  `8,9,11 → -1` (dropped) · `0,4,5,6,7,12,13,22,23 → 0` · `1 → 1` · `3,10 → 2` · `2 → 3`.
- **The map must be total** — an unmapped source value raises and fails that plot. No
  `unmapped_value` fallback: guessing would smuggle a bogus class into training.
- ForAINet's class 4 (`branches`) has no SegmentedForests counterpart and was removed
  from the framework; the submodule edits live in `patches/forainet-local.patch`.
- Stage 2 drops points **before** computing the centering offsets, so
  `<plot>_offsets.yml` describes the points actually in the PLY (`restore` validates
  against them). A restored cloud is therefore smaller than the 3DFin input.
- **Split lives in the FILE NAME.** ForAINet has no manifest: it tests
  `name[-7:-4]=="val"` then `name[-8:-4]=="test"`, everything else being training data.
  Stage 2's `split:` block (`n_val`/`n_test`, each a count or a fraction) ranks plots by
  point count and holds out the **smallest** — currently plot_11 → val, plot_14 + plot_01
  → test. Computed over *all* matched clouds before the `plots` filter, so a subset run
  gives the same suffix a full run would. Changing the split deletes the plot's old file:
  ForAINet globs `raw/**/*.ply`, so two suffixes for one plot would train on test data.
  `<plot>_offsets.yml` stays keyed by the plain plot name.
- **`treeID` convention (easy to get wrong):** ForAINet needs `0` = "not part of any
  tree", and skips any instance whose id also appears on a non-thing point. 3DFin
  instead gives every point the nearest stem's id — ground included — which made
  **0 of 42** instances survive on plot_10. `classes.instance_classes` ([3, 4] =
  stem_points, live_branches, in `class_names` numbering) drives
  `zero_stuff_tree_ids()`, clearing the id everywhere else. Do not "simplify" this away.
- The two `-1`s are different: ours (`drop_value`) deletes rows before writing;
  ForAINet's arrives after its `semantic_seg - 1` shift and means `ignore_index` in the
  loss — those points stay in the cloud and keep supporting their neighbours' features.
- Hydra gotcha (measured): the map's keys are **integers**, so
  `classes.class_map.5=1` and `classes.class_map={99: 0}` both fail with "Key ... is
  not in struct"; `++classes.class_map={99: 0}` merges; and
  `class_unifier.class_map={...}` *replaces* that stage's map, severing the link to the
  shared block so Stage 2 silently keeps the old one. Change the scheme by editing
  `conf/config.yaml`.

Config pattern: one shared `conf/config.yaml`, one section per stage
(`chain:`, `threedfin:`, `class_unifier:`, `forainet_prep:`), one hydra-zen entry script
per stage. Override keys with the section prefix: `forainet_prep.plots=[plot_02]`.
`main_pipeline.py` is the exception — the chain needs three sections at once, and `zen()`
maps a *single* section onto a function, so it converts the config itself
(`OmegaConf.to_container(cfg, resolve=True)`).

## Environment / running

- Everything runs in the **`aifor`** mamba env. In PowerShell, `mamba` is a batch file
  and cannot sit in a pipeline — invoke via `cmd /c`:
  `cmd /c "mamba run -n aifor python forainet_prep.py forainet_prep.dry_run=true"`
- Run entry scripts from `assignment/` (the repo root). Each run writes a Hydra log
  under `outputs/<date>/<time>/` (gitignored).
- No pyproj in the env: LAS CRS handling goes through WKT VLRs directly
  (`_crs_wkt` in `pipeline/forainet_prep.py`); GeoTIFF-key CRSs degrade to a warning.

## Data (gitignored, machine-local)

`SegmentedForests/`: `pointclouds/` (plot_01–14.laz, local coordinates, **no CRS**,
4.4 GB — the only permanent input), `3DFin_settings/` (per-plot .ini, tracked),
`3DFin_output/` (Stage 1 results — normally just `3dfin_log.txt` per plot, since the
chain consumes the clouds), `ForAINet_input/` (**now only the tracked
`*_offsets.yml`**; Stage 2's PLYs go straight to `paths.forainet_raw` inside the
ForAINet submodule, ~29 GB). Paths in `conf/config.yaml` are absolute and
machine-specific — adjust or override on other machines.
Delete `…/treeinsfused/processed_0.2/` whenever `raw/` changes: ForAINet caches its
preprocessed tensors there and will otherwise train on the stale ones.

## Conventions

- Heavy docstrings/comments (code doubles as teaching material for non-Python-experts).
- Engines follow the same batch skeleton: module logger, per-plot loop with
  `succeeded/failed/skipped`, `plots` filter, `overwrite`, `dry_run`,
  `continue_on_error`, `=`-rule summary log line.
- laspy gotcha: materialise accessors with `np.asarray(...)` (ScaledArrayView /
  SubFieldView are not plain arrays).
- **Intermediates are transient.** The chain deletes each plot's 3DFin `.las` once
  Stage 2 has succeeded for that plot — only point clouds, never `3dfin_log.txt`, never
  when Stage 2 is absent from `stages`, never after a failure (the retry must not cost
  another 3DFin run). `keep_intermediates` is `false` / `true` / a **list of plot
  names**; the list form is what makes the opt-in `class_unifier` diagnostics workable,
  since it reads the `tree_ID` only 3DFin produces. Measured: 108.6 → 34.3 GB.
- **The chain decides the split from the SOURCE headers** and passes it to Stage 2 as
  `split_of`, because a per-plot call cannot rank a plot against the others. Valid only
  because 3DFin preserves point counts exactly (verified, all 14 plots, zero delta) — if
  that ever stops being true, the split silently becomes arbitrary.
- Both stage engines **return** `{succeeded, failed, skipped}` (the chain acts on it) and
  take `log_summary=False` to drop the run-level framing when called once per plot.
- **Parallelism** (`pipeline/parallel.py`): `class_unifier`, `forainet_prep` and
  `misc/tree_splitter` run several plots at once in worker *processes*; 3DFin
  (Stage 1) stays sequential, and the chain passes `workers=1` since it hands Stage 2
  one plot at a time. Concurrency is bounded by **RAM, not cores** — Stage 2
  costs a measured ~134 bytes/point, so plot_08 (280 M pts) needs ~37 GB alone and a
  worker per core would exhaust the machine. Sizes come from LAS headers; work is
  admitted while it fits `memory_budget_frac` of RAM. `workers: 1` = inline, the old
  behaviour exactly. Worker log lines are replayed by the parent so Hydra's log keeps
  the per-plot detail. Full 14-plot Stage 2: 3.3 → 2.4 min (plot_08 alone is 33% of
  all points, so it sets the floor).
- A full 14-plot Stage 2 run writes **~25 GB**. Benchmarks and scratch output belong on
  `D:`, not the default temp on `C:` — filling the system drive surfaces as confusing
  failures deep inside `plyfile`'s write.
- Planned work is listed at the bottom of `progress.md`. Label reclassification is
  **done**: it ships both as the standalone `class_unifier` stage and, via
  `_semantic_labels()` in `pipeline/forainet_prep.py`, in the Stage 2 PLYs ForAINet
  actually trains on — both driven by the one `classes:` config block (see above).

## Adjacent (not part of the pipeline)

- `misc/`: standalone tools (per-tree splitter with own config; original
  `Points2ForAINet.py` converter).
- `ForAINet/`: **git submodule** of https://github.com/prs-eth/ForAINet.git, pinned at
  `5fe600a` — the deep-learning segmentation framework itself. Get it with
  `git submodule update --init`. Our local edits (debug training config, wandb entity,
  `debugpy` in `train.py`) are **not** stored by this repo — a submodule only records a
  commit SHA — so they live in `patches/forainet-local.patch`; reapply with
  `cd ForAINet && git apply ../patches/forainet-local.patch`. See `patches/README.md`.
  Upstream commits `__pycache__/*.pyc`, so training runs leave the submodule dirty with
  ~115 `.pyc` entries — harmless noise.
