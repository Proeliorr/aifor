# CLAUDE.md

Project context for Claude Code sessions. Read `README.md` for full documentation and
`progress.md` for the development history — keep the latter updated when adding
functionality.

## What this is

Forest LiDAR point-cloud pipeline preparing plot clouds for **ForAINet semantic
segmentation** (resolved at whole-plot level — per-tree data is intentionally out of
scope; tools for that live in `misc/`).

- **Stage 1** `main_pipeline.py` → `pipeline/threedfin.py`: batch-runs the 3DFin v0.6.0
  CLI per plot (instance segmentation, `<plot>_tree_ID_dist_axes.las`).
- **Stage 2** `forainet_prep.py` → `pipeline/forainet_prep.py`: centers coordinates
  (min-subtraction → all ≥ 0), writes ForAINet PLYs + `<plot>_offsets.yml`
  (shifts + source metadata: format, LAS version/point format/scales, CRS WKT);
  `mode=restore` brings classified PLYs back to original coordinates as LAZ 1.4
  (scales + CRS re-applied, all fields kept as extra-bytes dims).

Config pattern: one shared `conf/config.yaml`, one section per stage
(`threedfin:`, `forainet_prep:`), one hydra-zen entry script per stage.
Override keys with the section prefix: `forainet_prep.plots=[plot_02]`.

## Environment / running

- Everything runs in the **`aifor`** mamba env. In PowerShell, `mamba` is a batch file
  and cannot sit in a pipeline — invoke via `cmd /c`:
  `cmd /c "mamba run -n aifor python forainet_prep.py forainet_prep.dry_run=true"`
- Run entry scripts from `assignment/` (the repo root). Each run writes a Hydra log
  under `outputs/<date>/<time>/` (gitignored).
- No pyproj in the env: LAS CRS handling goes through WKT VLRs directly
  (`_crs_wkt` in `pipeline/forainet_prep.py`); GeoTIFF-key CRSs degrade to a warning.

## Data (gitignored, machine-local)

`SegmentedForests/`: `pointclouds/` (plot_01–14.laz, local coordinates, **no CRS**),
`3DFin_settings/` (per-plot .ini, tracked), `3DFin_output/` (Stage 1 results),
`ForAINet_input/` (Stage 2 PLYs; only the small `*_offsets.yml` are tracked).
Paths in `conf/config.yaml` are absolute and machine-specific — adjust or override
on other machines.

## Conventions

- Heavy docstrings/comments (code doubles as teaching material for non-Python-experts).
- Engines follow the same batch skeleton: module logger, per-plot loop with
  `succeeded/failed/skipped`, `plots` filter, `overwrite`, `dry_run`,
  `continue_on_error`, `=`-rule summary log line.
- laspy gotcha: materialise accessors with `np.asarray(...)` (ScaledArrayView /
  SubFieldView are not plain arrays).
- Planned work is listed at the bottom of `progress.md`. The next known item:
  `class_map` label reclassification — plug into `_semantic_labels()` in
  `pipeline/forainet_prep.py` (isolated there on purpose).

## Adjacent (not part of the pipeline)

- `misc/`: standalone tools (per-tree splitter with own config; original
  `Points2ForAINet.py` converter).
- `ForAINet/`: clone of https://github.com/prs-eth/ForAINet.git (gitignored, not a
  submodule yet) — the deep-learning segmentation framework itself.
