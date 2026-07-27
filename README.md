# Forest Point-Cloud Semantic Segmentation Pipeline

A small, self-contained batch pipeline that takes a folder of forest LiDAR plots to
ForAINet-ready semantic-segmentation inputs — and brings the classified results back.
It is configured and launched with
[hydra-zen](https://mit-ll-responsible-ai.github.io/hydra-zen/); each stage has its own
entry script reading its own section of the shared `conf/config.yaml`:

- **Stage 1 — 3DFin instance segmentation** (`main_pipeline.py`): drives the
  [3DFin](https://github.com/3DFin/3DFin) v0.6.0 CLI over every plot (`*.laz` +
  matching `<stem>.ini`), producing one instance-segmented cloud per plot.
- **Stage 2 — ForAINet preparation** (`forainet_prep.py`): centers each cloud's
  coordinates, writes ForAINet-ready PLYs + per-plot offset/metadata files, and
  **restores** classified results to original coordinates (LAZ 1.4, CRS re-applied).

The development history — what was added when and why — is tracked in
[`progress.md`](progress.md).

> The sibling `ForAINet/` folder is a **separate, deep-learning** approach to the same forest
> problem (panoptic point-cloud segmentation). It is unrelated to this batch runner and is
> documented on its own in [`ForAINet/ARCHITECTURE_ANALYSIS.md`](ForAINet/ARCHITECTURE_ANALYSIS.md).

> **Project scope.** The main approach here is **semantic segmentation resolved at
> whole-plot level**, so the pipeline stops at the plot-level 3DFin output. Standalone
> tools that fall outside that scope (e.g. splitting plots into per-tree files) live in
> [`misc/`](misc/README.md).

---

## Files and what they do

| File | Role |
|---|---|
| [`main_pipeline.py`](main_pipeline.py) | **Stage 1 entry point.** `@hydra.main` loads the shared `conf/config.yaml` and runs the 3DFin stage from its `threedfin:` section via `zen(run_pipeline)(cfg.threedfin)` (hydra-zen maps the section's keys onto the function's parameters). |
| [`pipeline/threedfin.py`](pipeline/threedfin.py) | **Stage 1 engine.** `run_pipeline(...)` collects `*.laz` files, pairs each with its `.ini`, builds and executes the `3DFin cli ...` command per plot, and prints a summary. Helpers: `_read_misc_flags` (read `[misc]` flags from an `.ini`), `_write_patched_ini` (write the patched copy — see quirks below), `_build_command` (assemble the argument vector). |
| [`forainet_prep.py`](forainet_prep.py) | **Stage 2 entry point.** Same pattern as `main_pipeline.py`, running the ForAINet preparation from `config.yaml`'s `forainet_prep:` section via `zen(prep_forainet)(cfg.forainet_prep)`. See "Stage 2" below. |
| [`pipeline/forainet_prep.py`](pipeline/forainet_prep.py) | **Stage 2 engine.** `prep_forainet(...)` centers each plot cloud (subtracts min x/y/z), writes ForAINet-ready PLYs + per-plot `<plot>_offsets.yml`, and can **restore** classified results to original coordinates. |
| [`pipeline/__init__.py`](pipeline/__init__.py) | Package init; re-exports `run_pipeline` and `prep_forainet`. |
| [`conf/config.yaml`](conf/config.yaml) | **Pipeline config.** One section per stage — `threedfin:` (Stage 1) and `forainet_prep:` (Stage 2) — each read by its own entry script. Override any key with its section prefix, e.g. `threedfin.plots=[plot_01]`. |
| [`misc/`](misc/README.md) | **Standalone tools outside the pipeline.** `misc/tree_splitter.py` (per-tree `.npy` splitting, own `misc/conf/config.yaml`) and `misc/Points2ForAINet.py` (the original generic converter Stage 2 was ported from). |
| [`requirements.txt`](requirements.txt) | Pinned Python deps (`hydra-zen`, `hydra-core`, `omegaconf`, `laspy`, `lazrs`, `plyfile`, `PyYAML`), all already satisfied by the `aifor` env. **3DFin itself is an external CLI, not a pip dependency of this package.** |
| [`progress.md`](progress.md) | **Development history** — what functionality was added when, and what is planned. |
| [`.gitignore`](.gitignore) | Excludes the large binary data (input clouds, Stage 1 outputs, Stage 2 clouds) from version control; the small `<plot>_offsets.yml` metadata files stay tracked. |

### Two 3DFin 0.6.0 quirks the runner works around

1. **Patched `.ini` copy.** 3DFin pydantic-validates the *entire* `.ini` on load, including
   `[misc] output_dir`, which must be an **existing directory**. The dataset's `.ini` files
   hardcode a Windows path (`F:\classification`) that may not exist, causing a
   `ValidationError`. So for each plot the runner writes a **patched copy** (`3dfin_params.ini`)
   whose `output_dir` points at the real per-plot output folder. **The original `.ini` files
   are never modified.**
2. **Flags derived from the `.ini`.** The CLI rebuilds `[misc]` from the command line, where
   `is_normalized = not --normalize`, `is_noisy = --denoise`, `export_txt = --export_txt`. To
   reproduce each plot's recorded settings, the runner **derives the flags from that plot's
   `.ini` `[misc]` section** by default (e.g. `is_normalized=False` → pass `--normalize`). The
   `normalize` / `denoise` / `export_txt` config keys override this when set to `true`/`false`
   (`null` = derive from the `.ini`).

---

## Dataset

The plot clouds are **not part of this repository** — they come from the public
**SegmentedForests** dataset: 14 terrestrial-LiDAR forest plots, over 920 million
labelled points, which is what `plot_01 … plot_14` refer to throughout.

| | |
|---|---|
| **Download** | [Zenodo record 17396681](https://zenodo.org/records/17396681) — one file, `SegmentedForests.zip` (4.4 GiB, MD5 `757d1d92924702e48637dd18647ab517`) |
| **Dataset DOI** | [10.5281/zenodo.17396681](https://doi.org/10.5281/zenodo.17396681) — this is **v1.0**, and the one to pin. The concept DOI [10.5281/zenodo.17396680](https://doi.org/10.5281/zenodo.17396680) always resolves to the newest version, so it is not reproducible. |
| **Paper** | [10.1093/forestry/cpaf062](https://doi.org/10.1093/forestry/cpaf062) |
| **License** | MIT — redistribution permitted with attribution |

### Citation

Cite the paper (and, for the exact data, the versioned Zenodo DOI above):

> Laino, D., Cabo, C., Ordóñez, C., Bolanos, R., Janvier, R., Giulioni, F., Herrmann, M.,
> Hudak, A., Parsons, R., & Santin, C. (2026). SegmentedForests: a labelled dataset of
> terrestrial LiDAR point clouds for semantic segmentation of forests. *Forestry: An
> International Journal of Forest Research*, **99**(2), cpaf062.
> <https://doi.org/10.1093/forestry/cpaf062>

## Dataset layout

```
SegmentedForests/
├── pointclouds/          # input clouds:  plot_01.laz … plot_14.laz   <-- unzip the download here
├── 3DFin_settings/       # per-plot 3DFin params: plot_01.ini … plot_14.ini (+ README.txt)
├── 3DFin_output/         # Stage 1 results, one subfolder per plot (created by the runner)
│   └── plot_01/ …
└── ForAINet_input/       # Stage 2 results: plot_01.ply + plot_01_offsets.yml … (created by forainet_prep)
```

**Where to put the download:** extract `SegmentedForests.zip` so the 14 plot clouds end up
directly in `SegmentedForests/pointclouds/` as `plot_01.laz … plot_14.laz` (flatten any
extra nesting the archive introduces). Verifying the MD5 above before unzipping is worth
the minute — a truncated 4.4 GiB download otherwise surfaces much later as a corrupt-LAZ
error inside Stage 1.

These folders ship in git as **empty placeholders** (the paths are hardcoded in
`conf/config.yaml`), so a fresh clone already has the structure — only the data is missing.
The `3DFin_settings/*.ini` files *are* tracked, so you do not need to obtain those
separately.

Each `<plot>.laz` must have a matching `<plot>.ini`. Plots with no `.ini` are **skipped**
with a warning.

### Semantic classes: SegmentedForests → ForAINet

The dataset's `Class` field and ForAINet's `semantic_seg` use different vocabularies, so
the labels are remapped. The scheme is defined **once** in the `classes:` block of
[`conf/config.yaml`](conf/config.yaml) and used by both the class unifier and Stage 2, so
the inspection exports and the clouds ForAINet trains on cannot disagree.

| SegmentedForests `Class` | → ForAINet | Meaning |
|---|---|---|
| 0, 4, 5, 6, 7, 12, 13, 22, 23 | **0** | low_vegetation |
| 1 | **1** | ground |
| 3, 10 | **2** | stem_points |
| 2 | **3** | live_branches |
| 8, 9, 11 | **−1** | no counterpart — **these points are removed** |

Verified against the data: the union of `Class` over all 14 plots is exactly those 16
values (individual plots differ — `plot_01` has only `0–4`, `plot_10` has
`0–5, 11, 12, 13, 22, 23`).

Two consequences worth knowing:

- **The map must be total.** A class present in the data but missing from `class_map`
  **fails that plot** rather than being silently kept or forced to a default — an
  unmapped label would otherwise reach training as a bogus class.
- **Output is smaller than input.** Dropped points are removed before the coordinates are
  centered, so `<plot>_offsets.yml` describes exactly the points in the PLY. A cloud put
  back through `mode=restore` is therefore not a point-for-point match of the 3DFin input
  (`n_points_source` and `n_points_dropped` in the offsets file record the difference).

ForAINet's fifth class (`branches`) receives nothing and was removed from the framework
itself — those edits live in [`patches/forainet-local.patch`](patches/README.md), since
`ForAINet/` is a pinned submodule.

#### Checking the merge did what you meant

`class_unifier.py` applies the same map and writes an inspection copy of each plot to
`SegmentedForests/ClassUnifier_output/`, keeping the **source label next to the unified
one** so both can be seen in one file:

```bash
mamba run -n aifor python class_unifier.py                     # all plots -> .ply
mamba run -n aifor python class_unifier.py class_unifier.plots=[plot_10]
```

Open the result in [`misc/view_split_point_cloud.ipynb`](misc/README.md) and colour by
`Class` in one panel and `semantic_seg` in the other.

`output_format` chooses what it writes — **`ply` (default)**, `npy`, or `both`; the
`.json` sidecar with the provenance (map used, drop counts, per-class counts before and
after) is written either way. Prefer the PLY for viewing: the viewer only draws class
colours, a legend and unique counts for **integer** columns, and the `.npy` is a single
float64 matrix, so every label in it renders as a continuous ramp instead. The PLY is
also smaller (~32 vs 56 bytes/point — 1289 MB vs 2256 MB for plot_10).

---

## Environment and prerequisites

Runs inside the **`aifor`** mamba/conda environment, which provides both the Python deps and
the 3DFin executable.

- **3DFin binary** (`threedfin_bin` in `conf/config.yaml`):
  `C:/ProgramData/miniforge3/envs/aifor/Scripts/3DFin.exe`
  On Windows a console entry point becomes a `Scripts\*.exe` launcher — the equivalent of the
  Linux `envs/aifor/bin/3DFin`.
- **Paths are machine-specific.** `pointclouds_dir`, `ini_dir`, `output_dir` and
  `threedfin_bin` in `conf/config.yaml` must point at this machine's dataset and env. Edit them
  there, or override on the command line (see below).

---

## Stage 1 — run the 3DFin batch

Run from inside the `assignment/` folder, within the `aifor` env. Overrides use the
`threedfin.` section prefix (the stage's keys live under `threedfin:` in `config.yaml`):

```bash
# Full batch (all plots)
mamba run -n aifor python main_pipeline.py

# Dry run — print the 3DFin command for each plot, write nothing
mamba run -n aifor python main_pipeline.py threedfin.dry_run=true

# A single plot (or a list)
mamba run -n aifor python main_pipeline.py threedfin.plots=[plot_11]
mamba run -n aifor python main_pipeline.py threedfin.plots=[plot_01,plot_05]

# Override any config key on the command line
mamba run -n aifor python main_pipeline.py threedfin.pointclouds_dir=/other/path threedfin.normalize=true
```

Useful keys under the `threedfin:` section of `conf/config.yaml`:

| Key | Meaning |
|---|---|
| `pattern` | Glob for selecting input clouds (default `*.laz`). |
| `plots` | `null` = all plots; or a list like `[plot_01, plot_05]`. |
| `normalize` / `denoise` / `export_txt` | `null` = derive from each `.ini`; `true`/`false` = force for all plots. |
| `prune_outputs` | `true` (default) = after a successful run, keep only `<plot>_tree_ID_dist_axes.las` + `3dfin_log.txt`; `false` = keep the full 3DFin output set. |
| `dry_run` | `true` = log commands only, no files written, no 3DFin run. |
| `continue_on_error` | `true` (default) = keep going after a plot fails; `false` = stop at first failure. |

---

## Outputs of a Stage 1 run

**1. Hydra run directory** — `outputs/<date>/<time>/`, created per invocation:
- `.hydra/` — the fully resolved config used for that run (config, overrides, hydra config).
- `main_pipeline.log` — the run's INFO log (per-plot progress + final summary).

**2. Per-plot results** — `SegmentedForests/3DFin_output/<plot>/`.

By default (`prune_outputs=true`) the folder is reduced, after a successful run, to just:
- `<plot>_tree_ID_dist_axes.las` — the input cloud enriched with per-point `tree_ID` and
  `dist_axes` (the only product this project needs).
- `3dfin_log.txt` — captured 3DFin stdout/stderr (starts with the exact command line).

3DFin actually emits ~10 files; with `prune_outputs=false` they are all retained:
- `3dfin_params.ini` — the **patched** parameter file passed to 3DFin.
- `<plot>.xlsx` — the metrics workbook (per-tree measurements).
- `.las` products — `<plot>_dtm_points`, `_stripe`, `_circ`, `_axes`, `_tree_ID_dist_axes`,
  `_tree_heights`, `_tree_locator` (DTM, detected stems, fitted circles/axes, tree IDs,
  heights, locators), plus a `<plot>_config.ini`.

---

## Stage 2 — prepare plots for ForAINet (center + PLY) / restore results

ForAINet needs binary PLY inputs with **non-negative coordinates** and the vertex fields
`x, y, z, intensity, semantic_seg, treeID`. `forainet_prep.py` bridges the gap, in one of
two modes. **The input format does not matter**: `.las` (any version 1.2–1.4), `.laz` or
`.ply` are processed identically, whether they sit directly in `input_dir` (flat folder)
or one level down in per-plot subfolders (the 3DFin output tree from Stage 1).

**`mode=preprocess`** (default, before classification) — for every matched cloud:

1. Computes the per-plot **coordinate shifts** `offset_x/y/z = min(x/y/z)` and subtracts
   them (**centering** — every coordinate becomes ≥ 0, with 0 at the plot corner).
2. Writes `SegmentedForests/ForAINet_input/<plot>.ply` with the ForAINet fields —
   `semantic_seg` from the first of `Class`/`semantic_seg`/`classification` found,
   `treeID` from `tree_ID`/`treeID`, `intensity` from the source (zeros + warning when
   a field is absent). For data with other ground-truth field names, set
   `semantic_field` / `tree_id_field` / `intensity_field` explicitly.
3. Saves `<plot>_offsets.yml` next to the PLY: the shifts, validation statistics
   (original mins/ranges, point count) **and the source metadata needed for a faithful
   restoration** — source format/file, LAS version, point format, scales, and the CRS
   as WKT (`null` for PLY sources or clouds without a CRS, like the current dataset).

**`mode=restore`** (after classification) — for every `*.ply` in `restore_dir`: loads the
plot's `<plot>_offsets.yml`, **adds the shifts back** to x/y/z (validating restored
minima/ranges against the recorded statistics), and writes `restored_<plot>.laz` —
**compressed LAS 1.4** with the source scales and CRS re-applied, and every extra vertex
field (`semantic_seg`, `treeID`, predictions, …) preserved as LAS extra-bytes dimensions.

```bash
# Preprocess all plots / dry run / a single plot
mamba run -n aifor python forainet_prep.py
mamba run -n aifor python forainet_prep.py forainet_prep.dry_run=true
mamba run -n aifor python forainet_prep.py forainet_prep.plots=[plot_02]

# Restore classified results back to original coordinates (-> restored_<plot>.laz)
mamba run -n aifor python forainet_prep.py forainet_prep.mode=restore \
    forainet_prep.restore_dir=/path/to/classified_plys
```

Key keys under the `forainet_prep:` section of [`conf/config.yaml`](conf/config.yaml):

| Key | Meaning |
|---|---|
| `mode` | `preprocess` (input clouds → centered PLYs + offset files) or `restore` (classified PLYs → original coordinates). |
| `input_dir` / `patterns` / `output_dir` | Preprocess: input root (flat or per-plot subfolders), the globs matched in both layouts (default `["*.las", "*.laz", "*.ply"]`), and the folder receiving `<plot>.ply` + `<plot>_offsets.yml`. Narrow `patterns` to `["*_tree_ID_dist_axes.las"]` when plot folders hold several clouds (e.g. `prune_outputs=false`). |
| `semantic_field` / `tree_id_field` / `intensity_field` | Preprocess: names of the source fields holding the ground-truth label, tree id, and intensity. `null` = auto-detect (`Class`/`semantic_seg`/`classification`, `tree_ID`/`treeID`, `intensity`/`scalar_Intensity`/…); set explicitly for data with other field names — a configured name missing from a file fails that plot loudly. |
| `restore_dir` | Restore: folder of classified `*.ply` files (required; `restored_*` files are ignored). |
| `offsets_dir` | Restore: where the `<plot>_offsets.yml` live; `null` = `output_dir`. |
| `restore_format` | `laz` (default) = compressed LAS 1.4, source scales + CRS re-applied, all fields kept as extra dims; `las` = same, uncompressed; `ply` = plain PLY. |
| `plots` / `overwrite` / `dry_run` / `continue_on_error` | Same semantics as in Stage 1. |

> **Planned:** semantic-label reclassification (merging classes and assigning new class
> numbers) via a `class_map` config key — it will plug into `_semantic_labels()` in
> [`pipeline/forainet_prep.py`](pipeline/forainet_prep.py), where the `Class` →
> `semantic_seg` assignment is isolated.

---

## `misc/` — standalone tools outside the pipeline

The main approach works on whole plots (semantic segmentation), so per-tree data is not
part of the pipeline. Tools for other approaches live in `misc/`, each self-contained
with its own config:

- **`misc/tree_splitter.py`** — splits each 3DFin plot cloud
  (`<plot>_tree_ID_dist_axes.las`) into **one `.npy` file per tree**, saved next to the
  input in the plot's folder. Configured by `misc/conf/config.yaml`
  (`tree_splitter:` section, same override prefix as before):

```bash
mamba run -n aifor python misc/tree_splitter.py                              # all plots
mamba run -n aifor python misc/tree_splitter.py tree_splitter.dry_run=true  # print plan only
```

- **`misc/Points2ForAINet.py`** — the original standalone argparse converter
  (LAS/LAZ ↔ PLY with offset management) that Stage 2 was ported from; kept for ad-hoc
  conversions of files outside the plot layout.

See [`misc/README.md`](misc/README.md) for the full description and config keys.

---

## Checking progress and status

A batch is trackable entirely from the logs and the output tree:

- **Live progress:** the Hydra `main_pipeline.log` (and console) logs `running 3DFin …` when a plot
  starts and `done` / `FAILED` when it finishes. The final line summarises the whole run:
  `Processed N plot(s): X ok, Y failed, Z skipped.`
- **Which plots are complete:** a plot is done when
  `SegmentedForests/3DFin_output/<plot>/` contains the `<plot>.xlsx` plus the `.las` product
  set. A `<plot>/` folder with only a partial set (or a non-empty `3dfin_log.txt` ending in an
  error) indicates a failed/interrupted plot.
- **Failures are non-fatal by default:** with `continue_on_error=true`, a failed plot is logged
  and the batch moves on; its details are in that plot's `3dfin_log.txt`. Re-running the
  pipeline reprocesses plots and **overwrites** their output folders.
