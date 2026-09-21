# Forest Point-Cloud Semantic Segmentation Pipeline

A small, self-contained batch pipeline that takes a folder of forest LiDAR plots to
ForAINet-ready semantic-segmentation inputs — and brings the classified results back.
It is configured and launched with
[hydra-zen](https://mit-ll-responsible-ai.github.io/hydra-zen/); each stage has its own
entry script reading its own section of the shared `conf/config.yaml`:

- **Stage 1 — 3DFin instance segmentation** (`pipeline/threedfin.py`): drives the
  [3DFin](https://github.com/3DFin/3DFin) v0.6.0 CLI over every plot (`*.laz` +
  matching `<stem>.ini`), producing one instance-segmented cloud per plot.
- **Stage 2 — ForAINet preparation** (`forainet_prep.py`): centers each cloud's
  coordinates, writes ForAINet-ready PLYs + per-plot offset/metadata files, and
  **restores** classified results to original coordinates (LAZ 1.4, CRS re-applied).
- **The chain** (`main_pipeline.py`): runs both stages **per plot** and deletes each
  3DFin intermediate as soon as Stage 2 has consumed it — see
  [Running the pipeline](#running-the-pipeline). One command takes the raw clouds to
  the PLYs ForAINet trains on.

The development history — what was added when and why — is tracked in
[`docs/progress.md`](docs/progress.md).

> The sibling `ForAINet/` folder is the **deep-learning framework** this pipeline feeds
> (panoptic point-cloud segmentation) — a pinned git submodule of
> [prs-eth/ForAINet](https://github.com/prs-eth/ForAINet). Get it with
> `git submodule update --init`. Our own analysis of it is in
> [`docs/ARCHITECTURE_ANALYSIS.md`](docs/ARCHITECTURE_ANALYSIS.md) (the model and data
> pipeline), [`docs/train_logs_explained.md`](docs/train_logs_explained.md) (a training
> run, line by line) and [`docs/eval_process.md`](docs/eval_process.md) (evaluation and
> every metric it reports). Our local edits to the submodule live in
> [`patches/forainet-local.patch`](patches/README.md) — verify they are still applied with
> `python misc/check_forainet_classes.py`.

> **Project scope.** The main approach here is **semantic segmentation resolved at
> whole-plot level**, so the pipeline stops at the plot-level 3DFin output. Standalone
> tools that fall outside that scope (e.g. splitting plots into per-tree files) live in
> [`misc/`](misc/README.md).

---

## Files and what they do

| File | Role |
|---|---|
| [`main_pipeline.py`](main_pipeline.py) | **Whole-pipeline entry point.** `@hydra.main` loads the shared `conf/config.yaml` and runs the chain from its `chain:` section. (It reads all three sections at once, so unlike the other entry scripts it converts the config itself instead of using `zen()`, which maps a *single* section onto a function.) |
| [`pipeline/chain.py`](pipeline/chain.py) | **Chain engine.** `run_chain(...)` calls the two stage engines **per plot** and deletes each 3DFin `.las` once Stage 2 has succeeded for that plot, so the transient cost is one intermediate rather than all fourteen. Adds no processing of its own. Helpers: `_call` (invoke an engine with its config section, filtered to the parameters it accepts), `_keeps` (the three-way `keep_intermediates` switch), `_delete_clouds`. |
| [`pipeline/threedfin.py`](pipeline/threedfin.py) | **Stage 1 engine.** `run_pipeline(...)` collects `*.laz` files, pairs each with its `.ini`, builds and executes the `3DFin cli ...` command per plot, and prints a summary. Helpers: `_read_misc_flags` (read `[misc]` flags from an `.ini`), `_write_patched_ini` (write the patched copy — see quirks below), `_build_command` (assemble the argument vector). |
| [`forainet_prep.py`](forainet_prep.py) | **Stage 2 entry point.** Same pattern as `main_pipeline.py`, running the ForAINet preparation from `config.yaml`'s `forainet_prep:` section via `zen(prep_forainet)(cfg.forainet_prep)`. See "Stage 2" below. |
| [`pipeline/forainet_prep.py`](pipeline/forainet_prep.py) | **Stage 2 engine.** `prep_forainet(...)` centers each plot cloud (subtracts min x/y/z), writes ForAINet-ready PLYs + per-plot `<plot>_offsets.yml`, and can **restore** classified results to original coordinates. |
| [`pipeline/__init__.py`](pipeline/__init__.py) | Package init; re-exports `run_pipeline` and `prep_forainet`. |
| [`conf/config.yaml`](conf/config.yaml) | **Pipeline config.** One section per stage — `chain:`, `threedfin:` (Stage 1), `class_unifier:` (diagnostics) and `forainet_prep:` (Stage 2) — plus the shared `paths:` and `classes:` blocks both label stages interpolate. Override any key with its section prefix, e.g. `threedfin.plots=[plot_01]`. |
| [`misc/`](misc/README.md) | **Standalone tools outside the pipeline.** `misc/tree_splitter.py` (per-tree `.npy` splitting, own `misc/conf/config.yaml`) and `misc/Points2ForAINet.py` (the original generic converter Stage 2 was ported from). |
| [`requirements.txt`](requirements.txt) | Pinned Python deps (`hydra-zen`, `hydra-core`, `omegaconf`, `laspy`, `lazrs`, `plyfile`, `PyYAML`), all already satisfied by the `aifor` env. **3DFin itself is an external CLI, not a pip dependency of this package.** |
| [`docs/progress.md`](docs/progress.md) | **Development history** — what functionality was added when, and what is planned. |
| [`docs/report_methodology.md`](docs/report_methodology.md) · [`docs/report_results.md`](docs/report_results.md) | **The write-up**, for a reader rather than a maintainer: materials, technologies and the changes made to ForAINet; then the training, the backbone comparison, the comparison with the publication, and the biggest challenges. Start here if you want the project rather than the code. |
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

| SegmentedForests `Class` | → `semantic_seg` | Meaning |
|---|---|---|
| 8, 9, 11 | **0** | unclassified — no ForAINet counterpart |
| 0, 4, 5, 6, 7, 12, 13, 22, 23 | **1** | low_vegetation |
| 1 | **2** | ground |
| 3, 10 | **3** | stem_points |
| 2 | **4** | live_branches |

Verified against the data: the union of `Class` over all 14 plots is exactly those 16
values (individual plots differ — `plot_01` has only `0–4`, `plot_10` has
`0–5, 11, 12, 13, 22, 23`).

Two consequences worth knowing:

- **The map must be total.** A class present in the data but missing from `class_map`
  **fails that plot** rather than being silently kept or forced to a default — an
  unmapped label would otherwise reach training as a bogus class.
- **The labels are written 1-based, and no points are removed.** ForAINet's reader
  subtracts one on load (`semantic_seg - 1`), so what it sees is `0`–`3` for the four real
  classes and `−1` = `IGNORE_LABEL` for *unclassified*. Its losses pass that as
  `ignore_index`: those points contribute no loss and no gradient, but they stay in the
  cloud and keep supporting their neighbours' features. Removing them instead would punch
  holes in the geometry. A cloud put back through `mode=restore` is therefore a
  point-for-point match of the 3DFin input (`n_points_dropped` is 0). To actually delete a
  class, map it to `drop_value` (`-1`) — see
  [Configuration reference → `classes:`](#classes--the-label-scheme).

`class_names` in the config gives the same five names, and every `.json`/`.yml` sidecar
carries a copy. ForAINet's own fifth class (`branches`) has no SegmentedForests
counterpart and was removed from the framework itself — those edits live in
[`patches/forainet-local.patch`](patches/README.md), since `ForAINet/` is a pinned
submodule.

### Train / val / test — carried by the file name

ForAINet has no split manifest. It decides from the **file name**, testing
`name[-7:-4] == "val"`, then `name[-8:-4] == "test"`, and treating anything else as
training data. Stage 2 therefore writes `<plot>_train.ply`, `<plot>_val.ply` or
`<plot>_test.ply` — without a suffix every plot would silently become training data.

Plots are ranked by point count and the **smallest are held out**, keeping the bulk of
the points for training. The `split:` block under `forainet_prep:` controls it:

```yaml
split:
  n_val: 1        # a count, or a fraction of the plots: 0.1 = 10%
  n_test: 2
```

On the 14 plots that gives `plot_11 → val`, `plot_14` and `plot_01 → test`, the other 11
training. Because each value may be a fraction, the same block fits a dataset with a
different number of plots. `split: null` writes plain `<plot>.ply`.

Two behaviours worth knowing:

- The assignment is computed over **every matched cloud, before** `plots` narrows the
  run — so processing one plot still gives it the suffix a full run would.
- If a plot changes split, its previous file is **deleted**. ForAINet globs
  `raw/**/*.ply`, so a plot left with two suffixes would be loaded into two sets at once
  and train on its own test data.

`<plot>_offsets.yml` keeps the plain plot name and records the split, so `restore` still
finds it.

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

## Running the pipeline

### The short version

```bash
mamba run -n aifor python main_pipeline.py
```

That takes the unprocessed clouds in `SegmentedForests/pointclouds/` all the way to the
training clouds, **one plot at a time**: 3DFin, then Stage 2, then the plot's 3DFin
`.las` is deleted because nothing needs it any more. Add `chain.dry_run=true` first to
see the plan (and each 3DFin command) without writing anything.

The result is **LAZ** in `SegmentedForests/ForAINet_export/` — **4.32 GB against 29.21 GB
of PLY, measured over all 14 plots** — and ready to upload. To train on this machine
instead, ask for PLY straight into ForAINet's data folder:

```bash
mamba run -n aifor python main_pipeline.py \
    forainet_prep.output_format=ply forainet_prep.output_dir='${paths.forainet_raw}'
```

### Why it works plot by plot

Run as separate stages, each hands the next a **complete folder**. Measured on the 14
SegmentedForests plots, for a 4.4 GB input:

| folder | size |
|---|---|
| `SegmentedForests/pointclouds/` (the input) | 4.4 GB |
| `SegmentedForests/3DFin_output/` | 46.3 GB |
| `SegmentedForests/ClassUnifier_output/` (diagnostics, optional) | 25.3 GB |
| `…/treeinsfused/raw/` (what ForAINet trains on) | ~25 GB |

3DFin writes all fourteen `.las` files before Stage 2 reads the first one, and every one
of them is dead weight the moment Stage 2 has consumed it. The chain instead keeps
**one** intermediate alive at a time — 15.2 GB for the biggest plot, rather than 46.3 GB
for all of them.

3DFin is an external command-line program, so it can only hand over its result as a
file; there is no way to avoid writing it. Consuming it immediately is the next best
thing.

**The price:** re-running Stage 2 alone later — a changed class map, a changed split —
needs that `.las` back, which means re-running 3DFin (hours, not minutes). While you are
still tuning the class scheme, set `chain.keep_intermediates=true`.

### Which command produces what

Every stage still has its own entry script and can be run on its own. Pick the row you
need:

| Run | Requires on disk | Produces | Disk |
|---|---|---|---|
| `python main_pipeline.py` | `pointclouds/*.laz` + `3DFin_settings/*.ini` | LAZ export + `<plot>_offsets.yml` | ≤15 GB transient, 4.3 GB kept |
| `python main_pipeline.py chain.stages=[threedfin]` | same | `3DFin_output/<plot>/*.las` (nothing deleted) | 46 GB |
| `python main_pipeline.py chain.stages=[forainet_prep]` | `3DFin_output/**/*.las` | LAZ export + offsets, **consuming** the `.las` | frees 46 GB |
| `python forainet_prep.py` | `3DFin_output/**/*.las` | LAZ export + offsets, keeping the `.las` | 4.3 GB |
| `python convert.py` | the LAZ export | the PLYs ForAINet trains on | 29.2 GB |
| `python class_unifier.py` | `3DFin_output/**/*.las` | diagnostic PLYs (source + unified labels side by side) | ~25 GB |
| `python forainet_prep.py forainet_prep.mode=restore forainet_prep.restore_dir=…` | classified PLYs + `<plot>_offsets.yml` | `restored_<plot>.laz` in original coordinates | — |

`python forainet_prep.py` and `python main_pipeline.py chain.stages=[forainet_prep]` do
exactly the same work — the chain form processes plots one at a time and deletes as it
goes, the standalone form runs several at once (memory-budgeted) and keeps everything.

### Re-entering at a stage

```bash
# Resume an interrupted run: chain.overwrite is false by default, so plots whose
# training cloud already exists are skipped without re-running 3DFin.
mamba run -n aifor python main_pipeline.py

# Redo one plot from scratch
mamba run -n aifor python main_pipeline.py chain.plots=[plot_02] chain.overwrite=true

# Stage 1 only, keeping the .las (this is what main_pipeline.py did before the chain)
mamba run -n aifor python main_pipeline.py chain.stages=[threedfin]

# Changed the class map? Stage 2 alone needs the .las back. Either it was kept, or:
mamba run -n aifor python main_pipeline.py chain.stages=[threedfin]   # hours
mamba run -n aifor python forainet_prep.py                            # minutes

# Inspect a class merge (class_unifier reads tree_ID, which only 3DFin produces),
# keeping just that plot's intermediate rather than all 46 GB:
mamba run -n aifor python main_pipeline.py chain.plots=[plot_02] chain.keep_intermediates=[plot_02]
mamba run -n aifor python class_unifier.py class_unifier.plots=[plot_02]
```

> **Delete `…/treeinsfused/processed_0.2/` whenever `raw/` changes.** ForAINet caches
> its preprocessed tensors there and will happily keep training on the stale ones.

Nothing is deleted unless `forainet_prep` is in `stages` **and** it reported that plot as
succeeded. A failure leaves the expensive `.las` in place, so the retry costs minutes.

The keys are in [Configuration reference → `chain:`](#chain--which-stages-run-and-what-survives).

### Training on a rented GPU

> **Step-by-step runbook: [`docs/gpu_training_runbook.md`](docs/gpu_training_runbook.md).**
> Local smoke test → Docker Hub → Cloudflare R2 → Vast.ai → two trained models (one per
> backbone) → downloading the results. The rest of this section explains *why* the export
> is LAZ; the runbook is the procedure.

This is why Stage 2 exports LAZ. A binary PLY is an uncompressed memory dump, so the
clouds ForAINet reads are **29.21 GB against 4.32 GB for the same points as LAZ** —
6.8× measured over all 14 plots — and on a rented machine every one of those bytes
crosses a network you are paying for and waiting on. The export travels compressed and
is expanded on the GPU box, where the disk is already rented and the conversion takes
seconds (14 plots in ~2 minutes).

```
LOCAL                                  object store              GPU INSTANCE
main_pipeline.py                                                 python -m pipeline.convert \
  3DFin -> Stage 2 -> plot_XX_<split>.laz  ---- upload ---->        --to ply <in> <raw/>
                    + plot_XX_offsets.yml                                    |
                                                                             v
                                                              ForAINet trains on raw/**/*.ply
```

1. **Export.** `mamba run -n aifor python main_pipeline.py` → `SegmentedForests/ForAINet_export/`.
2. **Bundle**, if you would rather push one object than fifteen. LAZ is already
   compressed, so use a plain tar — a zip would only re-pack it:
   `tar -cf export.tar *.laz *_offsets.yml`
3. **Upload** to object storage, and push your training image to a registry.
4. **On the instance**, pull both, then convert and train:
   ```bash
   python -m pipeline.convert --to ply /data/export \
       /workspace/PointCloudSegmentation/data_set1_5classes/treeinsfused/raw/SegmentedForests
   ```
5. **Retrieve the checkpoints and predictions** — not the data. See below.

Four things that will bite otherwise:

- **Allocate ≥60 GB of instance disk.** The LAZ expands to ~29 GB of PLY, ForAINet then
  builds a ~2 GB tensor cache in `processed_0.2/`, and checkpoints land on top. Running
  out mid-preprocessing looks like a framework crash.
- **Check CUDA compatibility before renting.** `ForAINet/` is pinned at `5fe600a` with
  older `torch-points3d`/`torch-scatter` builds, which will not compile against the
  newest GPU architectures. Filter by a generation your image actually supports.
- **Never bake credentials into the image** — it goes to a registry. Pass storage keys as
  environment variables on the instance.
- **Ship the `<plot>_offsets.yml` files with the export.** They are a few KB each and are
  the only thing that can put predictions back into real-world coordinates.

**Do not download the training data afterwards.** Those clouds contain no original
information — every byte is derived from the 4.4 GB of source LAZ plus this repo. What is
irreplaceable is the **checkpoint**, the run config and logs, and the **predictions on
val/test**, which `forainet_prep.py forainet_prep.mode=restore` turns back into
georeferenced LAZ. `processed_0.2/` is a cache: never upload it, never retrieve it, and
**delete it locally whenever `raw/` changes** or ForAINet will train on the stale one.

---

## Configuration reference

Everything is set in the single [`conf/config.yaml`](conf/config.yaml). It has one
top-level section per stage plus two shared blocks, and each entry script reads only its
own section. Any key can be overridden on the command line with its section prefix:

```bash
mamba run -n aifor python main_pipeline.py chain.plots=[plot_01] threedfin.normalize=true
```

Overrides last for that one run; edit the file to make them stick.

### Keys every stage has

These appear in `chain:`, `threedfin:`, `class_unifier:` and `forainet_prep:` with the
same meaning, so they are described once here rather than in every table below.

| Key | Default | What it does |
|---|---|---|
| `plots` | `null` | `null` = process everything found; or a list like `[plot_01, plot_05]`. Names are plot stems, without the `.laz`. |
| `overwrite` | varies | `false` = skip a plot whose output already exists, which is what makes an interrupted run resumable; `true` = redo it. `chain:` defaults to `false` (a redo can cost hours of 3DFin), `class_unifier:` and `forainet_prep:` to `true`. **`threedfin:` has no such key** — 3DFin always re-runs, so the chain's skip is the only guard against repeating it. |
| `dry_run` | `false` | Log what *would* happen — including each 3DFin command line — without reading point data, writing files or deleting anything. Worth doing before any long run. |
| `continue_on_error` | `true` | Keep going after a plot fails. `false` stops at the first one. |

### Keys the parallel stages have

`class_unifier:` and `forainet_prep:` process several plots at once in worker processes
(see [`pipeline/parallel.py`](pipeline/parallel.py)). 3DFin stays sequential, and the
chain hands Stage 2 one plot at a time, so these do nothing there.

| Key | Default | What it does |
|---|---|---|
| `workers` | `null` | `null` = as many plots at once as the memory budget allows, capped by CPU count; an integer caps it; `1` runs everything inline in this process. |
| `memory_budget_frac` | `0.7` | Fraction of total RAM the run may commit. The rest is headroom for the OS. |
| `bytes_per_point` | `140` | Measured peak RSS per point, used to size each plot's share. |

**Concurrency here is bounded by RAM, not cores.** These clouds run from 19 M to 280 M
points at ~140 bytes/point (measured: plot_09, 72.6 M points → 9.1 GB peak), so the
biggest needs ~39 GB on its own and a worker per core would exhaust the machine. Each
plot's size is read from its LAS header (instant, no point data) and work is admitted
only while it still fits the budget — a huge plot runs nearly alone while small ones pack
several deep. Lower `memory_budget_frac` if other work is competing for memory; raise
`bytes_per_point` if you see the machine swap.

### `paths:` — machine-specific locations

| Key | What it is |
|---|---|
| `assignment` | This repository's root. Every other path is built from it, so on another machine this is usually the only line to change. |
| `forainet` / `forainet_dataroot` | The ForAINet submodule checkout and the dataset root inside it. |
| `forainet_raw` | Where ForAINet looks for the PLYs it trains on. |
| `export` | Where Stage 2 writes its LAZ export — the folder you upload. |
| `offsets` | Where the `<plot>_offsets.yml` files live — deliberately outside the submodule. |

**`forainet_raw` is not a free choice.** ForAINet assembles that path itself and then
globs it:

```
base_dataset.py    _data_path = <dataroot>/<dataset_name>
                   ...where dataset_name falls back to the DATASET CLASS NAME, lower-cased
                   and minus "dataset" -> "treeinsfused". That part is baked into the
                   Python, not configurable.
torch_geometric    raw_dir    = <_data_path>/raw
treeins_set1.py    glob(raw_dir + "/**/*.ply", recursive=True)
```

`dataroot` (`data_set1_5classes`) comes from ForAINet's *own*
`conf/data/panoptic/treeins_set1.yaml` — change it there and change it here too. The
final folder (`SegmentedForests`) *is* free, because the glob is recursive; it just groups
our plots the way the reference data groups NIBIO2/CULS/SCION. Since this sits inside the
pinned submodule, that checkout has to exist (`git submodule update --init`) before
Stage 2 can write.

`offsets` stays in the repo on purpose: those files are small, tracked by git, and the
only thing that makes `mode=restore` possible, so they should not be buried in a data
folder that gets wiped when datasets change.

### `classes:` — the label scheme

Not a stage — there is no engine behind it. It is plain config, interpolated by **both**
stages that touch labels (`class_unifier:` and `forainet_prep:`), so the inspection
exports and the clouds ForAINet actually trains on can never disagree about what a class
number means. Pointing the pipeline at another dataset means a different `class_map` and
field names, not a code change.

| Key | Default | What it does |
|---|---|---|
| `semantic_field` | `Class` | Source column holding the ground-truth label. Stated explicitly rather than auto-detected: silently picking the wrong column is exactly the failure this block exists to prevent. |
| `tree_id_field` | `tree_ID` | Source column holding the tree/instance id. |
| `unified_field` | `semantic_seg` | Name of the **new** column that receives the remapped labels. |
| `drop_value` | `-1` | Points whose mapped label equals this are physically removed. `null` keeps every point. Currently inert — see below. |
| `class_map` | 16 entries | `{source: unified}`. Several sources sharing a target merge those classes. See [Semantic classes](#semantic-classes-segmentedforests--forainet). |
| `class_names` | 5 entries | The unified scheme, echoed into every `.json`/`.yml` sidecar. |
| `instance_classes` | `[3, 4]` | Which classes are made of individual trees. `null` passes tree ids through untouched. |

**`drop_value` is deliberately inert.** Nothing in `class_map` targets `-1`, so nothing is
dropped. The classes with no ForAINet counterpart (8, 9, 11) go to `0` = *unclassified*
instead, and ForAINet's reader subtracts one (`semantic_seg - 1`), turning `0` into
`-1` = `IGNORE_LABEL`. Its losses pass that as `ignore_index`, so those points contribute
no loss and no gradient but **stay in the cloud** — their geometry still supports the
neighbourhood features of every point around them. Deleting them instead would punch holes
in the cloud and change what nearby points see. The key is kept as a safety valve: map a
class to `-1` here if you ever do want it gone.

**`instance_classes` is why the instance head works at all.** Everything outside these
classes is "stuff" (ground, undergrowth) and must carry tree id `0` = "not part of any
tree" — the convention ForAINet's own files follow. 3DFin does not follow it: it gives
every point the id of the nearest stem, so ground and undergrowth beneath a tree inherit
that tree's id (measured: id 114189 covered 3.0 M points, 1.6 M of them ground and low
vegetation). ForAINet's `set_extra_labels()` then skips any instance whose id also appears
on a stuff point — which, with 3DFin ids, was **every single one: 0 of 42 tree instances
survived on plot_10**. Stage 2 therefore zeroes the tree id outside these classes.

> **Hydra gotcha — change this block by editing the file.** Its keys are *integers*, which
> makes CLI overrides fail in confusing ways (measured, not guessed):
>
> | Override | Result |
> |---|---|
> | `classes.class_map.5=1` | **Fails** `Key '5' is not in struct` — the override parser hands over a *string* key |
> | `classes.class_map={99: 0}` | **Fails** the same way |
> | `++classes.class_map={99: 0}` | Merges, adding that entry |
> | `class_unifier.class_map={…}` | **Replaces** that stage's map wholesale, severing the link to this block — so `forainet_prep` keeps the old one and the two stages silently disagree |
>
> Reserve the CLI for one-off experiments, never for a run whose output feeds training.

### `chain:` — which stages run, and what survives

Run with `python main_pipeline.py` (engine: [`pipeline/chain.py`](pipeline/chain.py)).
Plus the [shared run-control keys](#keys-every-stage-has).

| Key | Default | What it does |
|---|---|---|
| `stages` | `[threedfin, forainet_prep]` | Which stages, in order. `[threedfin]` alone = 3DFin only, nothing deleted (what `main_pipeline.py` meant before the chain existed). `[forainet_prep]` alone converts already-produced 3DFin output, consuming it as it goes. `null` = both. An empty list is an error. |
| `keep_intermediates` | `false` | `false` = delete each plot's 3DFin `.las` once Stage 2 has succeeded for it. `true` = keep them all — do this while you are still tuning the class scheme. A **list** of plot names keeps just those, which is how the diagnostics stay affordable (`class_unifier` needs the `tree_ID` only 3DFin produces, so inspecting one plot costs ~1.5 GB rather than 46 GB). |

Only point clouds are deleted — `3dfin_log.txt` stays, so each plot folder still records
its run. Nothing is deleted unless `forainet_prep` is in `stages` **and** reported that
plot as succeeded.

### `threedfin:` — Stage 1, 3DFin instance segmentation

Engine: [`pipeline/threedfin.py`](pipeline/threedfin.py). Reached through the chain;
`chain.stages=[threedfin]` runs it alone. Plus the
[shared run-control keys](#keys-every-stage-has).

| Key | Default | What it does |
|---|---|---|
| `pointclouds_dir` | `…/SegmentedForests/pointclouds` | Input clouds. |
| `ini_dir` | `…/SegmentedForests/3DFin_settings` | Where each plot's `<stem>.ini` parameter file lives. A plot with no matching `.ini` is skipped with a warning. |
| `output_dir` | `…/SegmentedForests/3DFin_output` | Results root — one subfolder per plot. |
| `threedfin_bin` | `…/envs/aifor/Scripts/3DFin.exe` | The 3DFin executable. On Windows a console entry point becomes a `Scripts\*.exe` launcher; on Linux it is `envs/aifor/bin/3DFin`. |
| `pattern` | `"*.laz"` | Glob selecting input clouds. |
| `normalize` / `denoise` / `export_txt` | `null` | Tri-state 3DFin CLI flags. `null` (recommended) derives each from that plot's `.ini` `[misc]` section; `true`/`false` forces it for every plot. Note 3DFin maps `is_normalized = not --normalize` — the dataset clouds are *not* height-normalized, so deriving from the `.ini` passes `--normalize`. |
| `prune_outputs` | `true` | After a successful run, delete everything 3DFin emitted for that plot except `<plot>_tree_ID_dist_axes.las` and `3dfin_log.txt` — the only file this project needs, plus its log. `false` keeps all ~10 files. |

### `class_unifier:` — diagnostics (optional)

Run with `python class_unifier.py` (engine:
[`pipeline/class_unifier.py`](pipeline/class_unifier.py)). Nothing downstream reads its
output; it exists so you can *look at* a class merge. Plus the
[shared run-control](#keys-every-stage-has) and [parallelism](#keys-the-parallel-stages-have) keys.

| Key | Default | What it does |
|---|---|---|
| `input_dir` | `…/3DFin_output` | Input root. Matched both directly (flat layout) and one level down in per-plot subfolders (the 3DFin output tree). |
| `patterns` | `["*.las", "*.laz", "*.ply"]` | Globs matched in both layouts. Any mix of LAS 1.2–1.4, LAZ and PLY is processed identically. |
| `output_dir` | `…/ClassUnifier_output` | Receives `<plot>.<ply\|npy>` + a `<plot>.json` sidecar (the map used, drop counts, per-class counts before and after). Created if missing. |
| `output_format` | `ply` | `ply`, `npy`, or `both`. The `.json` is written either way. |
| `semantic_field` / `tree_id_field` | from `classes:` | Interpolated from the shared block, so this stage and Stage 2 cannot disagree. |
| `intensity_field` | `null` | `null` = auto-detect (`intensity`, `scalar_Intensity`, …). Intensity has no bearing on the label scheme, so it stays local rather than living in `classes:`. A *configured* name missing from a file fails that plot loudly instead of silently writing zeros. |
| `unified_field` / `class_map` / `drop_value` / `class_names` | from `classes:` | The unification itself. |

**Use `ply` for viewing.** A PLY keeps each column's real dtype, and
[the viewer](misc/README.md) only draws class colours, a legend, unique counts and
per-value histogram bars for **integer** columns. A `.npy` is a single float64 matrix, so
every label in it renders as a continuous ramp instead — which defeats the point of
checking a merge. The PLY is smaller too (~32 vs 56 bytes/point). `npy` stays available
for loading straight into numpy.

Coordinates stay **original** here — centering belongs to Stage 2, which also writes the
offsets file that makes it reversible.

### `forainet_prep:` — Stage 2, centering and restore

Run with `python forainet_prep.py` (engine:
[`pipeline/forainet_prep.py`](pipeline/forainet_prep.py)); also called per plot by the
chain, from this same section. Plus the [shared run-control](#keys-every-stage-has) and
[parallelism](#keys-the-parallel-stages-have) keys.

| Key | Default | What it does |
|---|---|---|
| `mode` | `preprocess` | `preprocess` = input clouds → centered training clouds + `<plot>_offsets.yml`; `restore` = classified PLYs → original coordinates. |
| `input_dir` / `patterns` | `…/3DFin_output`, all three extensions | Preprocess input, same two-layout matching as `class_unifier`. Narrow `patterns` to `["*_tree_ID_dist_axes.las"]` when a plot folder holds more than one matching cloud (e.g. `prune_outputs: false`). |
| `output_dir` | `${paths.export}` | Where the training clouds land. Pair with `output_format`: the LAZ export goes to `${paths.export}` and is uploaded; a `ply` run belongs in `${paths.forainet_raw}`, where ForAINet globs for it. |
| `output_format` | `laz` | `laz` (default), `las` or `ply` — the container only; the points are identical. **A binary PLY is an uncompressed memory dump** (measured 37.00 B/pt; 29.21 GB against 4.32 GB of LAZ over the 14 plots), so the default keeps the upload to a rented GPU small. ForAINet reads PLY only — see [`convert:`](#convert--laz--ply-optional). |
| `semantic_field` / `tree_id_field` / `intensity_field` | from `classes:`, `null` | As in `class_unifier:` above. |
| `class_map` / `drop_value` | from `classes:` | The PLYs written here are what ForAINet trains on, so they carry the unified labels; without this the framework would see the raw source classes. |
| `instance_classes` | from `classes:` | Points outside these classes get tree id 0 — see the note above. |
| `split` | `{n_val: 1, n_test: 2}` | Appends `_train`/`_val`/`_test` to each file name, which is the only way ForAINet learns what a cloud is for. Each value is a count (int) or a fraction of the plots (float < 1). `null` writes a plain `<plot>` name. See [Train / val / test](#train--val--test--carried-by-the-file-name). |
| `restore_dir` | `null` | Restore only: folder of classified `*.ply` to restore. **Required** for `mode=restore`; files already named `restored_*` are ignored. |
| `offsets_dir` | `${paths.offsets}` | Where `<plot>_offsets.yml` live, in **both** modes. `null` = beside the clouds. |
| `restore_format` | `laz` | `laz` = compressed LAS 1.4 with the source scales and CRS re-applied and every extra field kept as an extra-bytes dimension; `las` = the same, uncompressed; `ply` = plain PLY. |

### `convert:` — LAZ ↔ PLY (optional)

Run with `python convert.py` (engine: [`pipeline/convert.py`](pipeline/convert.py)).
Expands the LAZ export back into the PLYs ForAINet reads. Plus the
[shared run-control](#keys-every-stage-has) and
[parallelism](#keys-the-parallel-stages-have) keys.

| Key | Default | What it does |
|---|---|---|
| `input_dir` | `${paths.export}` | Folder of clouds to convert, matched flat and one level down. |
| `output_dir` | `${paths.forainet_raw}` | Destination. `null` writes beside the inputs. |
| `to` | `ply` | Target format — `ply`, `laz` or `las`. Files already in that format are skipped, so a second run is a no-op rather than a corruption. |
| `patterns` | `["*.laz", "*.las", "*.ply"]` | Globs to match. |
| `offsets_dir` | `${paths.offsets}` | Only consulted when writing **LAS/LAZ from a PLY**, to recover the source scales and CRS that a PLY cannot carry. |

This is a **pure format conversion** — no centring, no class remapping, no tree-id
zeroing. All of that already happened in Stage 2, whose output carries centred
coordinates (LAS header offset 0), the unified `semantic_seg` and the zeroed `treeID`.
Everything that changes what the model learns stays in the pipeline; this stage only
moves bytes between containers.

**The file stem is preserved**, and that is load-bearing: `plot_11_val.laz` becomes
`plot_11_val.ply`, because ForAINet reads the split from the file *name*. A prefix or a
rename silently moves a plot between train and test.

Inside a training container there is no Hydra, so use the engine's own CLI — it imports
only `laspy`, `lazrs`, `numpy`, `plyfile` and `PyYAML`:

```bash
python -m pipeline.convert --to ply /data/export \
    /workspace/PointCloudSegmentation/data_set1_5classes/treeinsfused/raw/SegmentedForests
```

[`misc/Points2ForAINet.py`](misc/Points2ForAINet.py) is the same CLI under its historic
name, runnable straight from a checkout.

**Fidelity, measured on plot_11** (18,945,417 points, round trip PLY → LAZ → PLY):
`semantic_seg`, `treeID` and `intensity` come back **bit-identical** — the labels travel
as LAS extra-bytes dimensions — and the largest coordinate deviation is
**3.6 × 10⁻¹⁵ m**, i.e. float64 rounding, because the centred coordinates already sit
exactly on the source 1e-07 grid. The model voxelises at 0.2 m.

> **If you ever need the export smaller still, the lever is the coordinate scale.** LAZ
> compresses the *integer* coordinate deltas, and the export inherits each plot's source
> scale. That is visible in the measured ratios: the plots stored at 1e-07 compress
> ~4×, those at 1e-06 ~8×, and plot_13 at 1e-05 reaches **11.9×**. A 1 mm scale would put
> every plot in that range — still 200× finer than the model's grid — at the cost of no
> longer being bit-exact against the source coordinates.

---

## Stage 1 — run the 3DFin batch

Stage 1 is normally reached through the chain above. To run it on its own — the full
batch, nothing deleted — use `chain.stages=[threedfin]`. Its own settings live under
`threedfin:` in `config.yaml` and take the `threedfin.` prefix:

```bash
# Full batch (all plots), 3DFin only
mamba run -n aifor python main_pipeline.py chain.stages=[threedfin]

# Dry run — print the 3DFin command for each plot, write nothing
mamba run -n aifor python main_pipeline.py chain.stages=[threedfin] chain.dry_run=true

# A single plot (or a list)
mamba run -n aifor python main_pipeline.py chain.stages=[threedfin] chain.plots=[plot_11]
mamba run -n aifor python main_pipeline.py chain.stages=[threedfin] chain.plots=[plot_01,plot_05]

# Override any of the stage's own config keys on the command line
mamba run -n aifor python main_pipeline.py chain.stages=[threedfin] threedfin.normalize=true
```

> **Changed:** `python main_pipeline.py` used to mean "3DFin only". It now runs the whole
> chain; `chain.stages=[threedfin]` is the old behaviour. Note that `chain.plots` selects
> the plots (the chain drives the loop), while `threedfin.*` still configures how each
> 3DFin run is made.

The keys are in
[Configuration reference → `threedfin:`](#threedfin--stage-1-3dfin-instance-segmentation).

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
2. Writes `<output_dir>/<plot>_<split>.ply` — by default
   `ForAINet/…/treeinsfused/raw/SegmentedForests/plot_02_train.ply`, straight into the
   folder ForAINet globs. The vertex fields are `x, y, z, intensity, semantic_seg,
   treeID`: `semantic_seg` remapped through `class_map` from the first of
   `Class`/`semantic_seg`/`classification` found, `treeID` from `tree_ID`/`treeID` and
   zeroed outside `instance_classes`, `intensity` from the source (zeros + warning when
   a field is absent). For data with other ground-truth field names, set
   `semantic_field` / `tree_id_field` / `intensity_field` explicitly.
3. Saves `<plot>_offsets.yml` — in `offsets_dir`, i.e. `SegmentedForests/ForAINet_input/`,
   kept out of the submodule and tracked by git. It holds the shifts, validation statistics
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

The keys are in
[Configuration reference → `forainet_prep:`](#forainet_prep--stage-2-centering-and-restore).

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
