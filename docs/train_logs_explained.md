# Reading a ForAINet training run — codebase map + log walkthrough

This document explains the terminal output captured in [`train_logs.md`](train_logs.md),
and — because you cannot decode the log without knowing what produced it — maps every
component of the pipeline to the file it lives in.

The run being explained:

```bash
python train.py task=panoptic data=panoptic/treeins_set1 \
  models=panoptic/FORpartseg_3heads model_name=PointGroup-PAPER \
  training=treeins_set1 job_name=debug
```

All paths below are relative to `ForAINet/PointCloudSegmentation/`.

**Contents**
- [Part A — where everything lives](#part-a--where-everything-lives) (the eight questions)
- [Part B — the log, line by line](#part-b--the-log-line-by-line)
- [Part C — what the numbers mean, and three anomalies](#part-c--what-the-numbers-mean-and-three-anomalies)

---

# Part A — where everything lives

## A0. The one-paragraph summary

This is **panoptic segmentation** of forest point clouds: every point gets a *semantic*
class (ground, stem, branch…) **and** points belonging to the same tree get a shared
*instance* id. The model is **PointGroup-style**: a sparse-convolution U-Net backbone
produces a 16-dimensional feature per point; three small heads read those features to
predict (1) the semantic class, (2) an offset vector pointing at the tree's centre, and
(3) an embedding used to cluster points into trees. A fourth "scorer" network rates each
proposed cluster.

## A1. Gdzie są configi — where are the configs

Everything is [Hydra](https://hydra.cc). The root file is
[`conf/config.yaml`](ForAINet/PointCloudSegmentation/conf/config.yaml), whose `defaults:`
list names one file from each subdirectory:

```
conf/
├── config.yaml                 ← root; defaults list + model_name + tracker_options
├── task/                       ← panoptic / segmentation / …
├── data/panoptic/              ← dataset definitions   (treeins_set1.yaml)
├── models/panoptic/            ← model + backbone shapes (FORpartseg_3heads.yaml)
├── training/                   ← epochs, batch size, optimizer, wandb (treeins_set1.yaml)
├── lr_scheduler/               ← exponential.yaml, …
├── visualization/, debugging/, hydra/
```

Each CLI argument `data=panoptic/treeins_set1` replaces one entry of that defaults list,
i.e. it selects **which file** is loaded into that config group.

**The `# @package` directive is the thing that trips people up.** Line 2 of
[`conf/training/treeins_set1.yaml`](ForAINet/PointCloudSegmentation/conf/training/treeins_set1.yaml#L2)
says `# @package training`, which means everything in that file is mounted under the
`training.` prefix. So the `wandb:` block inside it is addressed as
`training.wandb.*` — this is why disabling wandb is `training.wandb.log=False`, not
`wandb.log=False`. Same story for `# @package data` and `# @package models`.

**Config values can be arithmetic strings.** `1.5 * ${data.grid_size}` and
`2*in_feat` are literally `eval()`-ed at model-build time by
[`model_definition_resolver.py:29-58`](ForAINet/PointCloudSegmentation/torch_points3d/utils/model_building_utils/model_definition_resolver.py#L29-L58),
which also injects the magic constants `FEAT` (= number of input features), `N_CLS`, and
`TASK` ([lines 16-26](ForAINet/PointCloudSegmentation/torch_points3d/utils/model_building_utils/model_definition_resolver.py#L16-L26)).
That is how `[FEAT, in_feat]` in the yaml becomes `[4, 16]` in the built network.

## A2. Gdzie jest definicja datasetu / loadera — dataset and loader

**Config** → [`conf/data/panoptic/treeins_set1.yaml`](ForAINet/PointCloudSegmentation/conf/data/panoptic/treeins_set1.yaml),
which names `class: treeins_set1.TreeinsFusedDataset` and `task: panoptic`.

**Class hierarchy** (note the panoptic layer *wraps* the segmentation layer):

```
InMemoryDataset (PyTorch Geometric)
 └─ TreeinsOriginalFused      segmentation/treeins_set1.py:185   raw .ply I/O + caching
     └─ TreeinsSphere         segmentation/treeins_set1.py:569   random sphere sampling
         └─ TreeinsCylinder   segmentation/treeins_set1.py:673   random cylinder sampling ← used

PanopticTreeinsBase           panoptic/treeins_set1.py:494       mixin: adds instance labels
 └─ PanopticTreeinsCylinder(PanopticTreeinsBase, TreeinsCylinder)   panoptic/treeins_set1.py:555 ← used

BaseDataset
 └─ TreeinsFusedDataset       panoptic/treeins_set1.py:563       ← what the config instantiates
```

`PanopticTreeinsBase` is a **mixin placed first in the MRO**, so its `__getitem__` runs
first, calls `super().__getitem__(idx)` (the segmentation sampling + transforms), then
adds the instance-level labels
([panoptic/treeins_set1.py:502-527](ForAINet/PointCloudSegmentation/torch_points3d/datasets/panoptic/treeins_set1.py#L502-L527)).

> ⚠️ There are **two** classes named `TreeinsFusedDataset` — one in `panoptic/`, one in
> `segmentation/`. `task: panoptic` selects the panoptic one; the segmentation one is
> dead code for this run.

**Raw data**: `data_set1_5classes/treeinsfused/raw/NIBIO2/*.ply` (50 files). Each PLY must
contain the fields `x, y, z, semantic_seg, treeID`
([segmentation/treeins_set1.py:64-83](ForAINet/PointCloudSegmentation/torch_points3d/datasets/segmentation/treeins_set1.py#L64-L83)).
The train/val/test split comes from the **filename suffix**, not a fold number
([lines 380-385](ForAINet/PointCloudSegmentation/torch_points3d/datasets/segmentation/treeins_set1.py#L380-L385)):
29 `_train`, 6 `_val`, 15 `_test`.

**Processed cache**: `data_set1_5classes/treeinsfused/processed_0.2/` (the `0.2` is the
grid size). If this directory exists, PyG **skips `process()` entirely** — so changing
`first_subsampling` creates a new cache dir, but editing the raw PLYs does not invalidate
the old one.

**DataLoader** is created in
[`base_dataset.py:194-250`](ForAINet/PointCloudSegmentation/torch_points3d/datasets/base_dataset.py#L194-L250)
(`create_dataloaders`), called from
[`trainer.py:115-121`](ForAINet/PointCloudSegmentation/torch_points3d/trainer.py#L115-L121).
Because `conv_type: "SPARSE"`, the collate function is plain
`torch_geometric.data.Batch.from_data_list`
([base_dataset.py:159-175](ForAINet/PointCloudSegmentation/torch_points3d/datasets/base_dataset.py#L159-L175)) —
every tensor is concatenated along dim 0 and a `batch` index vector is synthesised.

## A3. Gdzie jest preprocessing — preprocessing

Preprocessing happens in **three** stages, which is why the log prints several different
transform chains.

**Stage 1 — `pre_collate_transform`** (once, cached to disk):

| Transform | File | Effect |
|---|---|---|
| `SaveOriginalPosId` | `grid_transform.py:286-308` | adds `origin_id` so predictions can be mapped back to the original full-resolution cloud |
| `GridSampling3D(0.2, quantize_coords=False)` | `grid_transform.py:151-210` | voxel-subsamples the whole plot to a 0.2 m grid |

**Stage 2 — cylinder sampling** (not in the yaml; baked into the dataset). Training draws
**random 8 m-radius cylinders**
([segmentation/treeins_set1.py:674-691](ForAINet/PointCloudSegmentation/torch_points3d/datasets/segmentation/treeins_set1.py#L674-L691)),
rejecting any cylinder with no tree points. Val/test use a **fixed grid** of cylinders,
which is why their sizes (129, 319) are stable while train is a fixed 3000 draws/epoch.

**Stage 3 — per-sample `train_transform`** (the chain printed in the log):

| # | Transform | What it does to the tensors |
|---|---|---|
| 1 | `RandomNoise(σ=0.01)` | jitters `pos` |
| 2 | `RandomRotate(180°, axis=2)` | random yaw about Z |
| 3 | `RandomScaleAnisotropic([0.9,1.1])` | **per-axis independent** scale |
| 4 | `RandomSymmetry(x)` | 50 % chance of mirroring X |
| 5 | `XYZRelaFeature` | adds `pos_x_rela/y/z_rela` = position **minus the cloud mean** |
| 6 | `XYZFeature(add_z)` | adds `pos_z` = **absolute** Z (height above sea level) |
| 7 | `AddFeatsByKeys` | **stacks those four scalars into `data.x` `[N,4]`**, then deletes them |
| 8 | `Center()` | recentres `pos` (features were already frozen in step 7) |
| 9 | `GridSampling3D(0.2, quantize_coords=True)` | voxelises again; creates **`data.coords`** (int32 voxel indices) |
| 10 | `ShiftVoxels` | adds a random integer offset to `coords` — breaks parity bias in sparse convs |

`val_transform` / `test_transform` = steps **5-9 only** (no augmentation, no ShiftVoxels).

Steps 1-4 are *augmentation* (train only); steps 5-7 are *feature engineering*; steps
8-10 are *voxelisation* for the sparse backbone.

## A4. Gdzie jest model — the model

[`torch_points3d/models/panoptic/PointGroup3heads.py:22`](ForAINet/PointCloudSegmentation/torch_points3d/models/panoptic/PointGroup3heads.py#L22) —
`class PointGroup3heads(BaseModel)`.

**How the name resolves to that file** — this chain is worth internalising:

1. CLI `model_name=PointGroup-PAPER` + `models=panoptic/FORpartseg_3heads`
2. Hydra loads `conf/models/panoptic/FORpartseg_3heads.yaml` under the `models` package
3. [`model_factory.py:21`](ForAINet/PointCloudSegmentation/torch_points3d/models/model_factory.py#L21) picks the `PointGroup-PAPER:` **block** out of that yaml
4. [`model_factory.py:27-32`](ForAINet/PointCloudSegmentation/torch_points3d/models/model_factory.py#L27-L32) reads `class: PointGroup3heads.PointGroup3heads` and turns it into the import path `torch_points3d.models.panoptic.PointGroup3heads`

So **`class:` in the yaml is a string that becomes a Python import**. Sibling files
(`PointGroup3heads_new.py`, `pointgroup.py`, …) exist but are unused — only the `class:`
key decides.

Key methods:

| Method | Lines | Role |
|---|---|---|
| `__init__` | 29-105 | builds Backbone + all heads |
| `set_input` | 114-118 | unpacks the batch, moves labels to GPU |
| `forward` | 120-189 | backbone → 3 heads → clustering → scoring |
| `_cluster7` | 422-471 | region-grow on offsets **+** mean-shift on embeddings |
| `_compute_loss` | 632-743 | sums the seven loss terms |
| `backward` | 745-748 | `loss.backward()` |

## A5. Gdzie jest backbone — the backbone

Built at
[`PointGroup3heads.py:32-37`](ForAINet/PointCloudSegmentation/torch_points3d/models/panoptic/PointGroup3heads.py#L32-L37)
by calling `Minkowski(...)`, imported at
[line 11](ForAINet/PointCloudSegmentation/torch_points3d/models/panoptic/PointGroup3heads.py#L11):

```python
from torch_points3d.applications.minkowski import Minkowski
```

Its **shape** comes from the yaml `backbone:` block
([FORpartseg_3heads.yaml:89-127](ForAINet/PointCloudSegmentation/conf/models/panoptic/FORpartseg_3heads.yaml#L89-L127)):
a 7-level U-Net, channels `4 → 16 → 32 → 48 → 64 → 80 → 96 → 112` going down and
symmetrically back up to **16**, with skip connections (that is why the up-path input
widths are `2*k*in_feat` — the skip is concatenated).

> 🔑 **Critical gotcha for anyone planning to change the backbone:** the
> `class: minkowski.Minkowski_Model` key inside that yaml block
> ([line 92](ForAINet/PointCloudSegmentation/conf/models/panoptic/FORpartseg_3heads.yaml#L92)) is
> **inert — editing it does nothing.** The factory passes `model_type=None`, so module
> lookup falls back to `BaseFactory` and resolves `module_name: ResNetDown` / `ResNetUp`
> against whichever module was imported. The backbone is chosen by **the Python import on
> line 11**, not by the config. See [`backbone_torchsparse_1.4.md`](backbone_torchsparse_1.4.md).

Three separate sparse networks are built (Backbone, ScorerUnet, ScorerEncoder) — which is
why the Minkowski deprecation warning appears **three times** in the log.

## A6. Gdzie jest head — the heads

All heads consume the backbone's 16-dim per-point feature.

| Head | Lines | Shape | Purpose |
|---|---|---|---|
| `Semantic` | 77-81 | `[N,16] → [N,5]` | one logit per semantic class |
| `LogSoftLay` | 82-85 | `[N,5] → [N,5]` | log-softmax (so the loss is `nll_loss`, not `cross_entropy`) |
| `Offset` | 70-71 | `[N,16] → [N,3]` | xyz vector pointing to the tree centre |
| `Embed` | 74-75 | `[N,16] → [N,5]` | embedding for mean-shift clustering (`embed_dim: 5`) |
| `ScorerUnet` | 47 | 16 → 16 | re-encodes each cluster proposal |
| `ScorerHead` | 52 | `[K,16] → [K,1]` + Sigmoid | quality score per cluster |
| `ScorerEncoder`, `ScorerMLP` | 48-51 | — | **built but unused** at `scorer_type: "unet"` |

The `5` in `Semantic` is `dataset.num_classes`; the `5` in `Embed` is `embed_dim` from the
yaml. **They are the same number by coincidence**, not by design.

`ScorerEncoder` and `ScorerMLP` are constructed unconditionally but never called in this
configuration — they still appear in the log's module tree and still count toward
`Model size = 11872126`.

## A7. Gdzie jest trening — the training loop

[`torch_points3d/trainer.py`](ForAINet/PointCloudSegmentation/torch_points3d/trainer.py):

| Stage | Location |
|---|---|
| Setup | `_initialize_trainer` — lines 50-148 |
| Dataset built | line 97 |
| Model built | line 98 |
| Optimizer/schedulers | line 99 (`instantiate_optimizers`) |
| DataLoaders | lines 115-121 |
| Tracker (metrics) | line 138 |
| Epoch loop | `train()` — lines 150-173 |
| Train epoch | `_train_epoch` — lines 197-235 |
| `set_input` call | line 209 |
| Optimisation step | line 211 → `model.optimize_parameters2(...)` |
| Eval epoch | `_test_epoch` — lines 237-280 |
| Metrics published | line 190 → `tracker.publish(epoch)` → `wandb.log` |
| Best-model save | line 191 |

The actual `optimizer.step()` lives **inside the model**, not the trainer:
[`base_model.py:274`](ForAINet/PointCloudSegmentation/torch_points3d/models/base_model.py#L274)
(via a `GradScaler`, i.e. mixed precision is on).

> Note: during training the tracker only runs **every 50 iterations**
> ([trainer.py:212-214](ForAINet/PointCloudSegmentation/torch_points3d/trainer.py#L212-L214)),
> whereas evaluation tracks every batch. So `train_*` numbers are sampled, `val_*`/`test_*`
> are exhaustive.

Metrics come from `PanopticTracker` in
`torch_points3d/metrics/panoptic_tracker_pointgroup_treeins_partseg.py`; the loss names
themselves are declared at
[`PointGroup3heads.py:86`](ForAINet/PointCloudSegmentation/torch_points3d/models/panoptic/PointGroup3heads.py#L86).

## A8. Gdzie są wejścia/wyjścia — the tensor contract

This is the most useful table in this document. After collation, with `B=4` samples and
`N` = total points in the batch:

| Attribute | Shape | dtype | Where it comes from |
|---|---|---|---|
| **`x`** | **`[N, 4]`** | float32 | `AddFeatsByKeys`; columns `[x−x̄, y−ȳ, z−z̄, z_absolute]`. **This is the `in=4` on the first conv.** |
| `pos` | `[N, 3]` | float32 | point coordinates, centred per cylinder |
| `coords` | `[N, 3]` | **int32** | `round(pos / 0.2)` — the voxel indices the sparse conv actually indexes |
| `batch` | `[N]` | int64 | which sample each point belongs to |
| `y` | `[N]` | int64 | semantic label; `-1` = ignore, `0..4` = classes |
| `instance_labels` | `[N]` | int64 | `0` = not an instance, `1..K` per sample (**not** globally unique — always paired with `batch`) |
| `instance_mask` | `[N]` | bool | `instance_labels != 0` |
| `vote_label` | `[N, 3]` | float32 | regression target for the `Offset` head |
| `center_label` | `[B*200, 3]` | float32 | zero-padded instance centres (max 200 instances/sample) |
| `num_instances` | `[B]` | int64 | per sample |
| `origin_id` | `[N]` | int64 | index back into the full-resolution cloud |

**Not present**: `rgb`, `intensity`, `normal` (intensity would require
`add_input_features` in the data config — see `treeins_set1_add_intensity.yaml`).

**Classes** ([segmentation/treeins_set1.py:35-42](ForAINet/PointCloudSegmentation/torch_points3d/datasets/segmentation/treeins_set1.py#L35-L42)):

| id | name | stuff/thing |
|---|---|---|
| 0 | `low_vegetation` | stuff |
| 1 | `ground` | stuff |
| 2 | `stem_points` | **thing** |
| 3 | `live_branches` | **thing** |
| 4 | `branches` | **thing** |

"Stuff" classes never get instance ids; only "things" (2,3,4) are clustered into trees.
This is why the log's `iou_per_class` has exactly five entries `{0..4}`.

**Outputs** of `forward`: `semantic_logits [N,5]`, `offset_logits [N,3]`,
`embed_logits [N,5]`, plus `clusters` (a list of index tensors) and
`cluster_scores [K]` once clustering is active.

## A9. How it all wires together

```
train.py  (@hydra.main)
   │  composes conf/config.yaml + your CLI overrides  ->  cfg
   ▼
Trainer(cfg)                                     trainer.py:45
   ├─ instantiate_dataset(cfg.data)              trainer.py:97
   │     └─ TreeinsFusedDataset                  panoptic/treeins_set1.py:563
   │           ├─ reads raw *.ply                segmentation/treeins_set1.py:64
   │           ├─ pre_collate_transform (cached) grid_transform.py
   │           └─ cylinder sampling + transforms features.py / transforms.py
   ├─ instantiate_model(cfg, dataset)            trainer.py:98
   │     └─ PointGroup3heads                     panoptic/PointGroup3heads.py:22
   │           ├─ Backbone      = Minkowski(unet)      ← yaml backbone: block
   │           ├─ ScorerUnet    = Minkowski(unet)      ← yaml scorer_unet:
   │           ├─ ScorerEncoder = Minkowski(encoder)   ← yaml scorer_encoder:
   │           └─ Semantic / Offset / Embed / ScorerHead
   ├─ create_dataloaders(...)                    base_dataset.py:194
   └─ get_tracker(...)                           -> PanopticTracker
   ▼
trainer.train()                                  trainer.py:150
   for epoch in 1..N:
      _train_epoch                               trainer.py:197
         for batch in train_loader:
            model.set_input(batch, device)       PointGroup3heads.py:114
            model.optimize_parameters2(...)      base_model.py:259
               forward()                         PointGroup3heads.py:120
                  x[N,4] ─► Backbone ─► feats[N,16]
                                          ├─► Semantic ─► [N,5]
                                          ├─► Offset   ─► [N,3]
                                          └─► Embed    ─► [N,5]
                                                 │
                            (only if epoch > prepare_epoch)
                                                 ▼
                                    _cluster7 ─► ScorerUnet ─► ScorerHead ─► scores[K]
               backward()  = sum of 7 weighted losses
               optimizer.step()                  base_model.py:274
      _test_epoch (val, then test)               trainer.py:237
      tracker.publish(epoch) ─► wandb.log        base_tracker.py:100
      checkpoint.save_best_models(...)           trainer.py:191
```

---

# Part B — the log, line by line

## B1. Startup

```
wandb: WARNING Saving files without folders. ...
invalid syntax (<string>, line 1)
```

The wandb warning is cosmetic (it flattens the saved config files).

`invalid syntax (<string>, line 1)` is printed to **stdout**, not through the logger, and
is **non-fatal** — training proceeds normally. I could not trace it to a specific line, so
I will not invent a cause; treat it as noise. If you want to chase it, run under the
debugger with a breakpoint on `SyntaxError`.

```
[torch_points3d.applications.minkowski][WARNING] - Minkowski API is deprecated in favor of
the SparseConv3d API. It should be a simple drop in replacement (no change to the API).
[torch_points3d.applications.modelfactory][INFO] - The config will be used to build the model
   ... repeated 3× ...
```

Raised unconditionally at
[`applications/minkowski.py:48-50`](ForAINet/PointCloudSegmentation/torch_points3d/applications/minkowski.py#L48-L50)
on **every** `Minkowski()` call. Three calls = Backbone, ScorerUnet, ScorerEncoder.
This is the single most important line in your log if you care about the backbone —
it is the framework telling you this API is legacy. Details in
[`backbone_torchsparse_1.4.md`](backbone_torchsparse_1.4.md).

"The config will be used to build the model" means your yaml `backbone:` block was found,
so the built-in `conf/sparseconv3d/unet_4.yaml` fallback is **not** used (and `num_layers=4`
in the Python call is therefore dead).

```
[torch_points3d.models.base_model][WARNING] - The path does not exist, it will not load any model
```

`path_pretrained` at
[FORpartseg_3heads.yaml:177](ForAINet/PointCloudSegmentation/conf/models/panoptic/FORpartseg_3heads.yaml#L177)
points at `/cluster/work/igp_psr/...`, a path from the original authors' HPC cluster.
Harmless — it just means **you are training from scratch**, which is what you want.

## B2. The module tree

The big `PointGroup3heads(...)` block is just `print(model)`. Read it against the yaml:

```
(0): MinkowskiConvolution(in=4, out=16, ...)     ← [FEAT, in_feat] with FEAT=4, in_feat=16
(1): MinkowskiConvolution(in=16, out=16, stride=[2,2,2])
(2): ... in=32 ... (3): in=48 ... (4): in=64 ... (5): in=80 ... (6): in=96 → 112
```

That is exactly `down_conv_nn` from the yaml. Each `ResNetDown` has `N: 2` `ResBlock`s.
`stride=[2,2,2]` = the level halves spatial resolution; the first level has stride 1.

The up-path mirrors it, and you can see the skip connections in the widths:
`ResNetUp(1)` takes `in=192` = `2 × 96` — the 96 from the previous up-level concatenated
with the 96 from the matching down-level.

`Model size = 11872126` ≈ **11.9 M parameters**, and remember ~a third of that is
`ScorerEncoder` + `ScorerMLP`, which this config never calls.

> This log predates the 4-class patch. A current run prints **`11872109`** — 17 fewer,
> which is exactly one output of the `Linear(16→N)` semantic head (16 weights + 1 bias).

## B3. Dataset summary

```
Size of train_dataset = 3000
Size of test_dataset  = 319
Size of val_dataset   = 129
Batch size = 4
```

**`3000` is not a corpus size** — it is `sample_per_epoch`, hard-coded at
[panoptic/treeins_set1.py:590](ForAINet/PointCloudSegmentation/torch_points3d/datasets/panoptic/treeins_set1.py#L590).
Each epoch draws 3000 *random* 8 m cylinders from the 29 training plots. `319`/`129` are
real counts of fixed grid cylinders.

That gives `3000/4 = 750` training iterations, `129/4 = 33` val, `319/4 = 80` test —
exactly the progress-bar denominators in the log.

## B4. An epoch

```
EPOCH 1 / 10
100%|█| 750/750 [17:16<00:00, 1.38s/it, train_acc=75.12, train_ins_dist_loss=0.835, ...]
Learning rate = 0.000989
100%|█| 33/33 [00:25<00:00, 1.31it/s, val_acc=85.38, ...]
```

`0.000989 = 0.001 × 0.9885` — the `ExponentialLR` decay from
[`conf/lr_scheduler/exponential.yaml`](ForAINet/PointCloudSegmentation/conf/lr_scheduler/exponential.yaml)
(gamma chosen to divide LR by 10 every 200 epochs).

Then the tracker prints the full metric block. Reading the epoch-1 val numbers:

| Metric | Value | Meaning |
|---|---|---|
| `val_loss` | 0.993 | weighted sum of all terms |
| `val_semantic_loss` | 0.396 | NLL on the 5 classes |
| `val_offset_norm_loss` | 2.627 | how far off the centre-vectors are (weight 0.1) |
| `val_offset_dir_loss` | **-0.750** | cosine-similarity term — **negative is correct here**, it is `−cos(θ)`, so −1 is perfect |
| `val_ins_*` | 0.2-0.4 | discriminative-embedding losses (var/dist/reg) |
| `val_acc` | 85.4 | overall point accuracy |
| `val_macc` | 53.1 | mean **per-class** accuracy — much lower, so rare classes are weak |
| `val_miou` | 45.0 | mean IoU, the number to actually report |
| `val_iou_per_class` | `{0:61.7, 1:24.8, 2:46.4, 3:86.4, 4:5.6}` | per class |

The per-class IoU is the interesting one: class 3 (`live_branches`) is at 86 % while
class 4 (`branches`) is at 5.6 % and class 1 (`ground`) at 24.8 %. That spread — not the
average — is what you tune against.

## B5. Progress across epochs

```
loss: 0.993 -> 0.865, semantic_loss: 0.396 -> 0.332,
acc: 85.4 -> 86.8, macc: 53.1 -> 63.7, miou: 45.0 -> 53.2
```

This one-line diff (printed by `torch_points3d.utils.colors`) is the best summary in the
log: **every semantic metric improved**, and `miou` jumped 8 points. Class 4 went from
5.6 → 35.6 IoU. The model is learning correctly.

---

# Part C — what the numbers mean, and three anomalies

## C1. 🔴 Every instance metric is exactly 0.0 — and that is expected

```
val_Iacc = 0.0    val_cov = 0.0    val_wcov = 0.0
val_mIPre = 0.0   val_mIRec = 0.0  val_F1 = 0.0
val_pos = 0.0     val_neg = 0.0
```

**Cause:** clustering is gated behind an epoch threshold.
[`PointGroup3heads.py:139`](ForAINet/PointCloudSegmentation/torch_points3d/models/panoptic/PointGroup3heads.py#L139)
only runs `_cluster7` when `epoch > self.opt.prepare_epoch`, and your config sets

```yaml
prepare_epoch: 30      # FORpartseg_3heads.yaml:168
```

Your run did **10 epochs**. So no clusters were ever produced, no instances were ever
scored, and every instance metric is trivially zero. The same gate skips the IoU and
score losses in `_compute_loss` (lines 702-732) — which is also why `score_loss` and
`mask_loss` never appear in the printed metrics.

**This is by design.** The offset and embedding heads need to produce something
meaningful before clustering them is worthwhile; clustering random embeddings would
just waste compute and inject noise into the score head.

**What to do:**
- To actually see instance results: **train for more than 30 epochs** (the config's own
  `epochs: 150` is the intended setting).
- To smoke-test the clustering code quickly: temporarily set
  `models.PointGroup-PAPER.prepare_epoch=2` on the command line. Expect poor scores —
  the point is to exercise the code path, not to get good numbers.

## C2. 🟡 `val_*` and `test_*` losses are bit-identical

```
val_loss  = 0.992956280708313
test_loss = 0.992956280708313      ← identical to 16 decimal places
```

…while `val_acc` (85.38) and `test_acc` (85.21) *do* differ. Identical accuracy would be a
coincidence; identical loss to the last bit is not.

The accuracy metrics are recomputed per stage from a confusion matrix, but the **loss**
values are read from the model's stored `self.loss*` attributes via
`get_current_losses()`, which are not reset between the val and test passes. So the test
block re-reports the last loss computed during validation.

**Practical consequence: trust `test_acc`/`test_miou`/`test_iou_per_class`, but do not
read anything into `test_loss`** — it is not a test-set loss. This is a framework quirk,
not something your config caused.

## C3. 🟡 Epoch 2 was 5.5× slower than epoch 1

```
EPOCH 1: 750/750 [17:16<00:00, 1.38s/it]
EPOCH 2: 750/750 [1:35:49<00:00, 7.67s/it]
```

Same work, same data, wildly different wall time — while the *reported* `iteration` time
actually went **down** (0.503 → 0.331 s) and `data_loading` went from 1.359 → 0.472 s.
So the time is being spent somewhere the instrumentation is not measuring.

I have not diagnosed this, so treat the following as hypotheses to test rather than
conclusions:
- host contention (another process, or thermal throttling on the GPU),
- disk/IO pressure on the mounted volume — the container mounts `./ForAINet` from
  Windows, and bind-mount IO on Docker Desktop is slow,
- memory pressure causing swapping.

Worth watching `nvidia-smi` and `docker stats` during a run before changing anything.
(This is a different issue from the earlier `Bus error` crash, which was `/dev/shm`
exhaustion and is already fixed via `shm_size` in
[`docker-compose.yml`](docker-compose.yml).)

## C4. Which numbers to actually report

For a semantic-segmentation result: **`miou`** first, then `iou_per_class` to show where
the model is weak, then `macc`. Plain `acc` is misleading here because the dataset is
dominated by a few classes.

For instance/panoptic results: `cov`, `wcov`, `F1`, `mIPre`, `mIRec` — but only from a run
that got past `prepare_epoch`.

---

## Where to go next

- Changing the backbone → [`backbone_torchsparse_1.4.md`](backbone_torchsparse_1.4.md)
  and [`backbone_torchsparse_v2.md`](backbone_torchsparse_v2.md)
- Running / debugging the container → [`run_debugger.md`](run_debugger.md)
