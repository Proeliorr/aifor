# ForAINet — The Point-Cloud Segmentation Pipeline

This document explains the **point-cloud classification / segmentation** part of
ForAINet only — everything under the `PointCloudSegmentation/` directory. The other
two ForAINet stages (`superpoint_graph/` feature extraction and `tree_metrics/`
inventory measurement) are intentionally out of scope here.

All paths below are relative to `ForAINet/PointCloudSegmentation/` unless stated
otherwise.

---

## 1. What this part of ForAINet does

`PointCloudSegmentation/` is a **fork of [torch-points3d](https://github.com/torch-points3d/torch-points3d)**,
a PyTorch framework for deep learning on 3D point clouds. ForAINet specializes it for
**forest LiDAR**: the input is a dense 3D point cloud of a forest plot, and the network
labels every single point.

It performs **panoptic segmentation**, which is the combination of two tasks done at
once:

- **Semantic segmentation** — *"what kind of thing is this point?"* Each point is
  assigned a class (ground, low vegetation, stem, branches, …). Two points on two
  different trees both get the class `stem_points`; semantic segmentation does not tell
  them apart.
- **Instance segmentation** — *"which individual object does this point belong to?"*
  Points are grouped into discrete objects. Here an *instance* is a **single tree**, so
  instance segmentation assigns every tree point a **tree ID**.

Putting them together (panoptic) yields, for each point: a class label **and**, if it
belongs to a tree, the ID of the specific tree it is part of. That is exactly the
information the downstream `tree_metrics/` stage needs to measure each tree.

---

## 2. Key vocabulary

Because the rest of this document leans on a few terms, they are defined here in plain
language.

- **Backbone.** The *shared* feature-extraction network. It takes the raw per-point
  inputs (coordinates and a few features) and transforms each point into a learned
  feature vector — a compact numeric description that captures local 3D geometry and
  context. The backbone does not itself make the final prediction; it produces the
  representation that every downstream task reads from. In ForAINet the backbone
  outputs a 16-dimensional feature vector per point (`[N, 16]`).

- **Head.** A small, task-specific sub-network attached *on top of* the backbone
  features. Each head is responsible for producing **one** kind of prediction. The
  design is "one backbone, several heads": the expensive shared computation happens once
  in the backbone, and lightweight heads branch off it. ForAINet's model is literally
  named `PointGroup3heads` because it hangs multiple prediction heads off one backbone.

- **Semantic vs. instance, restated.** Semantic = a *class* per point (a fixed, small
  set of categories). Instance = an *object ID* per point (an open-ended number of trees
  that is not known in advance and must be discovered by grouping points).

---

## 3. The full pipeline, end to end

The pipeline runs left-to-right. Each stage below names the file where it lives.

**Stage 1 — Raw input: `.ply` point-cloud files.**
The network consumes `.ply` files. Each file is a forest tile where every point carries
coordinates, a semantic label, and a tree ID (see §5). *(These `.ply` files are produced
upstream by the `superpoint_graph/` stage, which converts raw `.las` LiDAR — but that
conversion is outside the scope of this document.)*

**Stage 2 — Reading the file into tensors.**
`torch_points3d/datasets/segmentation/treeins_set1.py::read_treeins_format()` opens each
`.ply` (via `torch_points3d/modules/KPConv/plyutils.read_ply`) and pulls out the columns
into arrays: the `x,y,z` coordinates become the point positions, `semantic_seg` becomes
the semantic label, and `treeID` becomes the instance label.

**Stage 3 — The dataset object.**
`torch_points3d/datasets/panoptic/treeins_set1.py::TreeinsFusedDataset` wraps the raw
points into the dataset the trainer uses. It relies on the samplers `TreeinsCylinder` /
`TreeinsSphere` (in the segmentation module) to cut the large plot into manageable
training samples — here **vertical cylinders of radius 8 m** (`sampling_format: cylinder`,
`radius: 8`). Which dataset class is used is selected in the config
`conf/data/panoptic/treeins_set1.yaml` (`class: treeins_set1.TreeinsFusedDataset`).

**Stage 4 — Preprocessing / transforms.**
Also declared in `conf/data/panoptic/treeins_set1.yaml` and implemented under
`torch_points3d/core/data_transform/`. Two groups:
- *Voxelization / subsampling*: `GridSampling3D` collapses points onto a regular 0.2 m
  grid so the cloud has a manageable, roughly uniform density (`first_subsampling: 0.2`).
- *Training-time augmentation* (train only): `RandomNoise`, `RandomRotate` about the
  vertical axis, `RandomScaleAnisotropic`, `RandomSymmetry` — random perturbations that
  make the model robust. Then **input features are built**: `XYZRelaFeature` +
  `XYZFeature` + `AddFeatsByKeys` assemble the per-point feature vector
  `[pos_x_rela, pos_y_rela, pos_z_rela, pos_z]` (relative x/y/z within the sample plus
  absolute height), followed by `Center` and a quantizing `GridSampling3D`. Validation
  and test use the same feature build **without** the random augmentation.

**Stage 5 — Backbone (shared).**
`torch_points3d/applications/minkowski.py` provides a **Minkowski-Engine sparse-convolution
ResNet U-Net**. "Sparse convolution" means it only computes on occupied voxels (point
clouds are mostly empty space), which is what makes 3D convolution affordable. "U-Net"
means it downsamples to capture large-scale context, then upsamples back to full point
resolution with skip connections. The backbone is instantiated once inside
`PointGroup3heads.__init__` as `self.Backbone` and, in the forward pass, turns the input
into `backbone_features` of shape `[N, 16]`.

**Stage 6 — Heads (task-specific branches).**
Defined in `torch_points3d/models/panoptic/PointGroup3heads.py` and all reading the same
`backbone_features`:
- `self.Semantic` → semantic class scores per point *(semantic branch)*.
- `self.Offset` → a 3D vector per point *(instance branch)*.
- `self.Embed` → an embedding vector per point *(instance branch)*.
See §4 for exactly which head serves which task.

**Stage 7 — Clustering into tree instances (instance branch only).**
Heads alone do not produce tree IDs — the per-point offset and embedding outputs must be
**grouped** into instances. `PointGroup3heads` does this between the forward pass and the
loss, using one of several strategies `_cluster` … `_cluster7` (region growing on
offset-shifted positions and/or mean-shift on embeddings). The 5-class forest setup uses
`_cluster7`, which combines the semantic, offset, and embedding outputs. Each resulting
cluster is one candidate tree.

**Stage 8 — Scoring the proposed instances (instance branch only).**
Each candidate cluster is passed through the scorer networks (`self.ScorerUnet`,
`self.ScorerEncoder`, `self.ScorerMLP`, `self.ScorerHead` + Sigmoid, and optional
`self.MaskScore`) to predict a confidence score — how likely the cluster is a real, clean
tree instance. Low-quality proposals can then be filtered.

**Stage 9 — Training loop and losses.**
Entry point `train.py` (`@hydra.main(...) → Trainer(cfg).train()`); the loop, optimizer,
checkpointing and logging live in `torch_points3d/trainer.py`. The combined loss is
computed in `PointGroup3heads._compute_loss`, drawing on
`torch_points3d/core/losses/panoptic_losses.py` (see the per-branch losses in §4).

**Stage 10 — Outputs.**
The model emits, per point, a predicted semantic class and (for tree points) an instance
ID. These are written back to `.ply` for inspection and evaluation via `to_ply`,
`to_ins_ply`, and `to_eval_ply` (evaluation files carry columns `x, y, z, preds, gt`).
Large plots are processed by tiling: `split_largePC_to_tiles.py` → predict →
`merge_tiles.py` (orchestrated by `large_PC_predict.sh`); metrics come from
`eval.py` / `evaluation_stats_FOR.py`.

---

## 4. Which parts belong to SEMANTIC vs. INSTANCE segmentation

This is the key structural point: **the backbone and all preprocessing are shared**, and
the pipeline only splits into two branches at the heads.

| Pipeline element | Belongs to | Where |
|---|---|---|
| `.ply` reader, cylinder sampling, transforms | **Shared** | `datasets/segmentation/treeins_set1.py`, `conf/data/panoptic/treeins_set1.yaml` |
| Minkowski U-Net backbone → `[N,16]` features | **Shared** | `applications/minkowski.py`, `self.Backbone` |
| `self.Semantic` head → class logits (+ LogSoftmax) | **Semantic** | `PointGroup3heads.py` |
| `self.BiSemantic` (optional stuff/thing binary) head | **Semantic** | `PointGroup3heads.py` |
| Ground-truth `data.y` (class labels) | **Semantic** | from `semantic_seg` column |
| Semantic loss = **NLL** (optional class/height weighting) | **Semantic** | `_compute_loss` |
| `self.Offset` head → per-point 3D offset `[N,3]` | **Instance** | `PointGroup3heads.py` |
| `self.Embed` head → per-point embedding `[N,5]` | **Instance** | `PointGroup3heads.py` |
| Clustering (`_cluster`…`_cluster7`) into trees | **Instance** | `PointGroup3heads.py` |
| Scorer nets `Scorer*` + `MaskScore` | **Instance** | `PointGroup3heads.py` |
| Ground-truth `instance_labels` / `vote_label` | **Instance** | from `treeID` column, built by `datasets/panoptic/utils.set_extra_labels` |
| Offset loss (L1 + cosine), discriminative embedding loss, instance-IoU score loss, mask loss | **Instance** | `core/losses/panoptic_losses.py` |

**In prose:**

- **Semantic segmentation** is the simpler branch: one head (`self.Semantic`) maps the
  shared 16-D features to a score for each class, and an NLL loss compares those scores to
  the ground-truth class labels (`data.y`, derived from the `semantic_seg` column). The
  optional `BiSemantic` head adds a coarse "stuff vs. thing" (non-tree vs. tree)
  distinction.

- **Instance segmentation** is a multi-step branch because the number of trees is unknown
  and must be *discovered*:
  1. The **offset head** predicts, for each tree point, a vector pointing toward the
     center of the tree it belongs to — so points of one tree get pulled together.
  2. The **embedding head** learns a per-point vector such that points of the same tree
     have similar embeddings and different trees are pushed apart (that separation is
     enforced by the *discriminative loss*).
  3. **Clustering** then groups points into candidate trees using the shifted positions
     and/or embeddings.
  4. The **scorer networks** rate each candidate, supervised by an *instance-IoU loss*
     (and a per-point *mask loss*), which measure how well a proposed cluster overlaps a
     real ground-truth tree.
  The instance ground truth comes from the `treeID` column, expanded into
  `instance_labels` and `vote_label` by `datasets/panoptic/utils.set_extra_labels`.

Only the tree-like semantic classes are eligible to become instances: the code marks
`SemIDforInstance = [2, 3, 4]` (stem / live branches / branches). Ground and low
vegetation are "stuff" — they receive a semantic class but no instance ID.

---

## 5. Inputs and outputs (columns, tensors, classes)

### Expected `.ply` input columns
Read by `read_treeins_format()` (`datasets/segmentation/treeins_set1.py`):

- **`x`, `y`, `z`** — point coordinates → become `data.pos`, shape `[N, 3]`, `float32`.
- **`semantic_seg`** — the semantic class, remapped by `-1` in code so that unlabelled
  points become `-1` (ignored during training).
- **`treeID`** — the instance label; a tree identifier per point (shifted by `+1`, with
  `0` reserved for "no instance").
- *Optional* handcrafted geometric/return features, used only if `add_input_features` is
  set: `Sum, Omnivariance, Eigenentropy, Anisotropy, planarity, linearity, Surface_var,
  scattering, verticality, Verticality2, moment1_1, moment1_2, moment2_1, moment2_2,
  intensity, return_num, scan_angle_rank`. (By default these are *not* used — the model
  runs on geometry alone.)

### Semantic classes
`INV_OBJECT_LABEL` in `datasets/segmentation/treeins_set1.py` (`Treeins_NUM_CLASSES = 5`;
dataroot `data_set1_5classes`):

| ID | Class | Instance? |
|---|---|---|
| 0 | low_vegetation | no (stuff) |
| 1 | ground | no (stuff) |
| 2 | stem_points | yes (thing) |
| 3 | live_branches | yes (thing) |
| 4 | branches | yes (thing) |

### Tensors the model consumes
Assembled as a torch-geometric `Data` / Minkowski batch:

- `data.pos` `[N, 3]` — coordinates.
- `data.x` — the input feature vector per point; by default **4 channels**
  `[pos_x_rela, pos_y_rela, pos_z_rela, pos_z]` from the transform chain. Plus quantized
  `data.coords` and `data.batch` for the sparse backbone.
- `data.y` `[N]` — semantic ground truth *(semantic branch supervision)*.
- `data.instance_labels` `[N]` and `vote_label` — instance ground truth *(instance branch
  supervision)*.

### What the model outputs
From `PointGroup3heads.forward` (containers in
`torch_points3d/models/panoptic/structure_3heads.py`):

- `semantic_logits` `[N, num_classes]` — the semantic prediction.
- `offset_logits` `[N, 3]` and `embed_logits` `[N, embed_dim=5]` — the instance
  intermediates.
- `all_clusters` + `cluster_scores` — the discovered tree instances and their confidence.

Combined and written out, this is a **per-point semantic class + per-tree-point instance
ID**.

---

## 6. How to run it

Training is Hydra-configured; you compose a run by selecting a model, a data config, and a
training config:

```bash
# from PointCloudSegmentation/
python train.py model_name=<model> data=panoptic/treeins_set1 training=treeins_set1
```

- **Entry:** `train.py` → `torch_points3d/trainer.py`.
- **Model config:** `conf/models/panoptic/FORpartseg_3heads.yaml` (defines the backbone
  and the three heads); variants `_BiLoss`, `_mlpScore`, `_encoderScore`, `_heightweight`.
- **Data config:** `conf/data/panoptic/treeins_set1.yaml`.
- **Training config:** `conf/training/treeins_set1.yaml`.
- **Evaluation:** `eval.py`, `evaluation_stats_FOR.py`; large clouds via
  `split_largePC_to_tiles.py` → predict → `merge_tiles.py`.

---

## 7. Quick technical reference — where is each piece

This section is a plain lookup table: for each part of the system, the file and the
function/class/key that implements it. All paths are relative to
`PointCloudSegmentation/`.

| Question | File | Function / class / key |
|---|---|---|
| **Config files** | `conf/` (Hydra YAML) | root `conf/config.yaml`; model `conf/models/panoptic/FORpartseg_3heads.yaml`; data `conf/data/panoptic/treeins_set1.yaml`; training `conf/training/treeins_set1.yaml` |
| **Dataset / loader** | `datasets/panoptic/treeins_set1.py` | class `TreeinsFusedDataset` |
| ↳ raw file reader | `datasets/segmentation/treeins_set1.py` | `read_treeins_format()` |
| ↳ how a plot is split into samples | `datasets/segmentation/treeins_set1.py` | samplers `TreeinsCylinder`, `TreeinsSphere` |
| ↳ dataset is instantiated by | `datasets/dataset_factory.py` | `instantiate_dataset()` (base plumbing in `base_dataset.py`) |
| **Preprocessing** | `conf/data/panoptic/treeins_set1.yaml` (declared) + `core/data_transform/transforms.py` (implemented) | `XYZRelaFeature`, `XYZFeature`, `AddFeatsByKeys`, `Center`, `RandomNoise/Rotate/ScaleAnisotropic/Symmetry` |
| **Voxelization of the input clouds** | `core/data_transform/grid_transform.py` | class `GridSampling3D` (line 151), method `_process` (line 181); helper `group_data` (line 33) |
| **Model** | `models/panoptic/PointGroup3heads.py` | class `PointGroup3heads`; outputs container `models/panoptic/structure_3heads.py`; built by `models/model_factory.py::instantiate_model` |
| **Backbone** | `applications/minkowski.py` | Minkowski sparse-conv ResNet U-Net; held as `self.Backbone` in `PointGroup3heads.__init__` |
| **Head(s)** | `models/panoptic/PointGroup3heads.py` (in `__init__`) | `self.Semantic` (semantic); `self.Offset`, `self.Embed` (instance); `self.ScorerUnet/ScorerEncoder/ScorerMLP/ScorerHead`, `self.MaskScore`; optional `self.BiSemantic` |
| **Training** | `train.py` → `torch_points3d/trainer.py` | `Trainer.train()`; loss in `PointGroup3heads._compute_loss` + `core/losses/panoptic_losses.py` |
| **TreeMix data augmentation** | `core/data_transform/transforms.py` | `Tree3DMix2` (line 1343, used by the configs), `Tree3DMix` (line 1233); generic `Mix3D` (line 1162) |
| **Inputs / outputs** | `datasets/segmentation/treeins_set1.py`, `models/panoptic/PointGroup3heads.py` | see notes below |

**Notes on the trickier rows:**

- **Voxelization.** `GridSampling3D` collapses points onto a regular grid (grid size
  `first_subsampling: 0.2` m) so density is uniform and manageable. When called with
  `quantize_coords=True` (see `_process`, around line 193), it also turns each point's
  position into **integer voxel coordinates** — that integer grid is exactly what the
  Minkowski sparse-convolution backbone needs as input.

- **TreeMix.** This is a forest-specific augmentation: it takes whole **tree instances**
  from one cloud and inserts them into another, creating new plausible plots and more
  varied training data. The active class is `Tree3DMix2`; you switch it on by using a
  data config such as `conf/data/panoptic/treeins_set1_treemix3d_pd*.yaml` (which lists
  `- transform: Tree3DMix2` in its `pre_collate_transform`) together with a training
  config like `conf/training/mixtree_*.yaml`.

- **Inputs.** The reader `read_treeins_format()` expects `.ply` files with columns
  `x, y, z` (coordinates), `semantic_seg` (class), and `treeID` (tree/instance id).
  These become the tensors `data.pos [N,3]`, `data.x [N,4]` (built features
  `pos_x_rela, pos_y_rela, pos_z_rela, pos_z`), `data.y [N]` (semantic labels), and
  `data.instance_labels [N]`.

- **Outputs.** `PointGroup3heads.forward` returns `semantic_logits [N, num_classes]`,
  `offset_logits [N, 3]`, `embed_logits [N, 5]`, and the discovered `all_clusters` with
  their `cluster_scores`. Predictions are written to `.ply` by `to_ply` / `to_ins_ply` /
  `to_eval_ply` (evaluation files use columns `x, y, z, preds, gt`).

---

## 8. Wiring, swapping the sparse backend, and data format

This section explains **how the components connect**, then answers three practical
questions: how to replace **MinkowskiEngine** with **TorchSparse++**, whether switching to
`.npy` single-tree files matters, and how the two sparse-convolution libraries correspond
class-for-class.

### 8.1 How the elements are wired together

The important idea is that the pieces talk to each other through **a few well-defined
boundary objects**, so most parts do not know or care what the others are built from.

```
.ply file
  │  read_treeins_format()            datasets/segmentation/treeins_set1.py
  ▼
torch_geometric Data                  fields: pos [N,3], y [N], instance_labels [N]
  │  transforms (incl. GridSampling3D) core/data_transform/  → adds x [N,4], coords [N,3 int], batch [N]
  ▼
Data(pos, x, coords, batch)  ───────► this is the single hand-off into the network
  │
  │  model.set_input(data)            models/panoptic/PointGroup3heads.py
  ▼
Backbone(self.input)                  applications/minkowski.py  (self.Backbone)
  │    _set_input() builds a backend SparseTensor from (x, coords, batch)
  │    U-Net of ResNetDown / ResNetUp blocks   modules/MinkowskiEngine/api_modules.py
  ▼
Data(x = backbone_features [N,16])    ← only .x is read downstream
  │
  ├── self.Semantic(x)  → semantic_logits         (semantic branch)
  ├── self.Offset(x)    → offset_logits           (instance branch)
  ├── self.Embed(x)     → embed_logits            (instance branch)
  │
  ▼  clustering + scorers read self.input.coords / self.input.batch (NOT ME objects)
all_clusters + cluster_scores → written to .ply
```

Two wiring facts matter for everything below:

1. **The model itself never touches MinkowskiEngine classes.** It reads plain
   torch-geometric fields (`self.input.x`, `self.input.coords`, `self.input.batch`). The
   only place a real `SparseTensor` is created is inside the backbone application
   (`applications/minkowski.py::BaseMinkowski._set_input`). That isolation is what makes the
   backend swap easy.

2. **The backbone is reached through a small factory function**, `Minkowski(architecture,
   input_nc, num_layers, config)`, called three times in `PointGroup3heads.__init__`:
   `self.Backbone` (line 32), `self.ScorerUnet` (line 47), `self.ScorerEncoder` (line 48).

### 8.2 Replacing MinkowskiEngine with TorchSparse++

**First, what "TorchSparse++" is.** It is not a different library from TorchSparse — it is
**TorchSparse v2.1**, the rewritten version presented at MICRO 2023 ([paper](https://arxiv.org/abs/2204.10319),
[repo](https://github.com/mit-han-lab/torchsparse)). Its headline contributions are an
**adaptive matrix-multiplication grouping** scheme and **locality-aware memory access** for the
sparse kernels, plus an **autotuner** (`torchsparse.tune()`) that picks the fastest dataflow for
your specific model and GPU. This matters here because the version pinned in ForAINet's
`Dockerfile` (`torchsparse.git@v1.4.0`) is the *old* TorchSparse, and — as detailed below — v2.1
changed a few conventions, so "++" is not a transparent bump.

**Good news: torch-points3d was built for exactly this.** It already has a backend-agnostic
layer, and ForAINet's `applications/minkowski.py` even prints *"Minkowski API is deprecated
in favor of the SparseConv3d API… a simple drop-in replacement."* The two relevant pieces:

- `modules/SparseConv3d/nn/` — a thin abstraction that exposes one unified API
  (`Conv3d, Conv3dTranspose, BatchNorm, ReLU, cat, SparseTensor`) with **two
  interchangeable implementations**: `nn/minkowski.py` and `nn/torchsparse.py`. You pick one
  with `set_backend("minkowski" | "torchsparse")` (`nn/__init__.py`).
- `applications/sparseconv3d.py` — the backend-agnostic backbone builder
  `SparseConv3d(architecture, input_nc, num_layers, config, backend="minkowski"|"torchsparse")`.
  Its constructor signature is **identical** to `Minkowski(...)`.

**What you change, smallest first:**

1. **Point the model at the backend-agnostic backbone** — in
   `models/panoptic/PointGroup3heads.py`:
   ```python
   # was:  from torch_points3d.applications.minkowski import Minkowski
   from torch_points3d.applications.sparseconv3d import SparseConv3d
   ...
   self.Backbone       = SparseConv3d("unet", input_nc=dataset.feature_dimension,
                                      num_layers=4, config=backbone_options.get("config", {}),
                                      backend="torchsparse")
   self.ScorerUnet     = SparseConv3d("unet",   input_nc=self.Backbone.output_nc,
                                      num_layers=4, config=option.scorer_unet,   backend="torchsparse")
   self.ScorerEncoder  = SparseConv3d("encoder",input_nc=self.Backbone.output_nc,
                                      num_layers=4, config=option.scorer_encoder, backend="torchsparse")
   ```
   (Alternatively, once the model uses `SparseConv3d`, leave `backend` at its default and set
   the environment variable `SPARSE_BACKEND=torchsparse` — `sparseconv3d.py` honours it.)

2. **Configs and preprocessing need no changes.** The residual blocks referenced by the
   config (`module_name: ResNetDown` / `ResNetUp`) exist under *both* backends with identical
   names (`modules/SparseConv3d/modules.py` vs `modules/MinkowskiEngine/api_modules.py`). And
   the voxelization step `GridSampling3D(quantize_coords=True)` already produces the integer
   voxel `coords` that both libraries expect — so no transform changes either.

3. **Install TorchSparse++ (v2.1).** The repo's `Dockerfile` installs the *old* TorchSparse
   (`torchsparse.git@v1.4.0`). Replace that with v2.1 — the recommended route is the project's
   prebuilt wheels via its install script, or `pip install git+https://github.com/mit-han-lab/torchsparse.git`
   (needs PyTorch ≥ 1.9 built with CUDA). Update `requirements.txt` to match.

4. **⚠️ Fix the adapter for the v2.1 breaking changes — this is the real work.** The thin
   adapter `modules/SparseConv3d/nn/torchsparse.py` was written against **v1.4** and is *not*
   correct as-is for v2.1/++. Concretely:
   - **Coordinate order flipped.** v1.4 used `[x, y, z, batch]`; **v2.1 uses `[batch, x, y, z]`**
     (now the *same* convention as MinkowskiEngine). The adapter currently builds batch-last
     (`nn/torchsparse.py:67`, `coords = torch.cat([coordinates.int(), batch.int()], -1)`), so you
     must swap it to batch-first: `torch.cat([batch.int(), coordinates.int()], -1)`. Getting this
     wrong does not crash — it silently corrupts geometry. This is the single most important edit.
   - **`Conv3d` gained a `generative` argument** (default `False`); the existing `transposed=True`
     path still works, so no change is needed unless you want generative transposed convs.
   - Minor v2.1 renames to be aware of if referenced elsewhere: `.dense()` (not `.to_dense()`),
     and negative coords now require `set_allow_negative_coordinates(True)` (not relevant here
     because voxel `coords` are non-negative after `GridSampling3D`).

5. **Optional performance tuning (new in ++).** v2.1 exposes a tuning API you can call once at
   startup: `torchsparse.tune()` to autotune the dataflow, `F.set_conv_mode(0|1|2)` to pick a
   convolution mode, and `F.set_kmap_mode("hashmap")` — the `hashmap` kernel-map mode is the one
   documented as **compatible with MinkowskiEngine**, so it's the safest choice when reproducing
   ME results. (`F` = `torchsparse.nn.functional`.) None of this is required to run; it only
   affects speed.

6. **Caveats already handled for you:**
   - *Weight init:* `BaseSparseConv3d.weight_initialization` initializes `m.kernel`, which the
     TorchSparse `Conv3d` exposes — no change.
   - Once migrated you can drop `MinkowskiEngine==0.5.4` from `requirements.txt`.

**In short:** the model-side change is ~4 lines in `PointGroup3heads.py`, but the substantive
work for **++ specifically** is fixing the coordinate order (and version/install) in the
TorchSparse adapter — because that adapter predates v2.1's `[batch, x, y, z]` convention.

### 8.3 Does it matter if the input is `.npy` single-tree files with semantic labels?

**This is completely independent of the Minkowski↔TorchSparse decision.** The sparse
backend only ever sees `coords`, `features`, and `batch`; it has no idea whether those came
from `.ply`, `.npy`, or anything else. So changing the file format touches only the **data
layer**, not the model.

What you would change:

- **The reader.** `read_treeins_format()` in `datasets/segmentation/treeins_set1.py` currently
  calls `read_ply` and pulls columns `x, y, z, semantic_seg, treeID`. For `.npy` you write a
  small variant that does `np.load(...)` and maps the array columns to positions and the
  semantic label. (You need to know your `.npy` column order — e.g. columns 0–2 = xyz, column
  3 = semantic class.)
- **Instance labels for single-tree files.** The panoptic model expects an instance label per
  point. If every file is exactly one tree, set `instance_labels` to a single constant per file
  (all foreground points = instance 1), or use the file index as the tree id when you later fuse
  files. If you only care about **semantic** segmentation, you can effectively ignore the
  instance branch (e.g. zero its loss weights) — the semantic head works regardless.
- **A dataset class + config.** Add a dataset class analogous to `TreeinsFusedDataset`
  (`datasets/panoptic/treeins_set1.py`) that lists your `.npy` files, and a data config under
  `conf/data/panoptic/` pointing `dataroot` at them and `class:` at the new class.
- **Unchanged:** the backbone, the voxelization (`GridSampling3D` operates on `data.pos`, which
  exists no matter the source format), the heads, and the training loop.

### 8.4 MinkowskiEngine vs TorchSparse — corresponding classes

This table is taken directly from the two adapter files, which are the single place each
mapping is defined: `modules/SparseConv3d/nn/minkowski.py` and
`modules/SparseConv3d/nn/torchsparse.py`. Reading them side by side is the fastest way to
understand how the libraries line up.

| Concept | MinkowskiEngine | TorchSparse (++ / v2.1) | Unified name in torch-points3d |
|---|---|---|---|
| Sparse tensor | `ME.SparseTensor(features, coordinates)` | `torchsparse.SparseTensor(feats, coords)` | `sp3d.nn.SparseTensor(...)` |
| Coordinate order | `[batch, x, y, z]` (batch **first**) | **v2.1/++: `[batch, x, y, z]`** (batch first, same as ME); v1.4 was `[x, y, z, batch]` | set in each adapter — see the ⚠️ note in §8.2, the shipped adapter still uses the v1.4 order |
| Spatial convolution | `ME.MinkowskiConvolution` | `torchsparse.nn.Conv3d` | `sp3d.nn.Conv3d` |
| Transposed / up conv | `ME.MinkowskiConvolutionTranspose` | `torchsparse.nn.Conv3d(transposed=True)` | `sp3d.nn.Conv3dTranspose` |
| Batch norm | `ME.MinkowskiBatchNorm` | `torchsparse.nn.BatchNorm` | `sp3d.nn.BatchNorm` |
| Activation | `ME.MinkowskiReLU` | `torchsparse.nn.ReLU` | `sp3d.nn.ReLU` |
| Concatenate tensors | `ME.cat` | `torchsparse.cat` | `sp3d.nn.cat` |
| Read features | `tensor.F` | `tensor.F` | `.F` (same on both) |
| Read coordinates | `tensor.C` | `tensor.C` | `.C` (same on both) |
| Residual blocks | `modules/MinkowskiEngine/api_modules.py` | `modules/SparseConv3d/modules.py` | same class names: `ResBlock`, `ResNetDown`, `ResNetUp` |
| Backbone builder (application) | `applications/minkowski.py::Minkowski` | `applications/sparseconv3d.py::SparseConv3d(backend="torchsparse")` | — |

**Where to look in the code to learn each library:** open the two adapter files together —
`nn/minkowski.py` (~65 lines) and `nn/torchsparse.py` (~68 lines). They implement the exact
same six names, so reading them side by side is the quickest way to see how the libraries line
up. Note the caveat from §8.2: `nn/torchsparse.py` was written for TorchSparse **v1.4**, so its
`SparseTensor` still assembles coordinates as `[x, y, z, batch]`. Under **TorchSparse++ (v2.1)**
the convention is `[batch, x, y, z]` — identical to Minkowski — so for ++ that line must be
flipped to match `nn/minkowski.py`.
