# MinkowskiEngine vs. TorchSparse++ as the ForAINet backbone

Could **TorchSparse++** replace **MinkowskiEngine** as the sparse-convolution backbone in
ForAINet's `PointCloudSegmentation/`, and would it achieve better results?

## Short answer

**Yes, it's technically feasible — but it's a nontrivial port, not a config flip, and it
would very likely improve *speed/memory*, not *segmentation accuracy*.** TorchSparse++ is an
efficiency framework, not a more-accurate operator.

## How the backbone is wired today

ForAINet's panoptic model is **hard-wired to MinkowskiEngine**, and it deliberately bypasses
the one abstraction that would have made this easy:

- `torch_points3d/models/panoptic/PointGroup3heads.py` (lines 11, 32) imports
  `from torch_points3d.applications.minkowski import Minkowski` and builds
  `self.Backbone = Minkowski(...)`. The scorer sub-nets (`ScorerUnet`, `ScorerEncoder`) are
  Minkowski too.
- The backbone's `BaseMinkowski._set_input` (in
  `torch_points3d/applications/minkowski.py`, line 113) constructs an `ME.SparseTensor`
  directly, and the forward passes read Minkowski-specific `.F` / `.C` accessors.

Meanwhile the repo **already has a backend-agnostic layer** that ForAINet does *not* use:

- `torch_points3d/modules/SparseConv3d/nn/` exposes a common API (`Conv3d`, `SparseTensor`,
  `cat`, …) with a `set_backend("torchsparse" | "minkowski")` switch, and there's an
  `applications/sparseconv3d.py` with a `backend=` argument. The README even documents
  `SparseConv3d(..., backend="torchsparse")`.

**The catch:** that built-in `torchsparse.py` backend targets **TorchSparse v1**
(`>= v1.4.0`) — a different, older API than **TorchSparse++ (v2.x)**. See
`torch_points3d/modules/SparseConv3d/nn/torchsparse.py`
(`import torchsparse as TS; TS.nn.Conv3d(...)`) and the README line "torchsparse >= v1.4.0".

## So the actual work to use TorchSparse++

Two layers of effort:

1. **Port the panoptic model off raw Minkowski onto the `SparseConv3d` abstraction** —
   rewrite `_set_input`, the U-Net forward, and the scorer nets to stop calling
   `ME.SparseTensor` / `.F` / `.C` directly. This is the bulk of the work and touches the
   clustering/scoring path too.
2. **Write a TorchSparse++ (v2) adapter** in `modules/SparseConv3d/nn/`, because the existing
   torchsparse backend won't run against v2's changed API/build. Plus environment friction:
   TorchSparse++ needs its own CUDA build and (historically) has no CPU path.

## Would results actually get better?

Distinguish two meanings of "better":

- **Accuracy (IoU / panoptic quality): almost certainly no meaningful change.** Sparse
  convolution is the same mathematics regardless of backend — same weights, same architecture
  → essentially the same predictions, modulo numerical noise and any mixed-precision effects.
  TorchSparse++'s contributions (adaptive kernel grouping, kernel-map reuse, better
  scheduling) optimize *how fast* the ops run, not *what* they compute. Small differences can
  appear from how each library hashes/aggregates duplicate voxel coordinates and breaks ties
  in downsampling, but that's as likely to hurt as help — not a reliable accuracy win.

- **Efficiency: yes, plausibly a real win** — higher throughput, lower memory, and
  mixed-precision support. On this pipeline that matters: the backbone runs on dense 8 m
  cylinders voxelized to 0.2 m, so faster/leaner sparse conv lets you use bigger batches,
  larger radii, or more augmentation on the same GPU.

The *indirect* path to better accuracy is real but second-order: cheaper training → more
experimentation (more epochs, bigger models, the ablations the paper used).

## Where ForAINet's accuracy is actually bottlenecked

Worth being clear about this, because it's the crux of the "better results" question:
ForAINet's quality is dominated by the **instance branch** (offset/embedding heads, the
`_cluster7` grouping, and the scorer nets), the **semantic head**, and **class imbalance**
(ground/veg vs. stems), plus data. The paper's gains came from backbone-independent tricks —
class weights, height weights, binary-semantic loss, TreeMix (see `ForAINet/readme.md`). None
of those are affected by which sparse-conv kernel runs underneath.

## Recommendation

- If the goal is **faster/cheaper training or bigger inputs** → TorchSparse++ (or even just
  the built-in torchsparse v1 backend with mixed precision) is a defensible investment, but
  budget for the port described above.
- If the goal is **higher segmentation accuracy** → don't expect it from a backbone swap.
  Spend the effort on the instance-clustering/scorer design, loss weighting, and data (the
  paper's own ablation levers).
