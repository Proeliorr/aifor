# TorchSparse 2.x / TorchSparse++ — feasibility assessment

Companion to [`backbone_torchsparse_1.4.md`](backbone_torchsparse_1.4.md). That document
describes a change you can realistically make this week. **This one describes a change you
probably should not make at all** — and explains why, so the decision is yours and
informed.

---

## 0. The short answer

| | TorchSparse **1.4** | TorchSparse **2.x / ++** |
|---|---|---|
| Already installed in your image | ✅ yes (`Dockerfile:77-78`) | ❌ no |
| torch-points3d backend exists | ✅ `modules/SparseConv3d/nn/torchsparse.py` | ❌ **none — you would write it** |
| Config-level switch | ✅ `backend: "torchsparse"` | ❌ not a recognised backend name |
| Compatible with this image's torch 1.9 / CUDA 11.1 | ✅ | ⚠️ **unlikely — needs verification** |
| Realistic effort | ~1 file, ~5 lines | new backend module + probable full image rebuild |
| Risk to your working environment | low, reversible | **high** — the image "cannot be reproduced" |

**Recommendation: validate on 1.4 first.** If 1.4 gives you the speedup you need, stop
there. Only revisit v2 if you have a measured, specific reason that 1.4 cannot satisfy,
*and* you are willing to rebuild the environment.

---

## 1. Why this is a different kind of change from 1.4

The 1.4 swap works because someone already wrote the adapter. torch-points3d's backend
switch only knows two names
([`modules/SparseConv3d/nn/__init__.py:25-26`](ForAINet/PointCloudSegmentation/torch_points3d/modules/SparseConv3d/nn/__init__.py#L25-L26)):

```python
def backend_valid(_backend):
    return _backend in {"torchsparse", "minkowski"}
```

and `set_backend` imports a sibling module of exactly that name
([line 46](ForAINet/PointCloudSegmentation/torch_points3d/modules/SparseConv3d/nn/__init__.py#L46)).
The existing `nn/torchsparse.py` targets the **1.4** API.

So for v2 there is nothing to flip. You would be **writing a new backend**, implementing
six symbols against the v2 API:

```python
__all__ = ["cat", "Conv3d", "Conv3dTranspose", "ReLU", "SparseTensor", "BatchNorm"]
```

That is genuinely doable — the surface is small, and `nn/torchsparse.py` (~70 lines) is a
working template. The problem is not the adapter. **The problem is the environment.**

---

## 2. The real blocker: your dependency stack is from 2021

Read straight off [`Dockerfile`](ForAINet/PointCloudSegmentation/Dockerfile):

| Component | Pinned version | Line |
|---|---|---|
| Base image | `nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04` | 1 |
| Python | 3.8 | 10 |
| PyTorch | **1.9.0+cu111** | 60 |
| torchvision | 0.10.0 | 61 |
| torch-scatter | 2.0.8 | 66 |
| torch-sparse | 0.6.12 | 67 |
| torch-cluster | 1.5.9 | 68 |
| torch-geometric | **1.7.2** | 70 |
| torch-points-kernels | 0.7.0 | 83 |
| MinkowskiEngine | git HEAD | 74 |
| torchsparse | **v1.4.0** | 78 |

Every one of the `torch-*` packages is a **compiled CUDA extension built against torch
1.9**. They are not pure Python — you cannot bump torch without rebuilding all of them.

TorchSparse 2.x is a modern release and, to my knowledge, targets substantially newer
PyTorch and CUDA than 1.9 / 11.1. **I have not verified this against the specific release
you would install** — check the project's own requirements before trusting my summary. But
the direction of the problem is clear enough to plan around:

```
install torchsparse 2.x
   → probably requires newer torch
       → must rebuild torch-scatter / torch-sparse / torch-cluster / torch-points-kernels
           → must rebuild MinkowskiEngine (which is famously fragile to build)
               → torch-geometric 1.7.2 API is old; newer PyG changed Data/Batch semantics
                   → dataset code in this fork may need porting
```

That last arrow is not hypothetical: the fork's dataset layer uses PyG 1.7-era idioms
(`Batch.from_data_list`, attribute-style `Data` construction) that changed in PyG 2.x.

### And the constraint you already stated

You told me the `for-ai-net` image **cannot be reproduced**. That is the decisive fact.
A v2 upgrade is not a code change to a working environment — it is a request to rebuild
the environment you cannot rebuild. If the rebuild fails, you do not have a fallback.

---

## 3. What v2 would actually buy you

Being fair to the option — the TorchSparse++ line exists because it is genuinely faster.
The published claims centre on better kernel scheduling and adaptive matmul grouping, with
meaningful speedups over both v1.x and MinkowskiEngine on segmentation/detection workloads.

**Treat those numbers as vendor claims until you measure them on your data.** Sparse-conv
performance depends heavily on voxel density and network shape, and your setup is unusual:
0.2 m voxels over 8 m forest cylinders, a 7-level U-Net with narrow channels (16→112).
That is a small, deep, narrow network — not the wide indoor-scan benchmarks these libraries
are usually tuned against. Your speedup may be much smaller than headline figures.

Also worth weighing: **~⅓ of your parameters are wasted anyway.** `ScorerEncoder` and
`ScorerMLP` are built but never called at `scorer_type: "unet"` (see
[`train_logs_explained.md` §A6](train_logs_explained.md)). Not building them is free
performance available *right now*, with no backend change and no risk.

And your biggest observed performance problem is not kernel speed at all — it is the
unexplained 5.5× epoch-2 slowdown ([§C3](train_logs_explained.md)), which no backend swap
will fix.

---

## 4. If you decide to do it anyway — the staged path

Do not attempt this as one change. Stage it so each step is independently verifiable and
you can stop at any point with a working system.

**Stage 0 — exhaust the cheap wins first**
- Make `ScorerEncoder` / `ScorerMLP` conditional on `scorer_type`.
- Diagnose the epoch-2 slowdown.
- Measure. You may already be done.

**Stage 1 — prove the abstraction works with 1.4**
Follow [`backbone_torchsparse_1.4.md`](backbone_torchsparse_1.4.md). This is the real
prerequisite: it proves the model runs on a non-Minkowski backend at all, and it forces you
to hit the coordinate-ordering and `.batch` issues documented there while you still have a
working fallback. **Do not skip to v2 from Minkowski.**

**Stage 2 — a throwaway environment**
Build a *new* image (new tag, do not overwrite `for-ai-net`) with the newer torch/CUDA and
torchsparse 2.x. Expect this to be the bulk of the work. Success criterion is narrow:
`import torchsparse` works and a toy sparse conv runs.

**Stage 3 — write the backend adapter**
Create `torch_points3d/modules/SparseConv3d/nn/torchsparse_v2.py` implementing the six
symbols, using `nn/torchsparse.py` as the template. Add `"torchsparse_v2"` to
`backend_valid` in `nn/__init__.py:25-26`. Pay specific attention to:
- **Coordinate ordering** — 1.4 uses `[x,y,z,batch]`; confirm what v2 expects.
- **`SparseTensor` constructor signature** — this changed between versions.
- **Transposed convolution** — 1.4 does it via a `transposed=True` flag.

**Stage 4 — equivalence testing**
Same checks as the 1.4 doc: `Model size` should match, channel widths should match, and
2-epoch metrics should land in the same ballpark. Divergence in parameter count means you
built a different network, not a faster one.

---

## 5. Honest summary of what I do and don't know

**Verified against this repo:**
- torch-points3d has exactly two sparse backends, `minkowski` and `torchsparse` (1.4-era) —
  `nn/__init__.py:25-26`
- No v2 backend exists anywhere in the fork
- The image pins torch 1.9.0+cu111, Python 3.8, CUDA 11.1, PyG 1.7.2 — `Dockerfile`
- torchsparse 1.4.0 **is** already installed — `Dockerfile:78`
- `PointGroup3heads.py` has no direct Minkowski API calls, making a backend swap viable

**My assessment, not verified:**
- That torchsparse 2.x requires newer torch/CUDA than this image provides
- That the upgrade would cascade through the compiled `torch-*` extensions
- Any specific v2 performance figure

**Unknown until you test:**
- Whether v2 helps *your* workload (narrow deep U-Net, 0.2 m voxels, forest cylinders)
- Whether the v2 API maps cleanly onto the six-symbol backend interface

Before committing engineering time, check the current torchsparse release notes and
requirements directly — my knowledge has a cutoff and this is exactly the kind of detail
that moves.

---

## 6. Bottom line

The 1.4 swap is a genuine option: the adapter exists, the library is installed, and the
rollback is one line.

The v2 swap is an environment migration wearing the costume of a backbone change. The code
work (a ~70-line adapter) is the *small* part; rebuilding a stack you have described as
irreproducible is the large and risky part — and the performance payoff on your particular
network shape is unproven.

**Do stage 0 and stage 1. Measure. Then decide whether stage 2 is worth it.**
