# Swapping the backbone: MinkowskiEngine → TorchSparse 1.4

How to replace the sparse-convolution backend in the ForAINet / torch-points3d fork, why
it is easier than it looks, and exactly which lines have to change.

> **This is now implemented.** Both backbones are available side by side and selected by
> `model_name`; **MinkowskiEngine remains the default**. Jump to
> [§0 Switching backbones](#0-switching-backbones-implemented) for the commands. The
> sections after it are the analysis the implementation was based on — still accurate, and
> worth reading before changing anything.

---

## 0. Switching backbones (implemented)

Two model blocks in `conf/models/panoptic/FORpartseg_3heads.yaml`, one knob:

| `model_name` | Backbone | Model file |
|---|---|---|
| **`PointGroup-PAPER`** | **MinkowskiEngine — default** | `PointGroup3heads.py` |
| `PointGroup-PAPER-TS` | TorchSparse 1.4 | `PointGroup3heads_ts.py` |

```bash
# MinkowskiEngine — the default. This is what conf/config.yaml already selects.
python train.py task=panoptic data=panoptic/treeins_set1 \
  models=panoptic/FORpartseg_3heads model_name=PointGroup-PAPER \
  training=treeins_set1

# TorchSparse 1.4
python train.py task=panoptic data=panoptic/treeins_set1 \
  models=panoptic/FORpartseg_3heads model_name=PointGroup-PAPER-TS \
  training=treeins_set1
```

Same switch for `eval.py`. To make TorchSparse permanent, change `model_name` in
`conf/config.yaml`; to roll back, run `PointGroup-PAPER`.

**How it is wired.** `PointGroup-PAPER-TS` is a four-line block that inherits the whole of
`PointGroup-PAPER` through a YAML merge key, so the two cannot drift:

```yaml
PointGroup-PAPER: &paper
  class: PointGroup3heads.PointGroup3heads
  ...

PointGroup-PAPER-TS:
  <<: *paper
  class: PointGroup3heads_ts.PointGroup3heads
  backend: "torchsparse"
```

`PointGroup3heads_ts.py` is a copy of `PointGroup3heads.py` differing in four lines: the
import, and `backend=` on the three `SparseConv3d(...)` constructions. It is a **copy
rather than an edit** because the two applications resolve `ResNetDown`/`ResNetUp` against
different modules (`modules/MinkowskiEngine/api_modules` vs `modules/SparseConv3d/modules`,
see §2), so editing in place would change the Minkowski path too and invalidate the
baseline in [`train_logs.md`](train_logs.md).

> ⚠️ **`SPARSE_BACKEND` silently overrides the `backend` key.** If that environment
> variable is set in your shell or container, `PointGroup-PAPER-TS` will not run
> TorchSparse no matter what the config says (`sparseconv3d.py:59-62`). It is not part of
> the interface here — `unset SPARSE_BACKEND` and choose with `model_name`. It also has no
> effect at all on `PointGroup-PAPER`, which calls `Minkowski()` and never reads it.

Both libraries live only inside the container. Confirm before a run:

```bash
python -c "import torchsparse; print(torchsparse.__version__)"   # 1.4.0
echo "SPARSE_BACKEND=[$SPARSE_BACKEND]"                          # must be empty
```

---

For TorchSparse **2.x / TorchSparse++**, see the separate
[`backbone_torchsparse_v2.md`](backbone_torchsparse_v2.md) — it is a very different story.
For the pipeline map this document assumes, see
[`train_logs_explained.md`](train_logs_explained.md).

All paths are relative to `ForAINet/PointCloudSegmentation/`.

---

## 1. What a "backbone" is here, and what you are actually changing

The **backbone** is the U-Net that turns your `[N,4]` input features into a `[N,16]`
per-point feature vector. The three heads (Semantic / Offset / Embed) sit on top of it
and are *not* affected by this change — they are plain `Linear` layers on point features.

What you are swapping is the **sparse-convolution library** underneath the U-Net:

| | MinkowskiEngine (current) | TorchSparse 1.4 |
|---|---|---|
| Author | NVIDIA | MIT Han Lab |
| Repo | `NVIDIA/MinkowskiEngine` | `mit-han-lab/torchsparse` |
| In your image? | ✅ `Dockerfile:73-75` | ✅ `Dockerfile:77-78` (already installed!) |
| Build pain | high (long compile, CUDA-arch sensitive) | moderate |
| Typical speed | baseline | often faster on the same kernels |
| Coordinate order | `[batch, x, y, z]` | `[x, y, z, batch]` ← **note the difference** |
| torch-points3d support | legacy wrapper | first-class backend |

Both implement the same idea (convolution over sparse voxels), so the *network shape*,
the yaml, and the training loop stay identical. Only the kernel implementation changes.

**Why bother?** Honestly: for this project, mainly speed and to escape a deprecated API.
The framework itself tells you so on every run —
`applications/minkowski.py:48-50` prints:

> Minkowski API is deprecated in favor of the SparseConv3d API. It should be a simple drop
> in replacement (no change to the API).

That warning appears 3× in your log (Backbone, ScorerUnet, ScorerEncoder).

---

## 2. The one thing you must understand first

### ❌ The `class: minkowski.Minkowski_Model` in the yaml is a red herring

Your model config contains this, three times
([FORpartseg_3heads.yaml:92](ForAINet/PointCloudSegmentation/conf/models/panoptic/FORpartseg_3heads.yaml#L92),
[:130](ForAINet/PointCloudSegmentation/conf/models/panoptic/FORpartseg_3heads.yaml#L130),
[:149](ForAINet/PointCloudSegmentation/conf/models/panoptic/FORpartseg_3heads.yaml#L149)):

```yaml
backbone:
  architecture: "unet"
  config:
    class: minkowski.Minkowski_Model     # ← looks like the backbone selector. It is NOT.
```

**Editing this key changes nothing.** Verified chain:

1. `MinkowskiFactory._build_unet` ([minkowski.py:58-66](ForAINet/PointCloudSegmentation/torch_points3d/applications/minkowski.py#L58-L66))
   constructs the U-Net with `model_type=None`.
2. `UnwrappedUnetBasedModel._get_factory` ([unet.py:450-454](ForAINet/PointCloudSegmentation/torch_points3d/models/base_architectures/unet.py#L450-L454))
   does `getattr(modules_lib, "NoneFactory", None)` → not found → falls back to `BaseFactory`.
3. Modules are then resolved purely by `module_name` (`ResNetDown` / `ResNetUp`) looked up
   in `modules_lib`, which `minkowski.py:65` sets to
   `sys.modules["torch_points3d.applications.minkowski"]` — i.e. whatever that file
   star-imported at its line 9.

### ✅ The real selector is a Python import

[`PointGroup3heads.py:11`](ForAINet/PointCloudSegmentation/torch_points3d/models/panoptic/PointGroup3heads.py#L11):

```python
from torch_points3d.applications.minkowski import Minkowski
```

**That line is the backbone choice.** Everything else follows from it. So this is a small
code change, not a config change — which is the opposite of what the config layout
suggests.

---

## 3. The good news: torch-points3d already has a backend switch

You do not have to write any sparse-conv code. `applications/sparseconv3d.py` is the
modern, backend-agnostic replacement for `applications/minkowski.py`, and it accepts a
`backend` argument ([sparseconv3d.py:26-34](ForAINet/PointCloudSegmentation/torch_points3d/applications/sparseconv3d.py#L26-L34)):

```python
def SparseConv3d(
    architecture: str = None,
    input_nc: int = None,
    num_layers: int = None,
    config: DictConfig = None,
    backend: str = "minkowski",     # ← "torchsparse" or "minkowski"
    ...
):
```

and applies it at [lines 59-62](ForAINet/PointCloudSegmentation/torch_points3d/applications/sparseconv3d.py#L59-L62):

```python
if "SPARSE_BACKEND" in os.environ and sp3d.nn.backend_valid(os.environ["SPARSE_BACKEND"]):
    sp3d.nn.set_backend(os.environ["SPARSE_BACKEND"])
else:
    sp3d.nn.set_backend(backend)
```

The switch itself lives in
[`torch_points3d/modules/SparseConv3d/nn/__init__.py`](ForAINet/PointCloudSegmentation/torch_points3d/modules/SparseConv3d/nn/__init__.py):

```python
__all__ = ["cat", "Conv3d", "Conv3dTranspose", "ReLU", "SparseTensor", "BatchNorm"]   # :21

def backend_valid(_backend):
    return _backend in {"torchsparse", "minkowski"}                                   # :25-26

def set_backend(_backend):                                                            # :33
    modules = importlib.import_module("." + _backend, __name__)                       # :46
    for val in __all__:
        exec("globals()['%s'] = modules.%s" % (val, val))                             # :51-52
```

It rebinds six symbols to whichever backend module you name. The two implementations are
`nn/minkowski.py` and `nn/torchsparse.py`, and the blocks in
`modules/SparseConv3d/modules.py` reference them **late** (`snn.Conv3d`, not
`from ... import Conv3d`), which is what makes the swap work at runtime.

**So the total work is: point the model at `SparseConv3d` instead of `Minkowski`, and pass
`backend="torchsparse"`.**

---

## 4. The change list

> **Implemented, with one deviation.** The boxed suggestion below — copy the model file
> rather than edit it — is what was actually done, so `PointGroup3heads.py` is untouched
> and MinkowskiEngine stays the default. The edits described in Steps 1-2 therefore live
> in `PointGroup3heads_ts.py` and the `PointGroup-PAPER-TS` block. See [§0](#0-switching-backbones-implemented).

### Step 1 — the one required code edit

[`torch_points3d/models/panoptic/PointGroup3heads.py`](ForAINet/PointCloudSegmentation/torch_points3d/models/panoptic/PointGroup3heads.py)

**Line 11** — swap the import:

```python
# from torch_points3d.applications.minkowski import Minkowski
from torch_points3d.applications.sparseconv3d import SparseConv3d
```

**Lines 32-37, 47, 48-50** — swap the three constructor calls and thread the backend
through. All three must use the **same** backend, because `set_backend` mutates global
module state:

```python
_backend = option.get("backend", "torchsparse")

self.Backbone = SparseConv3d(
    backbone_options.get("architecture", "unet"),
    input_nc=dataset.feature_dimension,
    num_layers=4,
    config=backbone_options.get("config", {}),
    backend=_backend,
)
...
self.ScorerUnet = SparseConv3d("unet", input_nc=self.Backbone.output_nc,
                               num_layers=4, config=option.scorer_unet, backend=_backend)
self.ScorerEncoder = SparseConv3d("encoder", input_nc=self.Backbone.output_nc,
                                  num_layers=4, config=option.scorer_encoder, backend=_backend)
```

> Consider copying `PointGroup3heads.py` to a new file (e.g. `PointGroup3heads_ts.py`) and
> pointing a new yaml block at it via `class:`, rather than editing in place. That way you
> can A/B the two backends without a git checkout — and the `class:` key *is* honoured for
> selecting the model file (unlike the backbone `class:` key).

### Step 2 — the config

Add one key under `PointGroup-PAPER` in
[`conf/models/panoptic/FORpartseg_3heads.yaml`](ForAINet/PointCloudSegmentation/conf/models/panoptic/FORpartseg_3heads.yaml)
(near `conv_type: "SPARSE"` on line 65):

```yaml
PointGroup-PAPER:
  class: PointGroup3heads.PointGroup3heads
  conv_type: "SPARSE"
  backend: "torchsparse"      # ← new
```

This mirrors the pattern already used by the segmentation models —
`conf/models/segmentation/sparseconv3d.yaml:5` has exactly this key.

The `down_conv` / `up_conv` blocks, `module_name: ResNetDown/ResNetUp`, and `dimension: 3`
need **no changes**. (`dimension` is not a parameter of the SparseConv3d blocks, but it is
absorbed by `**kwargs`, so it is harmless.)

You may also delete the misleading `class: minkowski.Minkowski_Model` lines (92, 130, 149)
since they do nothing — but leaving them is equally harmless.

### Step 3 — no-edit alternative for a quick test

Because of the env-var branch at `sparseconv3d.py:59-60`, once step 1 is done you can flip
backends without touching yaml at all:

```bash
export SPARSE_BACKEND=torchsparse   # or: minkowski
```

Good for A/B benchmarking in one container session.

---

## 5. What could bite you — honest list

These are real differences between the two code paths, ranked by how likely they are to
affect **your** config (`scorer_type: "unet"`).

### 🟢 Not a problem for you: the model has no direct Minkowski calls

`PointGroup3heads.py` contains **zero** `ME.*` API usage. Its only coupling to
MinkowskiEngine is the import on line 11 and the three constructor calls. The scorer path
passes plain PyG `Data` objects and only reads `.x` off the result — all backend-agnostic.
This is why the swap is realistic at all.

### 🟡 Coordinate column order differs

- `nn/minkowski.py` builds coords as `[batch, x, y, z]` (batch **first**)
- `nn/torchsparse.py` builds coords as `[x, y, z, batch]` (batch **last**)

Anything doing `data.C[:, 0]` to get the batch index is **wrong under torchsparse**.
There is exactly such a line at
[`sparseconv3d.py:164`](ForAINet/PointCloudSegmentation/torch_points3d/applications/sparseconv3d.py#L164)
in `SparseConv3dEncoder.forward`.

**Does it affect you?** With `scorer_type: "unet"` (your setting,
[FORpartseg_3heads.yaml:66](ForAINet/PointCloudSegmentation/conf/models/panoptic/FORpartseg_3heads.yaml#L66))
the encoder's `forward` is never called, so you dodge it. If you switch to
`FORpartseg_3heads_encoderScore.yaml`, fix that line first.

### 🟡 The U-Net return value differs

- `MinkowskiUnet.forward` returns `Data(x=..., pos=..., batch=data.C[:,0])`
- `SparseConv3dUnet.forward` returns `Batch(x=..., pos=...)` — **no `batch` field**

`PointGroup3heads.forward` only reads `.x`, so this is safe here. But any code you add
that expects `backbone_out.batch` will break.

### 🟡 BatchNorm scheduler is Minkowski-aware

[`core/schedulers/bn_schedulers.py:6-17`](ForAINet/PointCloudSegmentation/torch_points3d/core/schedulers/bn_schedulers.py#L6-L17)
builds its `BATCH_NORM_MODULES` tuple with `ME.MinkowskiBatchNorm` in it. The torchsparse
`BatchNorm` wrapper subclasses `nn.BatchNorm1d`, which *is* already in the tuple, so BN
momentum scheduling should still reach it — but this works by inheritance luck rather than
by design. If your BN momentum appears not to decay, look here first.

### 🟡 `set_backend` swallows import errors confusingly

In `nn/__init__.py`, `modules` is bound inside the `try` (line 46) but used outside it
(line 52). If torchsparse fails to import you get a `NameError` *after* the real exception
is logged — read the `log.exception` output, not the `NameError`.

### 🔴 Don't let PVCNN get imported

`torch_points3d/modules/PVCNN/*` targets the **torchsparse ≤1.2** API
(`torchsparse.sparse_tensor`, `torchsparse.utils.kernel_region`), which does not exist in
1.4. It is not on your code path — just don't import it.

### 🔴 Other panoptic model files are hard-wired to Minkowski

`PointGroup3heads_new.py`, `PointGroup3heads_backup.py`, `pointgroup.py`, and
`models/panoptic/minkowski.py` all use `ME.SparseTensor` / `MinkowskiGlobalMaxPooling`
directly. They are **not** used by your config, but they would each need a real rewrite.
Leave them alone.

---

## 6. Environment check

Both backends are installed in the Docker image:

```dockerfile
# Dockerfile:73-75  MinkowskiEngine
# Dockerfile:77-78  git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0
```

Confirm inside the container before starting:

```bash
python -c "import torchsparse;      print('torchsparse', torchsparse.__version__)"
python -c "import MinkowskiEngine;  print('ME ok')"
```

> Note: neither library is importable in the **Windows host** environment — they are
> CUDA extensions that only exist inside the container. All of this work happens in the
> container (see [`run_debugger.md`](run_debugger.md)).

---

## 7. Verification — how to know it worked

Do **not** judge by "it ran". Compare against the baseline already captured in
[`train_logs.md`](train_logs.md):

| Check | Baseline (Minkowski) | What to expect |
|---|---|---|
| Module tree | `MinkowskiConvolution(in=4, out=16, ...)` | should now print torchsparse `Conv3d` modules, same channel widths `4→16→32→48→64→80→96→112` |
| `Model size` | `11872126` in that July log, which **predates the 4-class patch** | **`11872109`** — the 4-class head, 17 parameters (one `Linear(16→N)` output) fewer. Both backends log exactly this; any other difference means the architecture changed, not just the kernels |
| Deprecation warning | present 3× | **gone** — you are off the legacy API |
| `train_acc` after 1 epoch | ~75 | same ballpark |
| `val_miou` after 2 epochs | ~53 | same ballpark |
| `s/it` | 1.38 s/it (epoch 1) | this is the number you are trying to improve |

Suggested procedure:

```bash
# 1. Baseline, for a fair timing comparison on the same machine
SPARSE_BACKEND=minkowski python train.py task=panoptic data=panoptic/treeins_set1 \
  models=panoptic/FORpartseg_3heads model_name=PointGroup-PAPER \
  training=treeins_set1 job_name=bench_mink training.epochs=2

# 2. TorchSparse
SPARSE_BACKEND=torchsparse python train.py task=panoptic data=panoptic/treeins_set1 \
  models=panoptic/FORpartseg_3heads model_name=PointGroup-PAPER \
  training=treeins_set1 job_name=bench_ts training.epochs=2
```

Then compare the two wandb runs side by side.

> ⚠️ **Two epochs will not tell you anything about instance quality** — clustering is
> gated behind `prepare_epoch: 30`, so all instance metrics will be 0.0 in both runs. See
> [`train_logs_explained.md` §C1](train_logs_explained.md). For a backend swap that is
> fine: you are validating *equivalence and speed*, not accuracy. Numerical results will
> not be bit-identical between backends (different kernels, different reduction order), so
> compare trends over a few epochs, not single values.

---

## 8. Summary

| | |
|---|---|
| **Files added** | `PointGroup3heads_ts.py` (copy + 4 lines), `FORpartseg_3heads.yaml` (anchor + 4-line block) |
| **Files NOT edited** | `PointGroup3heads.py` — MinkowskiEngine stays the default and the baseline stays reproducible |
| **Files NOT to edit** | anything under `modules/SparseConv3d/` — the backend switch already exists |
| **Risk** | low-moderate — the model has no direct Minkowski API calls |
| **Main trap** | the `class: minkowski.Minkowski_Model` yaml key is inert; the *import* is the real switch |
| **Switch** | `model_name=PointGroup-PAPER` (Minkowski) / `model_name=PointGroup-PAPER-TS` (TorchSparse) |
| **Rollback** | run `PointGroup-PAPER` |
