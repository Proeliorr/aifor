# Pre-flight on a cheap GPU before the paid A100

**Goal:** prove the whole chain runs — the LAZ→PLY converter, the smoke test, and training
**startup for both backbones** — on a GPU that costs cents, so the only thing left to buy on
the A100 is time. Everything expensive to discover on a paid machine (a broken data path, a
missing LAZ backend, an unapplied 4-class patch, an unsupported GPU) is caught here first.

This is the cheap-rented-GPU version of [`gpu_training_runbook.md`](gpu_training_runbook.md)
Part A. That runbook's Part A assumed a local NVIDIA GPU; when there isn't one, rent the
**cheapest supported card** for ~20–40 minutes instead. It costs about **$0.10–0.40** and
protects a **$16–30** run.

> One command does all of it: [`container_export/preflight.sh`](../container_export/preflight.sh).
> The sections below explain what it checks and why. Run it by hand the first time so you see
> each step; after that it is a repeatable PASS/FAIL gate.

---

## 0. What must be true before you start

| Thing | Why | How it's provided |
|---|---|---|
| The `for-ai-net` image on Docker Hub | the environment (CUDA 11.1, torch 1.9, MinkowskiEngine, TorchSparse) | you pushed it in `gpu_training_runbook.md` Part B |
| `DATASET_TRAIN_URL` env var | archive of the 14 `.laz` + `_offsets.yml` (the modified SegmentedForest dataset) | set on the Vast instance at creation |
| `DATASET_PATCH` env var | archive with `patches/forainet-local.patch` **and** the converter (`pipeline/`, `smoke_test.py`) | set on the Vast instance at creation |
| `WANDB_API_KEY` env var | only needed for the optional step F | set on the Vast instance at creation |

The pre-flight reads those env vars, so nothing is hardcoded and the same image works for the
cheap test and the A100 run.

---

## 1. Rent the right cheap GPU — this is the one that wastes money

The image's CUDA kernels were compiled for compute capability **`6.0 7.0 7.5 8.0 8.6`** with
**no `+PTX` fallback**. A card outside that list has neither a matching kernel nor PTX to
JIT-compile one, and every convolution dies with `no kernel image is available for execution`
— on a machine you are paying for.

| Rent (cheap **and** supported) | cc | Do **not** rent | cc |
|---|---|---|---|
| **T4** (usually cheapest, 16 GB) | 7.5 | RTX 4090 | 8.9 |
| **RTX 3090** (fast, 24 GB) | 8.6 | L40 / L40S | 8.9 |
| RTX 2080 Ti, A10, A40, V100, P100 | 7.5 / 8.6 / 7.0 / 6.0 | H100 | 9.0 |

> ⚠️ On Vast, 4090/L40S instances are common and often *cheaper per hour* than a T4. They
> will **not run this image.** Filter for **T4** or **3090**. The pre-flight also checks the
> live card's capability (step A) and fails loudly if you got this wrong — but it's cheaper to
> pick right than to rent, fail, and destroy.

**Disk:** ~40 GB is plenty for the pre-flight if you convert a 3-plot subset; use **60 GB**
if you convert all 14 (4.3 GB archive → ~29 GB of PLY). The full A100 run needs 60 GB.

**Why a cheap card is a valid test:** the 2-epoch run fits in **6 GB of VRAM** (it ran on a
GTX 1660 Ti), so a 16 GB T4 has room to spare. A cheap card validates the data path, the
patch, the converter and that both backbones *start*; it cannot tell you wall-clock speed or
whether 150 epochs finish — but the A100 only has **more** memory, so nothing that passes here
fails there for capacity reasons.

---

## 2. Run the gate

SSH into the instance (Vast shows the command). The script itself must be on the box before
it can fetch anything (step B is *inside* it), so get it there one of two ways:

- **Re-pack `container_export/` and point `DATASET_PATCH` at it.** `preflight.sh` now lives in
  `container_export/`, so a freshly regenerated `container_export.tar.gz`
  (`tar -czf container_export.tar.gz -C container_export .`) contains it. Then on the instance:
  ```bash
  mkdir -p /opt/prep && cd /opt/prep
  wget -O c.tgz "$DATASET_PATCH" && tar -xzf c.tgz    # now /opt/prep/preflight.sh exists
  bash /opt/prep/preflight.sh
  ```
- **Or `scp` it up directly** (it's a single ~18 KB file) and run it:
  ```bash
  scp -P <port> container_export/preflight.sh root@<host>:/root/
  ssh -p <port> root@<host> 'bash /root/preflight.sh'   # full gate, all 14 plots, both backbones
  ```

Common variants (env vars in front of the command):

```bash
# See the whole plan without running anything (no GPU needed — good for a first read):
PREFLIGHT_DRYRUN=1 bash preflight.sh

# Faster / cheaper: one plot per split is enough to prove startup:
PREFLIGHT_PLOTS="plot_02 plot_11 plot_01" bash preflight.sh

# Only the converter + smoke test (scope #1), skip training:
PREFLIGHT_SKIP_TRAIN=1 bash preflight.sh

# Also validate wandb (the key + entity) before the 8-hour A100 run uses it:
PREFLIGHT_WANDB=1 bash preflight.sh
```

Exit code `0` and a green **PRE-FLIGHT PASSED** = cleared to rent the A100.

---

## 3. What each step checks (and how to read a failure)

| Step | Checks | If it fails |
|---|---|---|
| **A. Environment** | `nvidia-smi` sees a GPU; `torch.version.cuda`; the card's compute capability is in the image's arch list; `SPARSE_BACKEND` is unset | `UNSUPPORTED` arch → you rented a 4090/L40S/H100; destroy and rent a T4/3090. `SPARSE_BACKEND` set → it silently overrides `model_name`; the script clears it |
| **B. Patch + converter** | fetches `$DATASET_PATCH` → `/opt/prep`; `train.py` present; `Treeins_NUM_CLASSES == 4` and `PointGroup3heads_ts.py` exists; applies `forainet-local.patch` if not | `NUM_CLASSES=5` after applying → wrong ForAINet commit. No `train.py` → the image is env-only; it needs `COPY ForAINet/ /workspace/` (see `container_export/README.md`) |
| **C. Smoke test** | runs [`smoke_test.py`](../container_export/smoke_test.py): LAZ backend present, `semantic_seg`/`treeID` survive a LAZ round trip, `header.parse_crs` works, no Hydra leaked, torch imports | no LAZ backend → `pip install "laspy[lazrs]"` (the `Dockerfile.train` layer should already do this) |
| **D. Data + convert** | fetches `$DATASET_TRAIN_URL` → `/data`; 14 `.laz` present; `pipeline.convert --to ply` into `…/raw/SegmentedForests`; 14 `.ply` out; clears `processed_0.2/` | 0 `.laz` → check `DATASET_TRAIN_URL`. Fewer `.ply` than expected → a conversion error printed above |
| **E. Training startup** | 2-epoch run for **PointGroup-PAPER** (Mink) then **PointGroup-PAPER-TS** (TorchSparse), `wandb.log=False` | see the captured `/tmp/preflight_{mink,ts}.log`; `Key 'area4_ablation_2' not found` means a required override was dropped |
| **F. wandb** (optional) | a 1-epoch `wandb.log=True` run reaches training | key-format `TypeError` → `pip install -U wandb`; `404 entity` → set `training.wandb.entity` to a team you belong to (see [`run_debugger.md`](run_debugger.md)) |

### Success signals in the training log

| Signal | Expected | Meaning if different |
|---|---|---|
| `Model size` | `11872126` | the architecture changed, not just the kernels |
| `iou_per_class` | **4** entries `{0,1,2,3}` | **5** ⇒ the 4-class patch is not applied |
| Minkowski deprecation warning | on run 1, **absent** on run 2 | present on run 2 ⇒ TorchSparse isn't really active (`SPARSE_BACKEND`?) |
| instance metrics | all `0.0` | **correct** — clustering is gated to epoch 31 (`prepare_epoch: 30`) |

Both runs finishing 2 epochs without error is the real result. Two epochs say nothing about
model quality — only that the pipeline is sound.

---

## 4. When it passes

```bash
# destroy the cheap instance the moment it's green — don't pay idle:
# (from the Vast web UI, or `vastai destroy instance <id>`)
```

Then follow [`gpu_training_runbook.md`](gpu_training_runbook.md) **Part D onward** on the
A100: same image, same env vars, `training.epochs=150`, `training.wandb.public=False`, and
`job_name=segforest_mink` / `segforest_ts`. The only differences from this pre-flight are the
epoch count and the card.

---

## Troubleshooting

Everything specific to the container — GPU not visible, `Bus error` / `/dev/shm`, wandb key
and entity failures — is covered in [`run_debugger.md`](run_debugger.md) and
[`gpu_training_runbook.md`](gpu_training_runbook.md) Part G. This doc does not repeat them.

One note on archive formats: `preflight.sh` auto-detects `.zip`, `.tar.gz`, `.tar` and `.7z`
(and sniffs content when the URL has no extension). `.7z` needs `p7zip-full`, which the base
image lacks — prefer `.tar.gz` or `.zip` for `DATASET_TRAIN_URL` / `DATASET_PATCH`, or
`apt-get install -y p7zip-full` first.
