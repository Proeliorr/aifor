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
| An image with the environment | CUDA 11.1, torch 1.9, MinkowskiEngine, TorchSparse | the `for-ai-net` image you pushed in `gpu_training_runbook.md` Part B — **or** any Vast template built from ForAINet's own Dockerfile |
| `DATASET_TRAIN_URL` env var | archive of the 14 `.laz` + `_offsets.yml` (the modified SegmentedForest dataset) | set on the Vast instance at creation; downloaded **once** and then checksum-verified (§2c) |
| `DATASET_PATCH` env var | archive with `patches/forainet-local.patch` **and** the converter (`pipeline/`, `smoke_test.py`) | set on the Vast instance at creation |
| `WANDB_API_KEY` env var | only needed for the optional step F | set on the Vast instance at creation |
| The ForAINet **source** | `train.py` and `torch_points3d/` — what actually trains | in the image if you baked it in; otherwise **the script clones it** at the pinned commit `5fe600a` |

The pre-flight reads those env vars, so nothing is hardcoded and the same image works for the
cheap test and the A100 run.

> **The source is not a prerequisite.** Most environment images — including the ForAINet
> template on Vast — carry the conda/CUDA stack and no code: locally it arrived through the
> `./ForAINet:/workspace` bind mount, which cannot exist on a rented host. That used to be a
> dead end (`no train.py at /workspace/PointCloudSegmentation`). It no longer is; see §2b.

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

- **Re-pack `container_export/` and point `DATASET_PATCH` at it.** `preflight.sh` lives in
  `container_export/`, so a freshly regenerated `container_export.tar.gz`
  (`tar -czf container_export.tar.gz -C container_export .`) contains it. Then on the instance:
  ```bash
  mkdir -p /opt/prep && cd /opt/prep && rm -rf ./*     # no leftovers from an older archive
  wget -O c.tgz "$DATASET_PATCH" && tar -xzf c.tgz
  bash "$(find /opt/prep -maxdepth 2 -name preflight.sh | head -n1)"
  ```
  Both layouts work: an archive packed as `tar -czf … container_export` nests everything one
  level deeper (`/opt/prep/container_export/…`), and the script *searches* for
  `*/pipeline/convert.py` rather than assuming. Only the path to `preflight.sh` itself still
  depends on how it was packed — hence the `find` above.
- **Or `scp` it up directly** (it's a single ~25 KB file) and run it:
  ```bash
  scp -P <port> container_export/preflight.sh root@<host>:/root/
  ssh -p <port> root@<host> 'bash /root/preflight.sh'   # full gate, all 14 plots, both backbones
  ```
  This is the fastest way to test a *changed* `preflight.sh` without re-uploading the archive
  to R2 — the script fetches `$DATASET_PATCH` itself for everything else it needs.

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

# Point it at a ForAINet tree you already have, instead of letting it look/clone:
PREFLIGHT_FORAINET_DIR=/workspace bash preflight.sh

# Re-pull the dataset even though /data already holds a verified copy (see 2c):
PREFLIGHT_FORCE_FETCH=1 bash preflight.sh

# Install a different laspy than the pinned 2.5.3 (drop the ==pin to stop version-checking):
PREFLIGHT_LASPY_SPEC="laspy[lazrs]==2.5.4" bash preflight.sh
```

Exit code `0` and a green **PRE-FLIGHT PASSED** = cleared to rent the A100.

---

## 2b. What the script repairs by itself

A rented box is disposable, so the gate fixes what it can instead of handing you a list of
manual steps. All five repairs are on by default and each has an off switch:

| It finds | It does | Turn off with |
|---|---|---|
| the `DATASET_PATCH` payload one level deep (`/opt/prep/container_export/…`) | searches for `*/pipeline/convert.py` and uses whatever directory holds it — for `PYTHONPATH`, the patch and the smoke test | — |
| no `train.py` anywhere | clones `prs-eth/ForAINet` at the pinned **`5fe600a`** (shallow fetch of that exact commit, ~16 MB, seconds) into `/workspace/ForAINet`, or `/workspace` itself if that is empty | `PREFLIGHT_NO_CLONE=1`, or point `PREFLIGHT_FORAINET_DIR` at a tree you already have |
| an unpatched tree (`Treeins_NUM_CLASSES = 5`, no `PointGroup3heads_ts.py`) | applies `patches/forainet-local.patch` from the archive and re-checks; if the patch only *reverses* cleanly, the tree already carries it and is left alone | — (an unpatched tree is never acceptable) |
| `import debugpy` in `train.py`, with no debugpy installed | deletes that one line — it is a local-debugging leftover the patch carries, and nothing attaches a debugger to a rented GPU | install `debugpy` and it is kept |
| laspy that is not the pinned **2.5.3**, has no LAZ backend, or is too old for `header.parse_crs` | `pip install "laspy[lazrs]==2.5.3"` (unpinned retry if that version cannot be satisfied), then re-checks that numpy/torch still import | `PREFLIGHT_NO_PIP=1`, or a different `PREFLIGHT_LASPY_SPEC` |

**Where things end up matters.** After a clone, the training root is
`/workspace/ForAINet/PointCloudSegmentation`, not `/workspace/PointCloudSegmentation`, so the
raw directory the PLYs go to moves with it. The script re-derives every downstream path from
the tree it found and prints all three at the end:

```
ForAINet : /workspace/ForAINet/PointCloudSegmentation
raw dir  : /workspace/ForAINet/PointCloudSegmentation/data_set1_5classes/treeinsfused/raw/SegmentedForests
converter: /opt/prep/container_export
```

Use those paths — not the ones in the runbook — for anything you do by hand afterwards.

> A cloned tree is the same code the local image bind-mounted: same commit, same patch. What
> it is **not** is a substitute for baking ForAINet into the image if you would rather not
> depend on GitHub at run time (`gpu_training_runbook.md` B4).

---

## 2c. The dataset is fetched once, then checksummed

`$DATASET_TRAIN_URL` is 4.3 GB. Re-running the gate — which you will, that is the point of a
cheap box — used to re-download all of it. It no longer does.

After a successful extraction, step D writes two files **inside `/data`**:

| File | Holds |
|---|---|
| `.preflight_manifest.sha256` | a sha256 for every extracted file, paths relative to `/data` |
| `.preflight_source` | the URL those files came from |

On the next run it re-verifies that manifest and skips the download when everything matches:

```
.. DATASET_TRAIN_URL: verifying 28 cached file(s) in /data against .preflight_manifest.sha256
[OK]   DATASET_TRAIN_URL: 28 file(s) already present and checksum-verified — download skipped
```

Any of these re-downloads instead: a **different URL** in `.preflight_source`, a **missing**
file, a **single bad checksum** (a truncated `.laz` from an interrupted extraction has a
plausible size — only a checksum catches it), or `PREFLIGHT_FORCE_FETCH=1`.

If `/data` holds clouds but **no manifest** — a hand-staged copy, or the first run after this
feature landed — they are adopted once with a `[WARN]` (there is nothing to compare them
against yet) and the manifest is written, so every later run is verified.
`PREFLIGHT_ADOPT_EXISTING=0` turns that off and always re-downloads.

Verification costs a sha256 pass over 4.3 GB (tens of seconds); the download it replaces
costs minutes and bandwidth. This only pays off when `/data` survives, so it is most useful
with a **mounted volume**, or when re-running on the same instance.

> **`$DATASET_PATCH` is deliberately not cached.** It is ~50 KB, and it is the archive you
> iterate on — a stale copy of the converter and the patch would defeat the point of
> re-running the gate.

---

## 3. What each step checks (and how to read a failure)

| Step | Checks | If it fails |
|---|---|---|
| **A. Environment** | `nvidia-smi` sees a GPU; `torch.version.cuda`; the card's compute capability is in the image's arch list; `SPARSE_BACKEND` is unset | `UNSUPPORTED` arch → you rented a 4090/L40S/H100; destroy and rent a T4/3090. `SPARSE_BACKEND` set → it silently overrides `model_name`; the script clears it |
| **B. Converter + tree + patch** | fetches `$DATASET_PATCH` → `/opt/prep` and locates the converter in it; finds or clones the ForAINet tree; `Treeins_NUM_CLASSES == 4` and `PointGroup3heads_ts.py` exist, applying `forainet-local.patch` if not; strips the `debugpy` import | `git apply --check failed` → the tree is not at `5fe600a` (the line prints the commit it *is* at). `no git to clone it` → `apt-get install -y git`. `NUM_CLASSES=5` after applying → wrong commit again |
| **B2. laspy** | the installed laspy is the pinned **2.5.3**, a LAZ backend is available and `header.parse_crs` exists — installing `laspy[lazrs]==2.5.3` if any of the three is missing; numpy/torch still import afterwards | `torch no longer imports` → the pip step moved numpy; reinstall the image's numpy (`Dockerfile.train` pins `1.24.4` for the known base). A wrong *version* is only a WARN — capabilities are what the gate is about |
| **C. Smoke test** | runs [`smoke_test.py`](../container_export/smoke_test.py): LAZ backend present, `semantic_seg`/`treeID` survive a LAZ round trip, `header.parse_crs` works, no Hydra leaked, torch imports | a failure *only* on `parse_crs` is downgraded to a WARN (see Troubleshooting); anything else is real |
| **D. Data + convert** | fetches `$DATASET_TRAIN_URL` → `/data` *unless a checksum-verified copy is already there* (§2c); 14 `.laz` present; `pipeline.convert --to ply` into the resolved `…/raw/SegmentedForests`; 14 `.ply` out; clears `processed_0.2/` | 0 `.laz` → check `DATASET_TRAIN_URL`. Fewer `.ply` than expected → a conversion error printed above |
| **E. Training startup** | 2-epoch run for **PointGroup-PAPER** (Mink) then **PointGroup-PAPER-TS** (TorchSparse), `wandb.log=False`. A run counts as passed only if the log contains the trainer's own `EPOCH n / m` line | see the captured `/tmp/preflight_{mink,ts}.log`; `Key 'area4_ablation_2' not found` means a required override was dropped |
| **F. wandb** (optional) | a 1-epoch `wandb.log=True` run reaches training | key-format `TypeError` → `pip install -U wandb`; `404 entity` → set `training.wandb.entity` to a team you belong to (see [`run_debugger.md`](run_debugger.md)) |

### Success signals in the training log

| Signal | Expected | Meaning if different |
|---|---|---|
| `Model size` | `11872109` | the architecture changed, not just the kernels. `11872126` (+17) ⇒ the old 5-class head, patch not applied |
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

A cheap way to reuse everything you just proved: run `PREFLIGHT_SKIP_TRAIN=1 bash preflight.sh`
on the A100 first. It repeats the whole setup — clone, patch, converter, data, cache clear —
in a couple of minutes, and leaves you at the same known-good state, at which point Part E's
`train.py` commands are the only thing left to type. Use the paths the summary prints; after a
clone they sit one directory deeper than the runbook's.

---

## Troubleshooting

Everything specific to the container — GPU not visible, `Bus error` / `/dev/shm`, wandb key
and entity failures — is covered in [`run_debugger.md`](run_debugger.md) and
[`gpu_training_runbook.md`](gpu_training_runbook.md) Part G. This doc does not repeat them.

### Green summary, but training never ran (fixed — check your old logs)

Until 2026-08-27 the script reported step E as passed whenever `train.py … | tee log` exited
0 **from `tee`**: the helper that runs pipelines started a fresh `bash -c`, which does not
inherit `set -o pipefail`. On the first 3090 run that turned

```
python3.8: can't open file 'train.py': [Errno 2] No such file or directory
```

into `[OK] MinkowskiEngine run finished 2 epoch(s)` — twice. If you have a transcript from
before that date, scroll *above* the summary: a real run prints hundreds of lines and an
`EPOCH 1 / 2`. The script now uses `bash -o pipefail` **and** requires that `EPOCH n / m`
line before it will call a run passed.

### `could not parse CRS (No attribute parse_crs in LasHeader)` on every plot

Benign. It means laspy in this image predates `header.parse_crs`, so the converter stores a
null CRS. The exported clouds have **no CRS to lose** (`SegmentedForests/pointclouds/` is in
local coordinates), and training never reads one. Step B2 upgrades laspy anyway so the
message disappears; with `PREFLIGHT_NO_PIP=1` it stays, and step C's `parse_crs` assertion is
downgraded to a WARN rather than failing the gate.

### `[WARN] 0 _offsets.yml (14 expected)`

Also benign **for training**. The offsets are what `forainet_prep mode=restore` needs to put
predictions back on the original coordinates, long after the GPU is destroyed — they are
18 KB, so add them to the dataset archive, but do not re-rent anything over this warning.

### `no ForAINet on this box and no git to clone it`

The image has neither the source nor `git`. `apt-get update && apt-get install -y git` and
re-run, or bake ForAINet into the image (`gpu_training_runbook.md` B4).

One note on archive formats: `preflight.sh` auto-detects `.zip`, `.tar.gz`, `.tar` and `.7z`
(and sniffs content when the URL has no extension). `.7z` needs `p7zip-full`, which the base
image lacks — prefer `.tar.gz` or `.zip` for `DATASET_TRAIN_URL` / `DATASET_PATCH`, or
`apt-get install -y p7zip-full` first.
