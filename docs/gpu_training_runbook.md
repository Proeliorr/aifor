# Training on a rented GPU — from a local test to two finished models

A step-by-step route: test everything on your own PC first, publish the image, upload the
data, rent an A100, train **two** models (one per backbone), and bring the results home.

Every command says what it does and what you should see. Nothing here assumes you have
done it before.

**The two models you will end up with**

| Model | Backbone | Why |
|---|---|---|
| `segforest_mink` | MinkowskiEngine | the default; the baseline everything else is measured against |
| `segforest_ts` | TorchSparse 1.4 | the newer backend — same network, different kernels |

Same data, same settings, same machine. The only difference is the sparse-convolution
library, so any difference in speed or accuracy is attributable to it.

**Related documents** — this one does not repeat them:

- [`run_debugger.md`](run_debugger.md) — running the container locally, and a large
  troubleshooting section (wandb, GPU visibility, `Bus error`)
- [`backbone_torchsparse_1.4.md`](backbone_torchsparse_1.4.md) — what the two backbones are
- [`eval_process.md`](eval_process.md) — how to evaluate the models afterwards
- [`train_logs_explained.md`](train_logs_explained.md) — how to read the training output

---

## Contents

- [Part 0. The route, and what you need](#part-0-the-route-and-what-you-need)
- [Part A. Test locally first](#part-a-test-locally-first)
- [Part B. Publish the image to Docker Hub](#part-b-publish-the-image-to-docker-hub)
- [Part C. The data on Cloudflare R2](#part-c-the-data-on-cloudflare-r2)
- [Part D. Rent the A100 on Vast.ai](#part-d-rent-the-a100-on-vastai)
- [Part E. Train both models](#part-e-train-both-models)
- [Part F. Download the results](#part-f-download-the-results)
- [Part G. Troubleshooting](#part-g-troubleshooting)
- [Part H. Time and cost](#part-h-time-and-cost)

---

## Part 0. The route, and what you need

```
   YOUR PC (Windows 11)                                    RENTED A100 (Vast.ai)
   ─────────────────────                                   ─────────────────────
   A. local smoke test
      docker run --name forainet-smoketest
      2 short runs, both backbones
              │
      B. push image  ──────────► Docker Hub ─────────────►  docker pull
              │                                                   │
      C. upload data ──────────► Cloudflare R2 ───────────►  wget + unzip
                                                                  │
                                                             convert .laz -> .ply
                                                                  │
                                                        E. train ×150 epochs, twice
                                                                  │
   F. scp results  ◄──────────────────────────────────────────────┘
      (~310 MB per run)
```

### What must be installed on your PC

| Thing | Why | Check it works |
|---|---|---|
| Docker Desktop (WSL2 backend) | runs the container | `docker version` |
| NVIDIA support in Docker | the GPU inside the container | `docker run --rm --gpus all nvidia/cuda:11.1.1-base-ubuntu20.04 nvidia-smi` |
| OpenSSH client | downloading results later | `scp` (built into Windows 11) |
| The `aifor` conda env | the pipeline scripts | `mamba run -n aifor python -c "print(1)"` |

`run_debugger.md` §0 covers the Docker/NVIDIA setup in more detail if any of the above fails.

### Accounts

- **Docker Hub** — free; hosts the image
- **Cloudflare R2** — free tier is 10 GB stored / month, and R2 charges **no egress fee**,
  which is why it suits a 4.3 GB dataset downloaded repeatedly
- **Vast.ai** — pay-as-you-go; add credit before you start
- **Weights & Biases** — free; you need your API key from
  [wandb.ai/authorize](https://wandb.ai/authorize)

### Your GPU situation

Your PC has a **GTX 1660 Ti (6 GB)**. That is a Turing card, compute capability **7.5**.
The image was compiled for `6.0;7.0;7.5;8.0;8.6`, so your card **is** supported — Part A is
a real GPU test, not a pretend one.

The A100 80GB is **8.0**, also supported.

> ⚠️ **Do not rent an RTX 4090, L40S, H100 or anything newer.** Those are compute
> capability 8.9 / 9.0 / 12.0, and the image's CUDA kernels were built without a `+PTX`
> fallback, so there is nothing for them to run. You would pay for an instance that cannot
> execute a single convolution. **A100, A40, A10, RTX 3090, V100, T4 and P100 all work.**

---

## Part A. Test locally first

The whole point: find broken things on hardware you are not paying for by the hour.

Nothing here has been run end to end yet. The training log on record
(`train_logs.md`) is from before the 4-class change, the LAZ export and the second
backbone existed.

### A0. First: ForAINet is patched, and the patch is not stored by git

Upstream ForAINet is a **5-class** model. This project uses **4** — SegmentedForests has
no counterpart to upstream's `branches` class. That change, and three others, live in
[`patches/forainet-local.patch`](../patches/README.md):

| What the patch changes | Why it matters here |
|---|---|
| **5 -> 4 semantic classes** | class tables, `VALID_CLASS_IDS`, `SemIDforInstance` and the `final_eval` counters. Without it the model trains a head for a class no label can ever reach |
| **Adds `PointGroup3heads_ts.py` + the `PointGroup-PAPER-TS` block** | this is what makes `model_name=PointGroup-PAPER-TS` (the TorchSparse run) exist at all |
| **`path_pretrained: null`** | upstream points it at a *5-class* checkpoint on the author's cluster |
| **wandb entity** | `binbin` -> `aifor` |

`ForAINet/` is a **git submodule**, and a submodule stores only a commit SHA — so
`git submodule update` silently reverts every one of those edits. Nothing crashes when
that happens; you simply get a 5-class model and metrics divided by the wrong number.

Check before you do anything else:

```powershell
mamba run -n aifor python misc\check_forainet_classes.py
```

Expect `OK: ForAINet agrees with conf/config.yaml (4 classes, things=[3, 4])`. If it
reports mismatches, reapply the patch:

```powershell
cd ForAINet
git apply ..\patches\forainet-local.patch
cd ..
```

Everything downstream assumes this passed — including "expect four classes" in A6, and the
image you build in Part B, which bakes whatever state the working tree is in.

### A1. Check what is inside the image

```powershell
docker run --rm for-ai-net ls /workspace/PointCloudSegmentation/train.py
```

**What this does:** starts the image, lists one file, exits.

- **File listed** → the image already contains the ForAINet code.
- **"No such file or directory"** → the image is *environment only* (CUDA, torch,
  MinkowskiEngine and nothing else). That is normal and expected. Locally the code comes
  from a folder on your disk (next step); for Vast it gets baked in during Part B.

### A2. Make a small test dataset

The rented GPU will use all 14 plots. Locally you only need enough to prove the code runs,
so convert **three** — one for each split:

```powershell
cd D:\Podyplomowe\04_AI_Intro\assignment
mamba run -n aifor python convert.py convert.plots=[plot_02,plot_11,plot_01]
```

**What this does:** expands three `.laz` files into the `.ply` files ForAINet reads, writing
them into `ForAINet\PointCloudSegmentation\data_set1_5classes\treeinsfused\raw\SegmentedForests\`.

You get `plot_02_train.ply`, `plot_11_val.ply` and `plot_01_test.ply` — about **2.2 GB**
instead of the full 29 GB. ForAINet needs at least one file of each split, and the split is
read from the file name.

### A3. Start a throwaway container

`docker-compose.yml` names its container `ai-debug-session` — that is your **primary**
debug container and this test leaves it completely alone. Start a second, disposable one
instead, with its own name:

```powershell
docker run -d --name forainet-smoketest --gpus all --shm-size=8g -v "D:\Podyplomowe\04_AI_Intro\assignment\ForAINet:/workspace" -w /workspace/PointCloudSegmentation for-ai-net sleep infinity
```

```powershell
docker exec -it forainet-smoketest bash
```

**Reading the flags:**

| Flag | Why |
|---|---|
| `--name forainet-smoketest` | its own name, so it cannot collide with `ai-debug-session` |
| `--gpus all` | hands the GTX 1660 Ti to the container |
| `--shm-size=8g` | DataLoader workers pass batches through `/dev/shm`; Docker's 64 MB default causes `Bus error` |
| `-v "...\ForAINet:/workspace"` | mounts your local code **and** the data you just converted, so nothing needs rebuilding |
| `-w /workspace/PointCloudSegmentation` | the directory `train.py` expects to be launched from |
| `sleep infinity` | keeps the container alive so you can `exec` into it |

You are now at a `#` prompt inside Linux, in `/workspace/PointCloudSegmentation`.

Check the GPU is visible:

```bash
nvidia-smi
```

You should see the GTX 1660 Ti. If not, see `run_debugger.md` -> *GPU not visible*.

### A4. Install the LAZ reader and run the self-check

```bash
python3.8 -m pip install "laspy[lazrs]" "numpy==1.24.4"
python3.8 /workspace/../container_export/smoke_test.py   # if you copied it in
```

**Why:** the image ships `laspy==2.0.3` with **no LAZ backend**, so it physically cannot
open a `.laz` file until `lazrs` is installed. `numpy` is pinned so pip cannot upgrade it
and break torch.

You do not strictly need this for Part A (the data is already `.ply`), but installing it
now proves the command works before you depend on it on the rented machine.

### A5. Two short training runs

One command per line — copy, paste, run:

```bash
cd /workspace/PointCloudSegmentation
```

Run 1, MinkowskiEngine (the default backbone):

```bash
python train.py task=panoptic data=panoptic/treeins_set1 models=panoptic/FORpartseg_3heads model_name=PointGroup-PAPER training=treeins_set1 training.epochs=2 training.wandb.log=False job_name=localtest_mink
```

Run 2, TorchSparse 1.4:

```bash
python train.py task=panoptic data=panoptic/treeins_set1 models=panoptic/FORpartseg_3heads model_name=PointGroup-PAPER-TS training=treeins_set1 training.epochs=2 training.wandb.log=False job_name=localtest_ts
```

**Reading the command:**

| Part | Meaning |
|---|---|
| `task=panoptic` | semantic **and** instance segmentation |
| `data=panoptic/treeins_set1` | which dataset definition to use |
| `models=panoptic/FORpartseg_3heads` | **required** — the file holding the model definitions |
| `model_name=PointGroup-PAPER` / `-TS` | **this is the backbone switch** |
| `training=treeins_set1` | epochs, batch size, optimiser |
| `training.epochs=2` | override: just a quick check |
| `training.wandb.log=False` | no cloud logging for a local test |
| `job_name=...` | names the output folder |

> `models=panoptic/FORpartseg_3heads` is not optional. `conf/config.yaml` defaults to
> `models: panoptic/area4_ablation_2`, **a file that does not exist**, so leaving it out
> fails immediately at startup.

### A6. What "it worked" looks like

| Check | Expected | If it differs |
|---|---|---|
| `Model size` | `11872126` | a very different number means the architecture changed, not just the kernels |
| `iou_per_class` | **four** entries `{0,1,2,3}` | **five** means the 4-class patch is not applied — run `python misc\check_forainet_classes.py` |
| Minkowski deprecation warning | 3× on run 1, **absent** on run 2 | if present on run 2, TorchSparse is not actually being used |
| All instance metrics | `0.0` | **correct** — clustering only starts after epoch 30 (`prepare_epoch`) |
| `train_acc` after 1 epoch | roughly 70-80 | wildly lower suggests a label problem |

Both runs finishing is the real result. Two epochs cannot tell you anything about model
quality.

### A7. Clean up before the real run

```powershell
# from PowerShell, back on the host
Remove-Item -Recurse -Force ForAINet\PointCloudSegmentation\data_set1_5classes\treeinsfused\processed_0.2
docker rm -f forainet-smoketest
```

**Why this matters:** ForAINet caches its preprocessed tensors in `processed_0.2/`. That
cache was built from your **three-plot** test set. If it survives, the real 14-plot run
will silently reuse it and train on three plots while appearing to work. Delete it every
time `raw/` changes.

---

## Part B. Publish the image to Docker Hub

### B1. Back up the base image first: do this before anything else

The `for-ai-net` image **cannot be rebuilt**. Its Dockerfile installs MinkowskiEngine from
an unpinned `git+https://...` and hdbscan from a `master.zip`, so a rebuild today pulls
different code and fails. It exists in exactly one place: your local Docker daemon.

```powershell
docker save for-ai-net | gzip > D:\for-ai-net-base.tar.gz
```

**What this does:** writes the whole image to a single file you can copy elsewhere. Takes a
while (~15 GB). Keep it. If Docker Desktop ever needs reinstalling, this file is the project.

### B2. Push the base

```powershell
docker login -u <your-dockerhub-username>
docker tag for-ai-net <your-dockerhub-username>/for-ai-net:base
docker push <your-dockerhub-username>/for-ai-net:base
```

**What this does:** uploads the image so the rented machine can pull it. Roughly 15 GB —
at 20 Mbit/s upload that is about 1.5 hours. Start it and do something else. The push is
resumable: if it drops, run it again and it continues from the layers already uploaded.

> Use a **personal access token** rather than your password:
> Docker Hub → *Account Settings* → *Personal access tokens* → *Generate*.

### B3. Build the layer with your code in it

The bind mount that works locally cannot work on a rented machine — there is no folder of
yours to mount. So the code goes into the image:

```powershell
cd D:\Podyplomowe\04_AI_Intro\assignment\container_export
docker build -f Dockerfile.train --build-arg BASE=for-ai-net `
  -t <your-dockerhub-username>/for-ai-net:v1 .
docker push <your-dockerhub-username>/for-ai-net:v1
```

**What this does:** starts *from* the base image and adds three things — the LAZ reader,
your conversion code, and a self-test that runs during the build. Because the base layers
are unchanged, only the new ~640 MB uploads.

If the build fails at the smoke-test step, that is the point of it: it means the container
could not read a `.laz` file, and it told you now instead of on a machine costing $1.50/hour.

> **The `Dockerfile` in the repository root is not this file and cannot be used.** It is a
> leftover VSCode template — `FROM python:3-slim`, no CUDA, no PyTorch, and a `CMD` written
> with a Windows backslash that Linux cannot run. Building it produces an image that cannot
> train anything. Always use `container_export/Dockerfile.train`.

### B4. Also copy ForAINet in, if the image lacks it

If step **A1** reported "No such file", add this line to `Dockerfile.train` before building:

```dockerfile
COPY ForAINet/ /workspace/
```

and copy `ForAINet\` (excluding `outputs\` and `data_set1_5classes\`) into
`container_export\` first. That is ~6.3 MB of code — the 627 MB `outputs\` folder and the
data must stay out.

---

## Part C. The data on Cloudflare R2

### C1. What is uploaded, and what it contains

| Object | Size | What it is |
|---|---|---|
| `ForAINet_export.zip` | ~4.3 GB | The 14 training clouds (`.laz`) **and** the 14 `_offsets.yml` files |
| `container_export.7z` | 125 KB | The converter code — only needed if it is not already baked into your image |

**What is inside each `.laz`** — the output of the whole preprocessing pipeline:

- **coordinates already centred** so the minimum is `0,0,0` (ForAINet requires
  non-negative coordinates); the shift subtracted is recorded in the matching `_offsets.yml`
- **`semantic_seg`** — the label, already converted from the SegmentedForests scheme to
  ForAINet's 4 classes (`low_vegetation`, `ground`, `stem_points`, `live_branches`)
- **`treeID`** — the tree each point belongs to, set to `0` on everything that is not part
  of a tree (ForAINet discards instances otherwise)
- **`intensity`** — the raw sensor return strength
- **the train/val/test split, in the file name** — `plot_11_val.laz` is the validation plot,
  `plot_01_test.laz` and `plot_14_test.laz` are the test plots, the other eleven are training

**Why the `_offsets.yml` files matter:** each holds the x/y/z shift that was subtracted, plus
the source scales and coordinate system. Without them the model's predictions can never be
put back on the map. 18 KB in total.

**What is deliberately *not* uploaded:**

| Not uploaded | Size | Why not |
|---|---|---|
| The `.ply` files | 29.2 GB | The instance regenerates them from the `.laz` in about two minutes. Uploading them would mean 29 GB instead of 4.3 GB |
| `processed_0.2/` | ~2 GB | A cache ForAINet builds itself; a stale one causes silent wrong results |
| The original `pointclouds/*.laz` | 4.4 GB | The *raw* survey clouds. Only needed to *produce* the training data, never to train |

### C2. The URLs

Both objects are public, so the instance fetches them with a plain `wget` and no credentials:

```
https://pub-20d48ebca25a482eb313b460427acaa8.r2.dev/ForAINet_export.zip
https://pub-20d48ebca25a482eb313b460427acaa8.r2.dev/container_export.7z
```

> The `r2.dev` development URL is rate-limited by Cloudflare and not meant for heavy
> traffic. Pulling one archive onto one instance a few times is exactly what it is for.

### C3. Re-uploading, if the data ever changes

The R2 dashboard's drag-and-drop uploader **stops at 300 MB**, and the archive is 4.3 GB
(5 of the 14 `.laz` files exceed 300 MB on their own), so it has to go over R2's S3 API.
With the `r2` rclone remote already configured:

```powershell
.\rclone.exe copy D:\ForAINet_export.zip r2:<bucket>/ --progress
```

It uploads in chunks and resumes if interrupted — just run the same command again.

### C4. Check they are reachable

Before renting anything, confirm both objects can be fetched **without credentials**,
exactly as the instance will:

```powershell
curl.exe -I https://pub-20d48ebca25a482eb313b460427acaa8.r2.dev/ForAINet_export.zip
```

**What to look for:** `HTTP/2 200` and a `content-length` around `4300000000`.

| Response | Meaning |
|---|---|
| `200` | Good — the instance will be able to `wget` this |
| `401` / `403` | The Public Development URL is not enabled on the bucket |
| `404` | Wrong object name — check with `.\rclone.exe ls r2:<bucket>/` |

Free to fix now, expensive to discover on a machine billing by the minute.

---

## Part D. Rent the A100 on Vast.ai

### D1. Choosing an instance

| Setting | Value | Why |
|---|---|---|
| GPU | **A100 80GB** (or A40 / A6000) | compute capability 8.0 — supported by the image |
| Disk | **at least 60 GB** | 4.3 GB archive + 29 GB PLY + ~2 GB cache + 2 checkpoints, with room to spare |
| Instance type | **On-demand**, not interruptible | an interruptible instance can be taken away mid-run and you lose ~8 hours |
| Image | `<your-dockerhub-username>/for-ai-net:v1` | what you pushed in Part B |

**Disk is the one people get wrong.** The 4.3 GB you upload becomes 29 GB after conversion.
An instance with 30 GB of disk fails halfway through preprocessing with a confusing error.

### D2. Instance configuration

- **Docker image:** `<your-dockerhub-username>/for-ai-net:v1`
- **Docker login:** enter your Docker Hub username and token even though the image is
  public. Vast machines pull anonymously from shared datacenter IP addresses that are often
  already rate-limited, and the failure looks like `toomanyrequests` *after* billing starts.
- **Environment variables:** `WANDB_API_KEY=<your key from wandb.ai/authorize>`
- **On-start script:** leave empty; you will run the steps by hand the first time so you can
  see each one work.

### D3. Connect

Vast shows an SSH command on the instance card. From PowerShell:

```powershell
ssh -p <port> root@<host>
```

---

## Part E. Train both models

Everything below runs **on the instance**.

### E1. Get the training data

```bash
mkdir -p /data && cd /data
wget https://pub-20d48ebca25a482eb313b460427acaa8.r2.dev/ForAINet_export.zip
unzip -q ForAINet_export.zip
ls *.laz | wc -l          # expect 14
ls *_offsets.yml | wc -l  # expect 14
```

**What this does:** downloads the 4.3 GB archive and unpacks the 28 files. On a datacenter
connection expect one to three minutes. `unzip` is already in the image.

If the files land in a subfolder (`ls` shows a directory instead of `.laz` files), use that
subfolder as the source path in E2.

### E2. Get the converter: only on Path B

There are two ways the converter code reaches the instance, and only one of them needs a
download:

| Path | How the code gets there | Need this step? |
|---|---|---|
| **A — baked into the image** | you ran `docker build -f Dockerfile.train` in Part B | **No.** It is already at `/opt/prep` |
| **B — fetched at runtime** | you are running the plain base image | **Yes** |

Path B skips the ~15 GB image push entirely, which is why it is worth having.

```bash
mkdir -p /opt/prep && cd /opt/prep
wget https://pub-20d48ebca25a482eb313b460427acaa8.r2.dev/container_export.tar.gz
tar -xzf container_export.tar.gz
export PYTHONPATH=/opt/prep
```

`tar` and `gzip` are in every Linux image, so this needs no installation.

> **If you only uploaded the `.7z`**, the base image has `wget`, `unzip` and `zip` but
> **no 7-Zip**, so it needs one extra line first:
>
> ```bash
> apt-get update && apt-get install -y p7zip-full
> 7z x container_export.7z -o/opt/prep
> ```
>
> Repacking as `.tar.gz` avoids that. From your PC:
> `tar -czf container_export.tar.gz -C container_export .` then upload it alongside the
> other object.

**What is in the archive**, and what each part is for:

| Inside | Used where | Purpose |
|---|---|---|
| `pipeline/` | **on the instance** | the LAZ -> PLY converter you run in E3 |
| `patches/forainet-local.patch` | on the instance, only if the image is unpatched | restores the 4-class scheme and the TorchSparse model — see [A0](#a0-first-forainet-is-patched-and-the-patch-is-not-stored-by-git) |
| `misc/Points2ForAINet.py` | on the instance | the same converter under its historic CLI name |
| `smoke_test.py` | on your PC, during the image build | checks the container can actually open a `.laz` |
| `Dockerfile.train` | **on your PC only** | the recipe that builds the image — see below |
| `.dockerignore`, `README.md` | on your PC | build hygiene and notes |

> **What `Dockerfile.train` is.** It is the recipe for building the training image: it
> starts `FROM for-ai-net` and adds the LAZ backend (`laspy[lazrs]`), the converter at
> `/opt/prep`, and a build-time smoke test. It runs **on your PC**, at `docker build` time.
> It is never executed inside the container and the instance has no use for it — it is in
> the archive only because the archive is a copy of the whole `container_export/` folder.
> At 2 KB it is not worth excluding.

### E3. Confirm the image really is patched

Before spending eight hours, check that the ForAINet code in this container is the 4-class
version. An unpatched image trains happily and gives you wrong numbers.

```bash
grep -n "Treeins_NUM_CLASSES = " /workspace/PointCloudSegmentation/torch_points3d/datasets/segmentation/treeins_set1.py
ls /workspace/PointCloudSegmentation/torch_points3d/models/panoptic/PointGroup3heads_ts.py
```

Expect `Treeins_NUM_CLASSES = 4` and the `_ts.py` file to exist. If you get `5`, or the
file is missing, the image was built from an unpatched tree — apply the patch now:

```bash
cd /workspace && git apply /opt/prep/patches/forainet-local.patch
```

(That needs Path B's archive, or the patch copied in some other way. It is much easier to
rebuild the image from a patched tree than to repair it here.)

### E4. Turn the `.laz` back into `.ply`

```bash
python3.8 -m pipeline.convert --to ply /data \
    /workspace/PointCloudSegmentation/data_set1_5classes/treeinsfused/raw/SegmentedForests
```

**What this does:** expands each compressed `.laz` into the uncompressed `.ply` format
ForAINet reads — 4.3 GB becomes about 29 GB. Takes a couple of minutes.

This is a **pure format conversion**: no coordinates move, no labels change. The file name
is preserved exactly, because ForAINet reads the train/val/test split from it.

That destination path is not arbitrary. ForAINet builds it itself from
`dataroot: data_set1_5classes` in its data config, resolved relative to the directory you
launch `train.py` from, then appends `treeinsfused/raw/` and searches it recursively.

Check:

```bash
ls /workspace/PointCloudSegmentation/data_set1_5classes/treeinsfused/raw/SegmentedForests/*.ply | wc -l
# expect 14
```

### E5. Clear the way

```bash
rm -rf /workspace/PointCloudSegmentation/data_set1_5classes/treeinsfused/processed_0.2
unset SPARSE_BACKEND
nvidia-smi
```

**Line by line:**

- `rm -rf ... processed_0.2` — deletes any cached preprocessed tensors so they are rebuilt
  from your 14 plots. A stale cache does not error; it silently trains on the wrong data.
- `unset SPARSE_BACKEND` — this environment variable, if set, **silently overrides** the
  backbone choice in the config. If a Vast template happens to set it, your TorchSparse run
  would quietly use MinkowskiEngine and you would compare a model against itself.
- `nvidia-smi` — confirms the A100 is visible and shows 80 GB free.

### E6. Run 1: MinkowskiEngine

```bash
cd /workspace/PointCloudSegmentation
tmux new -s mink

python train.py task=panoptic data=panoptic/treeins_set1 models=panoptic/FORpartseg_3heads model_name=PointGroup-PAPER training=treeins_set1 training.epochs=150 training.wandb.public=False job_name=segforest_mink
```

**Why `tmux`:** the job runs for 8-10 hours. Without it, closing your laptop or losing
Wi-Fi kills the training. Press `Ctrl+B` then `D` to detach; `tmux attach -t mink` to come
back.

**Why `training.wandb.public=False`:** this does **not** disable wandb — your metrics still
stream to the dashboard live. What it switches off is a per-epoch copy of the 308 MB
checkpoint into the wandb folder, which wandb then uploads. Over 150 epochs that is roughly
**46 GB of uploads per run**, which slows every epoch and would consume most of a free wandb
storage quota across two runs. You collect the checkpoint yourself in Part F.

**First-run note:** before epoch 1 the framework preprocesses all 14 plots into
`processed_0.2/`. This takes a while and prints little. That is normal, once.

### E7. Run 2: TorchSparse 1.4

When run 1 finishes:

```bash
tmux new -s ts

python train.py task=panoptic data=panoptic/treeins_set1 models=panoptic/FORpartseg_3heads model_name=PointGroup-PAPER-TS training=treeins_set1 training.epochs=150 training.wandb.public=False job_name=segforest_ts
```

**The only change is `model_name`.** Same data, same hyper-parameters, same machine — so
any difference between the two runs comes from the backbone.

**Do not delete `processed_0.2/` between the two runs.** Preprocessing does not depend on
the backbone, so run 2 reuses the cache and starts training immediately.

You can also run both at once — an 80 GB A100 has room for two of these models — but the
timings then are not comparable, because they compete for the same GPU.

### E8. Watching progress

```bash
tail -f outputs/segforest_mink/*/train.log
```

Or in the wandb dashboard, where both runs appear side by side.

Sanity checks in the first few minutes: `Model size = 11872126`, and `iou_per_class` with
**four** entries. All instance metrics stay `0.0` until **epoch 31** — clustering is gated
behind `prepare_epoch: 30`. That is correct behaviour, not a failure.

---

## Part F. Download the results

> **The instance's disk is deleted when you terminate it.** Nothing is recoverable
> afterwards. Download before you stop paying.

### F1. What to take

Each run produces a folder at
`outputs/<job_name>/<job_name>-<model_name>-<timestamp>/`:

| File | Size | Take it? |
|---|---|---|
| `PointGroup-PAPER.pt` | **308 MB** | **Yes — this is the trained model** |
| `train.log` | ~90 KB | Yes — the full metric history |
| `.hydra/` | a few KB | Yes — the exact configuration used |
| `change.patch` | a few KB | Yes — the state of the code that produced it |
| `wandb/` | 308 MB | **No** — it holds a second copy of the same checkpoint |
| `data_set1_5classes/` | 33 GB | **No** — regenerated from the 4.3 GB archive any time |

So about **310 MB per run**, 620 MB for both — even though the folders report ~620 MB each.

### F2. Method 1: `scp` (the normal way)

Windows 11 includes an SSH client, so this works in PowerShell with nothing to install.
Use the same host and port as your `ssh` command.

Find the exact folder name first, on the instance:

```bash
ls -d /workspace/PointCloudSegmentation/outputs/segforest_mink/*/
# e.g. .../outputs/segforest_mink/segforest_mink-PointGroup-PAPER-20260815_101530/
```

Then, from PowerShell **on your PC**:

```powershell
mkdir D:\Podyplomowe\04_AI_Intro\assignment\results

# the model
scp -P <port> root@<host>:/workspace/PointCloudSegmentation/outputs/segforest_mink/*/PointGroup-PAPER.pt `
    D:\Podyplomowe\04_AI_Intro\assignment\results\segforest_mink.pt

# the log and config
scp -P <port> root@<host>:/workspace/PointCloudSegmentation/outputs/segforest_mink/*/train.log `
    D:\Podyplomowe\04_AI_Intro\assignment\results\segforest_mink.log
scp -P <port> -r root@<host>:/workspace/PointCloudSegmentation/outputs/segforest_mink/*/.hydra `
    D:\Podyplomowe\04_AI_Intro\assignment\results\segforest_mink_hydra
```

Repeat with `segforest_ts` and `PointGroup3heads_ts`'s checkpoint (also named
`PointGroup-PAPER.pt`, since the checkpoint is named after `model_name`'s block — rename it
on arrival so the two do not collide).

`-P` is the port (capital P for `scp`; lowercase for `ssh`). At ~10 MB/s a 308 MB
checkpoint takes about half a minute.

**Copying the whole folder in one go** is simpler but transfers the 308 MB duplicate too:

```powershell
scp -P <port> -r root@<host>:/workspace/PointCloudSegmentation/outputs/segforest_mink `
    D:\Podyplomowe\04_AI_Intro\assignment\results\
```

Delete the `wandb\` subfolder afterwards.

### F3. Method 2: push to R2, download later

Better if the run finishes overnight: the results outlive the instance.

On the instance:

```bash
cd /workspace/PointCloudSegmentation/outputs
tar -cf /tmp/results.tar --exclude='wandb' segforest_mink segforest_ts
rclone copy /tmp/results.tar r2:forainet-data/ --progress
```

(or use Vast's built-in **Cloud Copy** button on the instance page, which targets
S3-compatible storage directly). Then download from R2 whenever you like — R2 charges no
egress fee.

This is the safer option: you can terminate the instance the moment the transfer completes,
rather than paying while a slow download runs.

### F4. Method 3: the Jupyter file browser

If your instance exposes Jupyter, its file browser can download files by clicking them.
Fine for `train.log` and the configs, workable for a 308 MB checkpoint, clumsy beyond that.

### F5. A habit worth having

The checkpoint is already useful after epoch ~40. Copy it once mid-run rather than betting
eight hours of GPU time on one transfer at the end:

```powershell
scp -P <port> root@<host>:/workspace/PointCloudSegmentation/outputs/segforest_mink/*/PointGroup-PAPER.pt `
    D:\...\results\segforest_mink_partial.pt
```

### F6. Back on your PC

Put predictions back into real-world coordinates using the offsets files:

```powershell
mamba run -n aifor python forainet_prep.py `
  forainet_prep.mode=restore `
  forainet_prep.restore_dir=D:\Podyplomowe\04_AI_Intro\assignment\results\predictions
```

See [`eval_process.md`](eval_process.md) for evaluating the two models and what each metric
means.

---

## Part G. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `toomanyrequests` when the instance pulls the image | anonymous Docker Hub pull limit on a shared datacenter IP | put your Docker Hub username + token in Vast's registry-login fields |
| `No space left on device` during preprocessing | instance disk under 60 GB | you cannot resize; destroy and rent one with more disk |
| `no kernel image is available for execution` | the GPU is newer than the image supports | you rented a 4090/L40S/H100. Use an A100/A40/A10/3090/V100/T4 |
| `iou_per_class` shows **five** classes | the local patch was reverted | `cd ForAINet && git apply ../patches/forainet-local.patch`, then `python misc\check_forainet_classes.py` |
| `Key 'area4_ablation_2' not found` at startup | `models=` was omitted | add `models=panoptic/FORpartseg_3heads` |
| The TorchSparse run prints the Minkowski deprecation warning | `SPARSE_BACKEND` is set in the environment | `unset SPARSE_BACKEND` and rerun |
| Metrics look wrong / suspiciously good | a stale `processed_0.2/` from a smaller dataset | delete it and let ForAINet rebuild |
| R2 dashboard refuses the upload: *"Files larger than 300 MB..."* | the dashboard uploader caps at 300 MB; the archive is 4.3 GB | upload over R2's S3 API with rclone — see C3 |
| `7z: command not found` on the instance | the base image has `wget`/`unzip`/`zip` but no 7-Zip | `apt-get update && apt-get install -y p7zip-full`, or skip it — the converter is already in the image if you built `Dockerfile.train` |
| `laspy` cannot open a `.laz` | no LAZ backend in the image | `pip install "laspy[lazrs]"` |
| Training stops when you close your laptop | not running under `tmux` | `tmux new -s <name>`, detach with `Ctrl+B` then `D` |

For **wandb API key problems**, **wandb `404 entity not found`**, **GPU not visible**, and
**`DataLoader worker killed by signal: Bus error`**, see the troubleshooting section of
[`run_debugger.md`](run_debugger.md) — all four are covered there in detail.

---

## Part H. Time and cost

Based on your recorded local run (~17 minutes/epoch on the GTX 1660 Ti) and an A100 80GB
being roughly 4-6× faster on this workload:

| Step | Time | Cost |
|---|---|---|
| Local smoke test (Part A) | ~1 h | free |
| Push the base image | 1-2 h upload | free |
| Upload 4.3 GB to R2 | 20-60 min | free tier |
| Instance setup + download + convert | ~15 min | ~$0.30 |
| **Run 1 — MinkowskiEngine, 150 epochs** | **~8-10 h** | ~$8-15 |
| **Run 2 — TorchSparse, 150 epochs** | **~8-10 h** | ~$8-15 |
| Download results | ~5 min | free |
| **Total on the rented GPU** | **~17-21 h** | **~$16-30** |

Treat the per-epoch figure as an estimate until you have watched the first two epochs — then
multiply and you will know what the run will actually cost before committing to it.

**Two things that waste money more than anything else:** renting a GPU the image cannot use,
and discovering a broken data path after the instance is already running. Part A exists to
prevent both.
