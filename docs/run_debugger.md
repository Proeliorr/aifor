# Remote-debugging ForAINet `train.py` in Docker from VSCode

This runbook starts the `for-ai-net` container with GPU + an interactive terminal,
launches [debugpy](https://github.com/microsoft/debugpy) inside it, and attaches the
VSCode debugger over port `5678`.

The container is defined once in [`docker-compose.yml`](docker-compose.yml) (at the repo
root) instead of a long `docker run` one-liner, so the mount and port are always correct.

---

## How the pieces line up

```
Host (this repo)                     Container (for-ai-net)
--------------------------------     ------------------------------------
assignment/                          /
  ForAINet/            <── mount ──> /workspace
    PointCloudSegmentation/          /workspace/PointCloudSegmentation  (working dir)
      train.py                       /workspace/PointCloudSegmentation/train.py
  docker-compose.yml
  .vscode/launch.json
```

The mount (`./ForAINet:/workspace`) and the VSCode `pathMappings` **must agree**, or
breakpoints silently fail to bind. Current [`.vscode/launch.json`](.vscode/launch.json):

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: Remote Attach (for-ai-net-dev)",
      "type": "debugpy",
      "request": "attach",
      "connect": { "host": "localhost", "port": 5678 },
      "pathMappings": [
        { "localRoot": "${workspaceFolder}/ForAINet", "remoteRoot": "/workspace" }
      ],
      "justMyCode": false
    }
  ]
}
```

`${workspaceFolder}` is `assignment/`, so `${workspaceFolder}/ForAINet` == `./ForAINet`
in the compose file. Don't change one without changing the other.

---

## 0. Prerequisites (once)

- The **`for-ai-net`** image exists locally (it is not rebuilt here):

  ```bash
  docker image ls for-ai-net
  ```

- The **NVIDIA Container Toolkit** is installed on the host (needed for `--gpus` / the
  compose `deploy.devices` block). Quick host check:

  ```bash
  nvidia-smi
  ```

- VSCode has the **Python** extension, and `.vscode/launch.json` matches the block above.

Run every `docker compose` command below **from the repo root** (`assignment/`, the
folder containing `docker-compose.yml`).

---

## 1. Start the container

```bash
docker compose up -d
```

Verify it is up and can see the GPU:

```bash
docker compose ps
docker compose exec for-ai-net nvidia-smi
```

If `nvidia-smi` fails with a GPU/driver error, see
[Troubleshooting → GPU not visible](#gpu-not-visible).

---

## 2. Open an interactive shell in the container

```bash
docker compose exec for-ai-net bash
```

You should land in `/workspace/PointCloudSegmentation` (the `working_dir`). Everything in
the next steps runs **inside this shell** unless noted otherwise.

---

## 3. Install extras not baked into the image

These live in [`ForAINet/PointCloudSegmentation/post_installation_commands.sh`](ForAINet/PointCloudSegmentation/post_installation_commands.sh),
so the whole set-up is one command (re-runnable on every fresh container). From the
container shell (already in `/workspace/PointCloudSegmentation`):

```bash
bash post_installation_commands.sh
```

That script currently runs `pip install debugpy` (already in the image — a no-op) and
`apt update && apt install -y mc` (`mc` is not in the image, and the apt cache was
cleared so `apt update` is required first). Extend it as you discover more — see
[§6 Adding your own fix commands](#6-adding-your-own-fix-commands).

---

## 4. Launch debugpy (waits for VSCode to attach)

Still inside the container shell, from `/workspace/PointCloudSegmentation`. Run it as a
**single line** — see the warning below about line continuations:

```bash
python -m debugpy --listen 0.0.0.0:5678 --wait-for-client train.py task=panoptic data=panoptic/treeins_set1 models=panoptic/FORpartseg_3heads model_name=PointGroup-PAPER training=default job_name=debug
```

> **Do not split this across lines with trailing `\`.** A `\` followed by a stray space
> (`\ `) is *not* a line continuation in bash — it becomes a literal-space argument that
> debugpy takes as the script name, giving
> `FileNotFoundError: ... '/workspace/PointCloudSegmentation/ '`. Keep it on one line.

> **Debugging? Disable wandb.** Append `training.wandb.log=False` to skip the
> Weights & Biases login prompt (see
> [Troubleshooting → wandb asks for an API key](#wandb-asks-for-an-api-key)):
> ```bash
> python -m debugpy --listen 0.0.0.0:5678 --wait-for-client train.py task=panoptic data=panoptic/treeins_set1 models=panoptic/FORpartseg_3heads model_name=PointGroup-PAPER training=treeins_set1 job_name=debug training.wandb.log=True
> ```

- `--listen 0.0.0.0:5678` — listen on all interfaces so the host (and VSCode) can reach
  it through the published port.
- `--wait-for-client` — **pauses before running a single line** until VSCode attaches, so
  you can catch errors that happen at startup. The command appears to "hang" — that is
  expected; it is waiting for step 5.
- The `train.py ...` arguments are the paper's **"basic setting"** run (from
  [`ForAINet/README.md`](ForAINet/README.md)). Swap them for a different experiment as
  needed; change `job_name=debug` to taste.

---

## 5. Attach from VSCode

1. Open the **Run and Debug** panel (`Ctrl+Shift+D`).
2. Select **"Python: Remote Attach (for-ai-net-dev)"** and press **F5**.
3. Set a breakpoint in [`ForAINet/PointCloudSegmentation/train.py`](ForAINet/PointCloudSegmentation/train.py)
   **before or right after attaching** to confirm the path mapping works — execution
   should stop there.

Once attached, the `--wait-for-client` guard releases and the script runs under the
debugger.

> **Restart the debug session after any `launch.json` change** — path mappings are read
> only at attach time, so edits don't apply to an already-running session.

---

## 6. Adding your own "fix these errors" commands

While debugging you'll often need extra setup (patch a config, install a missing wheel,
export an env var, download data). The home for these is
[`ForAINet/PointCloudSegmentation/post_installation_commands.sh`](ForAINet/PointCloudSegmentation/post_installation_commands.sh)
— the same script from §3. Edit it on the host (it's under the mounted `./ForAINet`
folder, so changes show up in the container immediately) and add your commands to the
**"Other fixes"** section:

```bash
# --- 3. Other fixes (add as you discover what's needed) ---
# pip install some-missing-package==1.2.3
# export CUDA_VISIBLE_DEVICES=0
# sed -i 's/old_value/new_value/' conf/config.yaml
```

Then re-run it inside the container:

```bash
bash post_installation_commands.sh
```

Because a container's installed packages are **not** persisted after `docker compose
down` (only the mounted `./ForAINet` folder is), keeping every fix in this script means
you can re-create the environment in one command on the next session.

> **Env vars:** `bash post_installation_commands.sh` runs in a child shell, so `export`
> lines inside it don't reach your interactive shell. If the training run needs an env
> var, either `source post_installation_commands.sh` or export it in your shell before
> launching debugpy (§4). The script has a note about this at the bottom.

---

## 7. Teardown

From the repo root:

```bash
docker compose down
```

This stops and removes the `ai-debug-session` container. Files under `./ForAINet` on the
host are untouched.

---

## Troubleshooting

### GPU not visible
`nvidia-smi` fails inside the container, or CUDA reports no device:
- Confirm the host has the NVIDIA Container Toolkit and `nvidia-smi` works on the host.
- Older Docker/Compose ignore the `deploy.devices` block. Edit
  [`docker-compose.yml`](docker-compose.yml): comment out the whole `deploy:` block and
  uncomment `runtime: nvidia`, then `docker compose up -d` again.

### Breakpoints don't hit / "unable to find translation for path"
The mount and `pathMappings` are out of sync. They must resolve to the same files:
`./ForAINet` (compose) ↔ `/workspace` (container) ↔ `${workspaceFolder}/ForAINet`
(launch.json `localRoot`). Fix whichever is wrong and **restart the debug session**.

### "port is already allocated" on `docker compose up`
Something else holds host port `5678`. Stop it, or change **both** the host side of the
port mapping in `docker-compose.yml` (e.g. `"5679:5678"`) and the `connect.port` in
`launch.json` to match.

### debugpy seems to hang
That's `--wait-for-client` doing its job — it blocks until VSCode attaches (§5). Drop
`--wait-for-client` if you want the script to start immediately and attach later.

### `FileNotFoundError: ... '/workspace/PointCloudSegmentation/ '` (path ends in a space)
debugpy received a lone space as the script name instead of `train.py`. This happens
when the launch command is split across lines and a `\` is followed by a trailing space
(`\ ` is a literal-space argument in bash, not a line continuation). Run the debugpy
command from §4 as a **single line**.

### wandb asks for an API key
`wandb: Paste an API key ...`, or `ValueError: API key must be 40 characters long`.
Training reached `Trainer.__init__` → `Wandb.launch` → `wandb.init`, which needs a login.
**For debugging, skip wandb entirely** by appending **`training.wandb.log=False`** to the
launch command (the config's `wandb:` block sits under `training.` because the file is
`# @package training`). Alternatives: `export WANDB_MODE=offline` (logs locally, no
login) before launching, or set `log: False` in
[`conf/training/treeins_set1.yaml`](ForAINet/PointCloudSegmentation/conf/training/treeins_set1.yaml).

### wandb: new `wandb_v1_...` key rejected / `wandb login` crashes with `TypeError ... not NoneType`
The wandb client in the image is **older than the new key format**. It hard-validates
keys at exactly 40 chars (so any `wandb_v1_`-prefixed, ~86-char key is rejected), and
`wandb login` with such a key crashes later on the viewer query. This is a client-version
mismatch, not a bad key. To log online, **upgrade the client**: `pip install -U wandb`
(add it to `post_installation_commands.sh`), then `wandb login` and paste the key at the
prompt. Otherwise just disable wandb (above) for debug runs.

> **Never hardcode the key** in `post_installation_commands.sh` or any tracked file — it
> gets committed. Use `wandb login` (stores to `~/.netrc`, outside the repo). If a key
> was committed/pushed, rotate it in your wandb settings.

### wandb: `404 ... entity <name> not found during upsertBucket`
Being logged in is **not** enough — the `entity` must be a user/team namespace that
exists *and that you are a member of*. Set `training.wandb.entity` to the **team** name
(shown in parentheses by `wandb login`, e.g. `Currently logged in as: proelior (aifor)`
→ entity is `aifor`), not your username. Configured in
[`conf/training/treeins_set1.yaml`](ForAINet/PointCloudSegmentation/conf/training/treeins_set1.yaml);
override per-run with `training.wandb.entity=<team>`. Note `name:` in that file is only
the run's display label — don't put the team there.

### `DataLoader worker ... killed by signal: Bus error` / `worker exited unexpectedly`
Not a code bug — the container is **out of shared memory**. PyTorch DataLoader workers
(`num_workers > 0`) pass batches through `/dev/shm`, and Docker defaults it to **64 MB**,
which point-cloud batches exhaust after a few dozen iterations. Fixed in
[`docker-compose.yml`](docker-compose.yml) via `shm_size: "8gb"` — you must
**recreate the container** for it to apply:

```bash
docker compose down && docker compose up -d
```

Verify inside the container with `df -h /dev/shm` (should show 8G, not 64M).
Alternatives: `ipc: host` in the compose file, or run with `training.num_workers=0`
(in-process loading, never touches `/dev/shm`, but slower — this is why
`training=treeins_set1`, which sets `num_workers: 0`, didn't hit this while
`training=default` with `num_workers: 6` did).
