# Local patches for the `ForAINet/` submodule

`ForAINet/` is a **git submodule** pinned to upstream
[prs-eth/ForAINet](https://github.com/prs-eth/ForAINet) at commit `5fe600a`.

A submodule records only a commit SHA in this repo — any edit you make *inside*
`ForAINet/` is invisible to this repo's history and would be lost by a re-clone or a
`git submodule update --force`. So the local edits we depend on live here as a patch
instead: versioned in **our** repo, never pushed upstream.

## `forainet-local.patch`

Adapts upstream to this project. **Seven files**, of which three carry the class-scheme
change and are the ones that matter:

| File | Why |
|---|---|
| `torch_points3d/models/panoptic/PointGroup3heads_ts.py` | **new file** — the same model on the TorchSparse 1.4 backbone, selected by `model_name=PointGroup-PAPER-TS`. MinkowskiEngine stays the default; see [`docs/backbone_torchsparse_1.4.md`](../docs/backbone_torchsparse_1.4.md) |
| `torch_points3d/datasets/segmentation/treeins_set1.py` | **5 → 4 classes.** `Treeins_NUM_CLASSES`, `INV_OBJECT_LABEL`, `OBJECT_COLOR` — upstream's class 4 `branches` has no SegmentedForests counterpart, so nothing can ever map to it |
| `torch_points3d/datasets/panoptic/treeins_set1.py` | the same table again (it is declared twice), plus `VALID_CLASS_IDS`, `SemIDforInstance`, and the `final_eval` counters `NUM_CLASSES_sem`, `NUM_CLASSES_count`, `sem_classcount`, `thing_classes` |
| `conf/models/panoptic/FORpartseg_3heads.yaml` | `path_pretrained: null` (upstream points it at a **5-class** checkpoint on the author's cluster, see below), plus the `PointGroup-PAPER-TS` block that selects the TorchSparse model file |
| `conf/training/treeins_set1.yaml` | our `wandb` entity + experiment name (upstream ships the author's `binbin`) |
| `conf/training/default.yaml` | short debug runs (`epochs: 5`, `num_workers: 0`, `batch_size: 4`), our `wandb` entity/project, tensorboard off |
| `train.py` | `import debugpy` + `debugpy.breakpoint()` for container debugging |

**Why `path_pretrained` had to go.** It named a checkpoint trained with five classes.
The path does not exist here, so every run so far logged *"The path does not exist, it
will not load any model"* (`base_model.py:154`) and trained from scratch **by accident**.
Worse, if the path ever resolved, `load_state_dict_with_same_shape(m, strict=False)`
silently skips layers whose shape changed — our semantic head is 4-wide, that one is
5-wide, so the head would be dropped without a word and you would believe you were
fine-tuning. `null` makes "from scratch" a decision.

`num_workers: 0` matters on Windows/Docker: DataLoader workers pass batches through
`/dev/shm`, which is why `docker-compose.yml` also raises `shm_size`.

> **The `train.py` breakpoint is for local debugging only.** `debugpy.breakpoint()` sits
> immediately before `trainer.train()`. Strip or guard it before baking a training image
> — it has no place on a rented GPU with no debugger attached.

## Reapply after a fresh clone / submodule reset

```bash
git submodule update --init            # get ForAINet at the pinned commit
cd ForAINet
git apply ../patches/forainet-local.patch
```

Then **verify it, rather than trusting it** — a reverted patch does not crash anything,
it just trains a 5-class head on 4-class data and divides the metrics by the wrong
number:

```bash
python misc/check_forainet_classes.py        # exits non-zero on any mismatch
```

That script re-derives what ForAINet should contain from the `classes:` block of
`conf/config.yaml` and parses the submodule to confirm. It reads the source rather than
importing it, so it needs no torch and runs anywhere. Run it after every submodule
operation, and before building a training image.

## Regenerate the patch after changing something in ForAINet/

```bash
cd ForAINet
# `git diff` cannot see a file git has never heard of, and the patch ADDS one.
# Without this line PointGroup3heads_ts.py silently vanishes from the patch.
git add -N PointCloudSegmentation/torch_points3d/models/panoptic/PointGroup3heads_ts.py
git diff -- . ':(exclude)*.pyc' > ../patches/forainet-local.patch
```

The `.pyc` exclusion is essential: upstream commits bytecode (see below), so a bare
`git diff` sweeps ~115 recompiled `.pyc` files into the patch.

Verify the result **without touching the working tree** — if the patch reverses cleanly,
it is an exact description of what you have:

```bash
git apply --check --reverse ../patches/forainet-local.patch   # exit 0 = faithful
```

(Do not try to verify by stashing: `git stash` refuses to move intent-to-add entries, so
the tree does not actually become pristine and the test silently proves nothing.)

**Other untracked files are still not captured.** `git diff` only sees tracked changes
plus anything you `add -N`, so anything else you created inside `ForAINet/` (e.g.
`omegatest.py`) lives nowhere but your disk. Copy it into this repo if it matters.

Write it with a tool that does **not** add a UTF-8 BOM — PowerShell's
`Out-File -Encoding utf8` does, and `git apply` then rejects the file. The shell
redirection above is fine.

## Note on `__pycache__`

Upstream commits `__pycache__/*.pyc`, so running the training rewrites tracked files
and `git status` inside the submodule goes dirty with ~115 `.pyc` entries. That is
harmless noise, not something you changed. To restore a clean tree:

```bash
cd ForAINet && git checkout -- . && git apply ../patches/forainet-local.patch
```
