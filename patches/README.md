# Local patches for the `ForAINet/` submodule

`ForAINet/` is a **git submodule** pinned to upstream
[prs-eth/ForAINet](https://github.com/prs-eth/ForAINet) at commit `5fe600a`.

A submodule records only a commit SHA in this repo — any edit you make *inside*
`ForAINet/` is invisible to this repo's history and would be lost by a re-clone or a
`git submodule update --force`. So the local edits we depend on live here as a patch
instead: versioned in **our** repo, never pushed upstream.

## `forainet-local.patch`

Adapts upstream to this project. **Twelve files**, of which three carry the class-scheme
change and are the ones that matter:

| File | Why |
|---|---|
| `conf/eval.yaml` | **Local inference.** Upstream ships an empty `checkpoint_dir` and eleven test paths on the author's cluster, so it cannot run. Now: `checkpoint_dir: /workspace/pre-trained_models` (a *container* path — `./ForAINet` is mounted at `/workspace`, and both `.pt` files live in `ForAINet/pre-trained_models/`, untracked on purpose), `fold` = plot_01_test, plot_14_test, plot_11_val, and a run dir that includes `${model_name}` so the two backbones' append-mode reports never share a folder. See [`docs/eval_process.md`](../docs/eval_process.md) §8 |
| `torch_points3d/utils/meanshift_cluster.py` | **Performance, both backends.** Upstream builds and destroys a `multiprocessing.Pool` inside `cluster_single` on *every* forward pass — ~196 pool lifecycles per epoch once `prepare_epoch: 30` opens the instance branch, each forking up to `batch_size` copies of an ~8 GB process holding a live CUDA context. Replaced with one process-wide pool. Results are unchanged: `pool.map` is order-preserving and `MeanShift(bandwidth, bin_seeding=True)` is deterministic |
| `torch_points3d/models/panoptic/PointGroup3heads_ts.py` | **new file** — the same model on the TorchSparse 1.4 backbone, selected by `model_name=PointGroup-PAPER-TS`. MinkowskiEngine stays the default; see [`docs/backbone_torchsparse_1.4.md`](../docs/backbone_torchsparse_1.4.md) |
| `torch_points3d/modules/SparseConv3d/nn/torchsparse.py` | **TorchSparse only — two fixes for one root cause.** `ScorerUnet`'s encoder starts at stride 2, so nothing in it ever operates at the input resolution, and TorchSparse 1.4 only records what a conv *produces*. Its decoder then needs two things that were never built: the **coordinate** map (`cmaps[(1,1,1)]`, seeded in `SparseTensor()`) and the **kernel** map (`kmaps[((1,1,1),(3,3,3),(1,1,1),(1,1,1))]`, avoided by running stride-1 "transposed" convs forward — at stride 1 the two differ only by a flip of a learned kernel). Without them: `KeyError: (1, 1, 1)` then `KeyError: ((1, 1, 1), (3, 3, 3), (1, 1, 1), (1, 1, 1))`. The backbone escapes both because its stride-1 stem builds them. **Invisible until epoch 31**, when `prepare_epoch: 30` runs the scorer for the first time |
| `evaluation_stats_FOR.py` | **The pooled report across plots** — the "final" number, and it was five-class too (`NUM_CLASSES_sem`, `sem_classcount`, `sem_classcount_remove_ground`, `thing_classes`). Also fixed: `np.float` (numpy ≥ 1.24), a non-raw Windows path that silently matched no files, binary counters that were re-zeroed per plot so every "Binary Semantic Segmentation" figure described only the last plot, and `positive_classes[[…]]` whose doubled brackets made `float()` raise. Now takes `<run_dir> [index …]` on the command line and, with the unused `torch_points3d` import dropped, runs on the host in `aifor` |
| `torch_points3d/metrics/panoptic_tracker_pointgroup_treeins_partseg.py` | `np.float` → `float`. NumPy 1.24 (what the training image ships) removed the alias, so `_compute_eval` raises `AttributeError`. Also invisible until epoch 31 — the instance metrics only run once clustering starts |
| `torch_points3d/datasets/segmentation/treeins_set1.py` | **5 → 4 classes.** `Treeins_NUM_CLASSES`, `INV_OBJECT_LABEL`, `OBJECT_COLOR` — upstream's class 4 `branches` has no SegmentedForests counterpart, so nothing can ever map to it |
| `torch_points3d/datasets/panoptic/treeins_set1.py` | the same table again (it is declared twice), plus `VALID_CLASS_IDS`, `SemIDforInstance`, and the `final_eval` counters `NUM_CLASSES_sem`, `NUM_CLASSES_count`, `sem_classcount`, `thing_classes`. Also: its three eval PLY writers (`to_ply`, `to_eval_ply`, `to_ins_ply`) emit **binary** instead of ASCII. ASCII was ~4× larger (1.32 GB vs 333 MB for one plot) and dominated a 1 h 52 min eval (the same eval takes ~12 min binary). Nothing downstream minds: `tree_metrics/`, `merge_tiles.py`, `evaluation_stats*.py` and our restore step all read via plyfile, which takes either format |
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

## Never use a YAML merge key in a config the patch touches

`<<: *anchor` **cannot be loaded by the training image.** It has `hydra-core==1.0.7`, which
pins omegaconf 2.0.x, whose yaml loader calls `construct_object()` on every key node and has
no constructor for `tag:yaml.org,2002:merge`. The file then raises *while parsing*:

```
yaml.constructor.ConstructorError: could not determine a constructor for the tag
'tag:yaml.org,2002:merge' in ".../conf/models/panoptic/FORpartseg_3heads.yaml", line 210
```

Parsing happens before hydra reads `model_name`, so **every** backbone dies and the message
names none of them. This is not theoretical: the TorchSparse block shipped with a merge key
and both runs of the pre-flight failed on a rented box that way.

The trap is that omegaconf **2.3** — what the `aifor` env has — overrides `construct_mapping`
instead and lets PyYAML expand merges, so the same file is perfectly valid on the dev machine.
Duplicate the keys instead, and let `misc/check_forainet_classes.py` guard the copy: it
re-implements 2.0's loader and asserts `PointGroup-PAPER-TS` still matches `PointGroup-PAPER`
except for `class` and `backend`. `container_export/preflight.sh` re-checks the same thing on
the box, right after applying the patch, so a bad config costs a second instead of a download
plus a conversion.

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

Verify the result **without touching the working tree**, in two steps:

```bash
git apply --check --reverse ../patches/forainet-local.patch   # necessary, NOT sufficient
```

**Exit 0 there does not prove the patch is complete.** A reverse check only needs each
hunk's *context* to match, so an edit lying outside every hunk passes unseen. This
happened: `to_eval_ply`'s switch to `text=False` sat between two existing hunks of
`panoptic/treeins_set1.py` and was missing from the patch for a whole session, while this
check kept returning 0. The real test rebuilds a pristine tree from the patch and compares
every file the patch touches:

```bash
P="$(pwd)/../patches/forainet-local.patch"
T=/d/temp/forainet-verify                    # scratch on D:, not the C: temp
rm -rf "$T" && mkdir -p "$T"
git archive 5fe600a | tar -x -C "$T"
(cd "$T" && git apply "$P") || echo "DOES NOT APPLY to a pristine 5fe600a"
for f in $(grep '^diff --git' "$P" | sed 's|.* b/||'); do
  # tr: this checkout has core.autocrlf=true, the pristine tree is LF
  diff -q <(tr -d '\r' < "$T/$f") <(tr -d '\r' < "$f") >/dev/null || echo "NOT IN PATCH: $f"
done
rm -rf "$T"                                  # silence = the patch reproduces your tree
```

`git apply` prints ~35 *trailing whitespace* warnings on the way. They are expected: they
come from `PointGroup3heads_ts.py`, which carries upstream's own formatting unchanged.

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
