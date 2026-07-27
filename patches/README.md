# Local patches for the `ForAINet/` submodule

`ForAINet/` is a **git submodule** pinned to upstream
[prs-eth/ForAINet](https://github.com/prs-eth/ForAINet) at commit `5fe600a`.

A submodule records only a commit SHA in this repo — any edit you make *inside*
`ForAINet/` is invisible to this repo's history and would be lost by a re-clone or a
`git submodule update --force`. So the local edits we depend on live here as a patch
instead: versioned in **our** repo, never pushed upstream.

## `forainet-local.patch`

Adapts upstream to this project. Three files:

| File | Why |
|---|---|
| `PointCloudSegmentation/conf/training/default.yaml` | short debug runs (`epochs: 5`, `num_workers: 0`, `batch_size: 4`), our `wandb` entity/project, tensorboard off |
| `PointCloudSegmentation/conf/training/treeins_set1.yaml` | our `wandb` entity + experiment name (upstream ships the author's `binbin`) |
| `PointCloudSegmentation/train.py` | `import debugpy` + `debugpy.breakpoint()` for container debugging |

`num_workers: 0` matters on Windows/Docker: DataLoader workers pass batches through
`/dev/shm`, which is why `docker-compose.yml` also raises `shm_size`.

## Reapply after a fresh clone / submodule reset

```bash
git submodule update --init            # get ForAINet at the pinned commit
cd ForAINet
git apply ../patches/forainet-local.patch
```

Check it worked — exactly these three files should be modified, nothing else:

```bash
git -C ForAINet status --porcelain | grep -v '^??'
```

## Regenerate the patch after changing something in ForAINet/

```bash
cd ForAINet
git diff -- PointCloudSegmentation/conf/training/default.yaml \
            PointCloudSegmentation/conf/training/treeins_set1.yaml \
            PointCloudSegmentation/train.py > ../patches/forainet-local.patch
```

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
