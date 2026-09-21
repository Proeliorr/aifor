# container_export/ — the LAZ→PLY converter, staged for the training image

Everything here is a **copy** from the repo root. Nothing is edited in place; regenerate
it by re-running the copy rather than by editing these files, or the two will drift.

## What is here, and why each file

| File | Why it must be present |
|---|---|
| `pipeline/convert.py` | the LAZ → PLY conversion itself |
| `pipeline/forainet_prep.py` | supplies `_read_cloud`, `_write_las`, `_write_ply`, `_FORAINET_DTYPE` |
| `pipeline/parallel.py` | the memory-budgeted worker pool `convert_clouds` uses |
| `pipeline/classes.py` | imported by `forainet_prep` |
| `pipeline/threedfin.py` | **dragged in by `pipeline/__init__.py`**, which imports it. Stdlib only — 3DFin never runs in the container, but the import would fail without it |
| `pipeline/__init__.py` | package marker |
| `misc/Points2ForAINet.py` | the same CLI under its historic name |
| `smoke_test.py` | build-time check (see below) |
| `preflight.sh` | one-command cheap-GPU pre-flight gate — see [`docs/preflight_cheap_gpu.md`](../docs/preflight_cheap_gpu.md) |
| `Dockerfile.train` | the thin layer |
| `.dockerignore` | stops clouds / `outputs/` / `.pyc` entering the image later |

105 KB in total. All seven Python files were checked to parse under **Python 3.8**
(`ast.parse(..., feature_version=(3,8))`) and contain no 3.9+ runtime constructs.

## Build

```bash
docker build -f Dockerfile.train -t <user>/for-ai-net:v1 .
docker push  <user>/for-ai-net:v1
```

The build is a thin layer — the base image's own layers are reused untouched, so none
of its unreproducible steps re-run.

## Run, inside the container

```bash
python3.8 -m pipeline.convert --to ply /data/export \
    /workspace/PointCloudSegmentation/data_set1_5classes/treeinsfused/raw/SegmentedForests
```

`--dry-run` lists what it would do; `--plots plot_11` restricts it; `--workers 1` runs
inline if memory is tight.

## Why the smoke test exists

The base image ships `laspy==2.0.3` and **no LAZ backend**, so it cannot open the export
at all until `lazrs` is installed — and 2.0.3 also predates `header.parse_crs`, which the
reader calls. The smoke test writes a two-point LAZ with `semantic_seg` and `treeID` as
extra-bytes dimensions, reads it back, and asserts the labels survived. It runs as a
`RUN` step so a missing backend fails the build in seconds instead of failing on a rented
GPU after the data has downloaded.

## Not here, on purpose

- **The point clouds.** 4.32 GB of `.laz` arrives from object storage at runtime.
- **The ForAINet framework.** See the note below — check whether your image already
  contains it.
- **The repo-root Hydra entry scripts** (`convert.py`, `main_pipeline.py`, …). They need
  `hydra-zen` and `version_base="1.3"`; the image has `hydra-core==1.0.7` and no
  hydra-zen, so they cannot run. The argparse CLI is the container path.

## Before you build: does the image already contain ForAINet?

The repo's copy of `ForAINet/PointCloudSegmentation/Dockerfile` never `COPY`s any source
— it builds an environment only. But that may not be the Dockerfile that produced your
`for-ai-net` image. Check:

```bash
docker run --rm for-ai-net ls /workspace/PointCloudSegmentation/train.py
```

- **Found** → this export is complete; build and push.
- **Not found** → the image has no training code (locally it came from the
  `./ForAINet:/workspace` bind mount, which does not exist on a rented host). Either let
  `preflight.sh` clone it at the pinned `5fe600a` on the instance (the normal route now — see
  [`docs/preflight_cheap_gpu.md` §2b](../docs/preflight_cheap_gpu.md#2b-what-the-script-repairs-by-itself);
  note the tree then lives at `/workspace/ForAINet/`, one level deeper), or bake it in by
  adding to `Dockerfile.train`:

  ```dockerfile
  COPY ForAINet/ /workspace/
  ```

  and copy `ForAINet/` in here too — 837 files, 6.3 MB, excluding `outputs/` (627 MB of
  past runs), `data_set1_5classes/`, `.git/` and `__pycache__/`.
