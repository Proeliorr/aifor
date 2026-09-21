# Methodology

My aim was narrow and deliberately falsifiable: take **ForAINet**, a published deep-learning
model for forest point-cloud segmentation, train it on a dataset it was never designed for,
then swap its sparse-convolution backbone from MinkowskiEngine to TorchSparse 1.4, retrain,
and compare the two models against each other and against the figures reported in the
original publication. Everything below serves that comparison.

---

## 1. Used materials

### The dataset

The input is **SegmentedForests** — fourteen **terrestrial** LiDAR forest plots, published
open-access on [Zenodo](https://doi.org/10.5281/zenodo.17396681) under an MIT licence and
described in *Forestry* ([10.1093/forestry/cpaf062](https://doi.org/10.1093/forestry/cpaf062)).
It is the only permanent input to the whole project: 4.4 GB of compressed LAZ, and
**847,715,392 labelled points** as prepared for training.

Plots were assigned to splits by point count, holding out the smallest so that the bulk of
the data stays in training:

| Split | Plots | Points |
|---|---|---|
| train | 11 plots | 787,235,845 |
| validation | plot_11 | 18,945,417 |
| test | plot_01, plot_14 | 41,534,130 |

Only the two **test** plots are used for the headline results; plot_11 is excluded from them
because training selected its checkpoints on it.

### Where the instance labels come from

SegmentedForests labels every point with a semantic class, but it contains no tree
identifiers — its published purpose is semantic segmentation, and the only annotation
columns in the source files are the class and a split marker. ForAINet, however, is a
*panoptic* model: it needs both a class for every point and an instance id for every
individual tree.

The instance labels therefore had to be generated. **3DFin 0.6.0**, an open-source forest
inventory tool, runs over every plot as the first stage of the pipeline and assigns each
point the id of the tree it belongs to. Only then can the preparation stage run at all — it
reads 3DFin's output, never the raw clouds.

This is a limitation worth stating up front rather than burying in the results: the instance
ground truth in this project is **algorithmically derived, not annotated by hand**, whereas
the publication's instance results are measured against manually delineated trees. The
semantic labels are unaffected — those are the dataset's own.

### Unifying the data into a form ForAINet accepts

The dataset and the model do not speak the same language, and four separate translations
were needed. None of them are cosmetic — each one, left undone, produces a model that trains
without complaint and learns the wrong thing.

**Labels: sixteen source classes into four.** The dataset's `Class` field takes sixteen
distinct values (verified as the exact union across all fourteen plots), while ForAINet
expects a field called `semantic_seg`. The map merges them:

| Source `Class` | → | Meaning |
|---|---|---|
| 8, 9, 11 | 0 | unclassified — ignored by the loss, but kept in the cloud |
| 0, 4, 5, 6, 7, 12, 13, 22, 23 | 1 | low vegetation |
| 1 | 2 | ground |
| 3, 10 | 3 | stem points |
| 2 | 4 | live branches |

The map is deliberately **total**: a source value missing from it stops that plot with an
error rather than being guessed at, because a guess would smuggle a meaningless class into
training. The *unclassified* points are not deleted — ForAINet's loss ignores them, but their
geometry still supports the features of every point around them, and removing them would
punch holes in the cloud.

**Tree identifiers: cleared everywhere except on trees.** ForAINet's convention is that
anything which is not part of a tree carries instance id `0`. The instance-segmentation tool
used to produce the identifiers (3DFin) follows the opposite convention — it stamps every
point with its nearest stem's id, ground included. ForAINet's own filter then discards any
instance whose id also appears on a non-tree point, so with the raw ids **every tree was
discarded**: zero of forty-two instances survived on one test plot. Clearing the id outside
the two tree classes touched **395,410,648 points — 46.6 % of the dataset** — and restored
all forty-two.

**Coordinates centred.** Each plot has its own minimum subtracted so that all coordinates are
non-negative. The shifts are recorded per plot so that predictions can be put back into the
original coordinate system afterwards.

**The split written into the file name.** ForAINet has no split manifest at all; it reads the
set from the filename, testing whether the characters before the extension spell `val` or
`test`. Without the suffix, all fourteen plots would silently have become training data.

### How this dataset differs from the publication's

This is the single most important difference in the project, and it frames every result.

ForAINet was built and tuned for **high-density airborne** LiDAR — the FOR-Instance dataset,
flown over the canopy. SegmentedForests is **terrestrial**: the scanner sits on the forest
floor. The two see almost inverted worlds. From above, crowns are well sampled and stems are
sparse; from below, stems are dense and crowns are occluded and interleaved with their
neighbours'. Point-density profiles, occlusion patterns and the very geometry that separates
one tree from the next are all different.

There is also one fewer class. Upstream's fifth class, `branches`, has no counterpart in
SegmentedForests, so rather than leave the model with an output that no label could ever
reach, the class was removed from the framework — a change described in section 3.

### Scripts written for the project

| Script | Role |
|---|---|
| `main_pipeline.py` + `pipeline/chain.py` | runs the whole preparation per plot, deleting each intermediate the moment it is consumed — 108.6 GB of disk down to 34.3 GB |
| `pipeline/threedfin.py` | batch-drives the 3DFin instance-segmentation tool over every plot |
| `forainet_prep.py` | the label mapping, id clearing, centring and split naming; also restores predictions to real-world coordinates |
| `pipeline/convert.py` | LAZ ↔ PLY conversion, written with no heavyweight dependencies so it also runs inside the training container |
| `class_unifier.py` | diagnostic export keeping the original and remapped labels side by side, for checking that a merge did what was intended |
| `pipeline/parallel.py` | runs several plots at once, bounded by available memory rather than by CPU count |
| `misc/check_forainet_classes.py` | re-derives what the framework should contain from my own configuration and verifies it — the guard against silently losing the patch |
| `container_export/preflight.sh` + `smoke_test.py` | a one-command pass/fail gate run on a cheap GPU before renting an expensive one |
| `misc/view_split_point_cloud.ipynb` + `view_cloud_3d.py` | a two-pane point-cloud viewer used to inspect label merges visually |

---

## 2. Used technologies and software

| Role | What was used |
|---|---|
| Data preparation (local) | Python 3.13, Hydra / hydra-zen for configuration, laspy + lazrs, plyfile, NumPy, PyYAML, 3DFin 0.6.0 |
| Visual inspection | matplotlib, tkinter, PyVista + VTK |
| The model | PyTorch, torch-points3d, ForAINet (pinned at commit `5fe600a`), MinkowskiEngine 0.5.x, TorchSparse 1.4 |
| Infrastructure | Docker Desktop (WSL2), Docker Hub, Cloudflare R2, Vast.ai (NVIDIA A100 80 GB) |
| Experiment tracking | Weights & Biases |
| Development | git with submodules, Claude Code as the coding assistant throughout |

### Encapsulating the training environment in a container

Training had to move to a rented GPU, which means the entire environment has to travel. The
steps were:

1. **Start from the existing `for-ai-net` base image rather than rebuilding it.** Several of
   its layers are no longer reproducible — MinkowskiEngine compiled from git master,
   TorchSparse, a dependency installed from a zip archive. Rebuilding risks never getting
   that environment back. It ships Python 3.8.10 and CUDA kernels compiled for compute
   capabilities 6.0 through 8.6, with no fallback for anything newer.
2. **Add one thin layer on top**, a purpose-built `Dockerfile.train`.
   It installs a LAZ backend, because the base image's laspy cannot physically open the
   dataset; pins NumPy to the version already present so that the package resolver cannot
   drag the PyTorch stack sideways; and copies the format converter to a path outside the
   working directory, which must stay an exact mirror of the local code.
3. **Make the build fail early.** A smoke test runs as a build step: it writes a two-point
   LAZ file carrying the label fields, reads it back, and asserts the labels survived. A
   broken reader is then caught on a laptop in seconds instead of on a rented GPU after the
   data has already downloaded.
4. **Publish the pieces separately** — the image to Docker Hub, and 4.32 GB of compressed
   point clouds to Cloudflare R2, chosen because it charges no egress fee for a file that
   will be downloaded repeatedly.
5. **Choose the hardware by compute capability, not by price.** The A100 is 8.0 and
   supported; a newer RTX 4090, L40S or H100 would have no compiled kernels and no
   just-in-time fallback, so it could not execute a single convolution.
6. **Gate the instance before committing to a long run.** The pre-flight script locates or
   clones the framework at the pinned commit, applies the local patch, checks that the
   configuration files parse under the *container's* library versions, expands the point
   clouds, and runs two epochs of each backbone.

One consequence is worth stating plainly: the image contains the *environment*, not the
code. Locally the code arrives through a folder mounted from the host, which a rented machine
cannot have — which is precisely why the pre-flight script clones it.

---

## 3. Changes to the original software

ForAINet is included here as a **git submodule**, and a submodule records nothing but a
commit hash. Every local edit is therefore invisible to this project's history and is
reverted by a routine `git submodule update` — without anything crashing. The edits are kept
instead as a patch file, `patches/forainet-local.patch`, covering **twelve files**, and a
checker script re-derives the expected contents from my own configuration and verifies them
in twenty-one separate assertions.

Grouped by purpose:

| Purpose | What changed |
|---|---|
| **Five classes to four** | the class tables (which the framework declares twice), the valid-class and instance-class lists, and the counters in both evaluation routines. Upstream's `branches` class has no counterpart in this dataset |
| **Making TorchSparse usable** | a new model file selected by a single configuration switch, plus two fixes in the TorchSparse compatibility layer. Without them that backbone crashes — it never records the coordinate and kernel maps its scoring sub-network needs, because that sub-network never operates at the input resolution |
| **Correctness and runnability** | replacing NumPy aliases removed in version 1.24; disabling a pre-trained checkpoint path that pointed at a *five-class* model on the original author's cluster, which would have been loaded silently and partially; making the evaluation configuration runnable at all; and five repairs to the final pooled-report script, which as shipped could not execute |
| **Performance** | replacing a worker pool that was created and destroyed on every forward pass — roughly 196 times per epoch — with a single shared one; and switching the three evaluation output writers from text to binary, which took a three-plot evaluation from 1 h 52 min to about 12 minutes |

Reapplying the patch is one command, but **verifying it is the part that matters**. The
obvious check — asking git whether the patch reverses cleanly — is *necessary but not
sufficient*: it only compares the context around each change, so an edit lying outside every
one of them passes unnoticed. That happened during the project. The reliable test rebuilds a
pristine copy of the framework from the patch and compares every file it touches.
