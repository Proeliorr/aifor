# How ForAINet evaluates — `eval.py`, end to end

`eval.py` is eighteen lines and tells you nothing. Everything that decides what a number
means happens four files deeper. This document traces that path and defines every
statistic the run produces.

Companion documents: [`ARCHITECTURE_ANALYSIS.md`](ARCHITECTURE_ANALYSIS.md) for the model
and data pipeline, and [`train_logs_explained.md`](train_logs_explained.md) for the
*training* loop and its log. This one covers evaluation only.

---

## 1. What `eval.py` actually does

```python
@hydra.main(config_path="conf", config_name="eval")
def main(cfg):
    OmegaConf.set_struct(cfg, False)
    trainer = Trainer(cfg)
    trainer.eval(stage_name="test")
```

That is the whole file. It loads `conf/eval.yaml`, builds a `Trainer`, and asks it to run
the **test** stage. No metric is computed here; no output path is chosen here.

The `Trainer` constructor is what actually does the heavy lifting — it restores the
checkpoint named by `checkpoint_dir` + `weight_name`, builds the dataset from `data.fold`,
and asks the **dataset** for a tracker.

## 2. The call path

```
eval.py
└─ Trainer.eval(stage_name="test")                   trainer.py:175
   └─ _test_epoch(epoch, "test")                     trainer.py:237
      │   for each test file, for each batch:
      │     model.forward()  →  tracker.track()      …treeins_partseg.py:153
      └─ _finalize_epoch(epoch)                      trainer.py:187
         ├─ tracker.finalise(**tracker_options)      …treeins_partseg.py:729
         │  ├─ block-merge predictions back to full resolution
         │  ├─ write the prediction PLYs
         │  └─ dataset.final_eval(...)               panoptic/treeins_set1.py:107
         └─ tracker.print_summary()
```

Three things in that path are not obvious and are worth knowing:

- **Evaluation cannot damage your checkpoints.** `eval()` sets `_is_training = False`, and
  `_finalize_epoch` guards the `publish()` / `save_best_models_under_current_metrics()`
  calls behind that flag. An eval run is read-only with respect to your training outputs.
- **The tracker is chosen by the dataset, not the config.** `get_tracker()` at
  `panoptic/treeins_set1.py:750` returns
  `panoptic_tracker_pointgroup_treeins_partseg.PanopticTracker`. There are **eleven**
  `panoptic_tracker*.py` files in `torch_points3d/metrics/`, and the obvious-looking
  `panoptic_tracker.py` is not the one in use. If you are reading metric code, start from
  `get_tracker`, not from the file name.
- **`final_eval` lives in the dataset module**, not in the metrics package — it is a plain
  module-level function in `panoptic/treeins_set1.py`, called by the tracker.

## 3. Two tiers of metrics, and only one of them is the answer

This is the single most important thing in this document. The run reports numbers twice,
computed differently, and they do not agree.

| | Tier 1 — live tracker | Tier 2 — `final_eval` |
|---|---|---|
| Computed | every batch | once per test file, at the end |
| Over what | subsampled cylinders, as the loader yields them | the **full-resolution** cloud, after block-merging |
| Shown where | tqdm postfix, wandb, `print_summary()` | `Evaluation_<i>.txt` on disk |
| Instance metrics | approximate, cheap | the real matching |
| Use it for | watching a run progress | **reporting results** |

Quote Tier 2. Tier 1 exists so you can see whether a run is alive.

### Tier 1 — what the tracker reports live

From `get_metrics()` (`…treeins_partseg.py:1197`), which extends the segmentation base:

| Metric | Meaning |
|---|---|
| `loss_*` | the individual loss terms (semantic, offset, score, embedding…) |
| `acc` | overall point accuracy |
| `macc` | mean per-class accuracy |
| `miou` | mean IoU over classes |
| `iou_per_class` | the per-class breakdown (verbose mode) |
| `pos`, `neg` | mean predicted score for matched / unmatched proposals |
| `Iacc` | instance accuracy |
| `cov`, `wcov` | coverage and size-weighted coverage |
| `mIPre`, `mIRec`, `F1` | instance precision / recall / F1 |
| `map` | mean average precision over the tracked classes |

All are running averages over whatever cylinders happened to be sampled. See
[`train_logs_explained.md`](train_logs_explained.md) §C1 for why the instance ones sit at
`0.0` early in training — that is expected, not a bug.

## 4. Tier 2 — the `Evaluation_<i>.txt` report

Written once per test file, appending to `Evaluation_<i>.txt`. **Append**, not truncate —
re-running eval into the same directory stacks a second report below the first, and it is
easy to read the stale one by mistake.

Before any metric is computed, `finalise` block-merges the per-cylinder predictions back
onto the original full-resolution points, which is what `tracker_options.full_res: True`
buys. Without it the numbers describe the subsample, not the plot.

## 5. Every statistic, defined

`final_eval` emits 44 numbers in four groups. Each is `mX` (mean over classes) or a
per-class array.

### Semantic — how well is each point labelled?

| Metric | What it is | What it hides |
|---|---|---|
| `oAcc` | overall accuracy: correct points ÷ all points | Dominated by whichever class is biggest. In a forest plot that is usually ground or low vegetation, so a lazy model scores well |
| `mAcc` | per-class accuracy, averaged | Treats a 2%-of-points class equally with a 60% class — the honest counterweight to `oAcc` |
| `IoU` | per class: TP ÷ (TP + FP + FN) | — |
| `mIoU` | the mean of those | The usual headline number |

Reported twice: over all four classes, and **"without ground points"** — over
`sem_classcount_remove_ground = [1, 3, 4]`, i.e. dropping *ground*. Ground is large, flat
and easy; excluding it shows whether the model can actually do the hard part.

### Binary semantic — tree vs non-tree

The same four metrics after collapsing everything to *stuff* vs *thing*. Useful as a
sanity floor: if binary mIoU is poor, nothing downstream can be good.

### Instance — did we find the individual trees?

A predicted instance and a ground-truth instance are **matched when their IoU ≥ 0.5**
(`at = 0.5`, `panoptic/treeins_set1.py:156`). Everything below follows from that
threshold.

| Metric | What it is | What it hides |
|---|---|---|
| `MUCov` | mean coverage: for each GT tree, the IoU of its best-matching prediction, averaged **unweighted** | Every tree counts the same — a sapling and a 40 m spruce are equal |
| `MWCov` | the same, **weighted by tree point count** | The opposite bias: big trees dominate, so you can lose every small tree and still score well |
| `Precision` | of the trees we predicted, how many were real | Nothing about the ones we missed |
| `Recall` | of the real trees, how many we found | Nothing about spurious extras |
| `F1` | harmonic mean of the two | Which of the two failed |

**Read `MUCov` and `MWCov` together.** A large gap between them is the signal that the
model handles one size class much better than the other — which for forest inventory is
usually the finding that matters.

### Panoptic — segmentation and detection in one number

| Metric | Definition | Reads as |
|---|---|---|
| `SQ` | Segmentation Quality — mean IoU of the *matched* pairs only | "when we found a tree, how well did we outline it?" |
| `RQ` | Recognition Quality — F1 over the matching, `TP / (TP + ½FP + ½FN)` | "how many trees did we find at all?" |
| `PQ` | `SQ × RQ` | the combined score |
| `PQ*` | PQ variant that treats stuff classes as always-matched | fairer to ground/undergrowth, which have no instances |

Each is reported **three ways** — all classes, `(things)` only, `(stuff)` only.

Because `PQ = SQ × RQ`, a single PQ cannot tell you which half failed. A model that finds
half the trees but outlines them perfectly, and one that finds every tree but outlines
them sloppily, can report the same PQ. **Always quote SQ and RQ alongside it.**

## 6. Class indexing — two schemes inside one function

The easiest thing to get wrong here. `final_eval` uses two different numbering systems
twelve lines apart.

The journey of one label:

| Stage | Value for *stem_points* | Set by |
|---|---|---|
| our `classes:` block, and on disk in the PLY | `3` | `conf/config.yaml` → Stage 2 |
| after ForAINet's reader subtracts one | `2` | `segmentation/treeins_set1.py` (`semantic_seg - 1`) |
| what the model predicts | `2` | head width = `dataset.num_classes` = 4 |
| inside `final_eval`, after `+ 1` | `3` | `panoptic/treeins_set1.py:170` |

So `final_eval` works in **1-based** labels where **`0` is the ignore class**:

```python
sem_classcount  = [1, 2, 3, 4]   # low_vegetation, ground, stem_points, live_branches
stuff_classes   = [1, 2]         # low_vegetation, ground
thing_classes   = [3, 4]         # stem_points, live_branches   <- carry instances
```

Meanwhile, **`NUM_CLASSES = 3` in the same function is a different universe**: a binary
collapse into *unclassified / stuff / thing*, used to size the instance arrays. There,
`stuff_classcount = [1]` and `ins_classcount = [2]`. Same naming style, unrelated
numbering — this is why the instance results print under index `[2]`.

`idxc` (`panoptic/treeins_set1.py:175`) restricts the instance evaluation to points that
are a thing class in either the prediction or the ground truth.

> **Keep this consistent.** These constants are set by
> [`patches/forainet-local.patch`](../patches/README.md), which a `git submodule update`
> silently reverts. `python misc/check_forainet_classes.py` re-derives all of them from
> `conf/config.yaml` and fails loudly on a mismatch. Run it before every training image
> build — a reverted patch does not crash, it just produces wrong numbers.

## 7. What lands on disk

Hydra run directory: `${checkpoint_dir}/eval/${model_name}/${now:%Y-%m-%d_%H-%M-%S}` —
`${model_name}` is our addition, so the two backbones never write into one folder. On this
machine that is `ForAINet/pre-trained_models/eval/<model_name>/<timestamp>/`. Per test file
`i`, where `i` is the file's **position in `data.fold`**:

| File | Contents |
|---|---|
| `Evaluation_<i>.txt` | the report from §5 — **appended** |
| `Semantic_results_forEval_<i>.ply` | predicted vs GT semantic labels, full resolution |
| `Instance_Results_forEval_<i>.ply` | predicted instance ids, full resolution |
| `Instance_results_withColor_<i>.ply` | the same, random colour per instance — for looking at |
| `Instance_subsample_<i>.ply` | instance ids at subsampled resolution |
| `vote1regular.ply_<i>.ply` | per-file voting output — the tracker appends `_<i>.ply` to the *whole* `tracker_options.ply_output`, hence the doubled extension |
| `vote1regularfull.ply`, `Instance_Offset_results_forEval.ply` | run-level, **not** per-file, so they are overwritten each file |
| `<model_name>.pt` | a **copy of the checkpoint** (761 MB) — `Checkpoint.load` copies it into every run directory when `checkpoint_dir` is set. Safe to delete afterwards |

The `_withColor` file is the one to open in a viewer; the others carry raw ids.

**Every PLY is binary little-endian** (our patch; upstream wrote ASCII). The format decides
how long eval takes. For plot_01 (20.8 M points), `Semantic_results_forEval_0.ply` is
333 MB binary against 1.32 GB as text. The first local eval, all ASCII, took **1 h 52 min**,
mostly formatting floats through the Windows bind mount; with the `_forEval` writers binary,
the next took 34 min. Nothing downstream minds: `tree_metrics/`, `merge_tiles.py`, the
standalone `evaluation_stats*.py` and `forainet_prep.py forainet_prep.mode=restore` all
read through plyfile, which takes either format. (The `evaluation_stats*.py` scripts do
import KPConv's `read_ply`, which rejects ASCII — but they never call it.)

## 8. Running it on SegmentedForests

Upstream's `conf/eval.yaml` pointed at the original author's machine and could not run.
[`patches/forainet-local.patch`](../patches/README.md) now carries a working one for the
**local Docker container** (`docker-compose.yml` mounts `./ForAINet` at `/workspace`):

| Key | Upstream shipped | Ours |
|---|---|---|
| `checkpoint_dir` | empty | `/workspace/pre-trained_models` — the host's `ForAINet/pre-trained_models/`, holding **both** trained models |
| `model_name` | `PointGroup-PAPER` | unchanged; `PointGroup-PAPER-TS` on the command line for TorchSparse |
| `weight_name` | `"latest"` | unchanged — see §9 before changing it |
| `data.fold` | eleven paths under `/cluster/work/igp_psr/binbin/…` | three **container** paths, below |

**Every path is as the container sees it.** MinkowskiEngine and TorchSparse are Linux-only,
so `eval.py` never runs on Windows, and a `D:\…` path in this file cannot work.

**One directory, two models.** `Checkpoint.load` resolves `<checkpoint_dir>/<model_name>.pt`,
and the files are named exactly after their model blocks. Keep `pre-trained_models/`
**untracked** in the submodule — never `git add -N` it — so the 1.5 GB cannot enter the patch.

**What a `.pt` contains.** 22 complete weight sets: `latest` (epoch 99) and 21
`best_<metric>` snapshots. `weight_name` picks one — `miou` loads `best_miou`.

```yaml
data:
  fold: ['/workspace/PointCloudSegmentation/data_set1_5classes/treeinsfused/raw/SegmentedForests/plot_01_test.ply',   # -> Evaluation_0
         '/workspace/PointCloudSegmentation/data_set1_5classes/treeinsfused/raw/SegmentedForests/plot_14_test.ply',   # -> Evaluation_1
         '/workspace/PointCloudSegmentation/data_set1_5classes/treeinsfused/raw/SegmentedForests/plot_11_val.ply']    # -> Evaluation_2
```

The PLYs come from the LAZ export — a pure format change, so the coordinates stay centred
exactly as the model trained on them:

```powershell
cmd /c "mamba run -n aifor python convert.py convert.plots=[plot_11_val,plot_01_test,plot_14_test]"
```

Then, inside the container, **from `/workspace/PointCloudSegmentation`** — `dataroot` is
resolved against the launch directory (`hydra.utils.to_absolute_path` in
`dataset_factory.py`), so starting anywhere else puts the cache in the wrong place:

```bash
python eval.py                                   # MinkowskiEngine
python eval.py model_name=PointGroup-PAPER-TS    # TorchSparse
```

**The `_val` / `_test` suffix means nothing to `eval.py`.** It matters only to `train.py`,
which splits `raw/**/*.ply` by name (`segmentation/treeins_set1.py:381-386`), evaluates the
`_val` set every epoch (`trainer.py:164-165`) and picks the `best_*` checkpoints on it
(`base_dataset.py:519-521`). Once `fold` holds paths, every split goes through
`process_test()` instead, and `eval(stage_name="test")` skips the val stage
(`trainer.py:179-181`). So any labelled PLY works, whatever it is called.

Why all three plots: the two `_test` plots never influenced any weights, so they are the
numbers to **report**. `plot_11_val` is still worth running — with `latest`, nothing picked
those weights on it either — but it is the plot training watched.

**On a 6 GB GPU**, `full_res: True` moves whole plots onto the device (`tracker.track`,
`self._test_area[i].to(model.device)`). Windows desktop apps already hold ~1.5 GB of it. If
eval runs out of memory, close GPU-heavy apps or evaluate one plot per run — its report is
then `Evaluation_0.txt`:

```bash
python eval.py "data.fold=['/workspace/PointCloudSegmentation/data_set1_5classes/treeinsfused/raw/SegmentedForests/plot_11_val.ply']"
```

Afterwards, put the predictions back into real-world coordinates:

```bash
mamba run -n aifor python forainet_prep.py \
    forainet_prep.mode=restore \
    forainet_prep.restore_dir=<the eval run directory>
```

which reads `<plot>_offsets.yml` and writes `restored_<plot>.laz` with the source scales
and CRS re-applied.

## 9. Traps

- **`weight_name: "latest"` evaluates the last epoch, not the best one.** If the run
  overfit after its best epoch, this quietly reports the worse model.
- **…but every `best_*` was picked on `plot_11_val`**, so a `best_*` snapshot evaluated on
  plot_11 itself is optimistic. It is fair only on the `_test` plots.
- **A mistyped `weight_name` does not fail.** `Checkpoint.get_state_dict`
  (`model_checkpoint.py:136-145`) wraps the `best_<name>` lookup in a bare `except:` and
  silently loads `latest`. Read the log line `Model loaded from …:<key>` to see what you
  actually got.
- **Never `weight_name: miou` for TorchSparse.** Its `best_miou` is from **epoch 22** —
  before `prepare_epoch: 30`, so that snapshot's ScoreNet was never trained and its instance
  results are meaningless. (Minkowski's `best_miou` is epoch 66, which is fine.)
- **The per-plot eval cache is reused by file name.** `process_test` skips any plot whose
  `processed_0.2_test/processed_<stem>.pt` already exists. Re-convert a PLY under the same
  name and eval silently uses the old tensors — delete `processed_0.2_test/` first. (The
  combined `processed_test.pt` is rewritten every run, so changing `fold` alone is safe.)
- **`conf/config.yaml` defaults to `models: panoptic/area4_ablation_2` — a file that does
  not exist.** Confirmed: `conf/models/panoptic/area4_ablation_2.yaml` is absent. You must
  override `models=` (the real one is `panoptic/FORpartseg_3heads`) or Hydra fails at
  composition.
- **`Evaluation_*.txt` is opened in append mode.** Re-running into the same directory
  stacks reports. Read the last block, or evaluate into a fresh directory.
- **`full_res: True` is what makes the numbers comparable.** Turning it off silently
  changes what the metrics describe.
- **`batch_size: 1`, `num_workers: 0`, `voting_runs: 1`** — eval defaults are
  conservative. `num_workers: 0` will bottleneck a fast GPU; raising it needs `shm_size`
  set on the container.
- **The 14 standalone `evaluation_stats_FOR*.py` scripts are not part of this path.**
  Several are named `set15classes` and still assume **five** classes. They were not
  touched by our patch; do not mix their output with `Evaluation_<i>.txt`.

## 10. Is an evaluation run correct? — five checks

The class count is not in `train.py` or `eval.py`. It lives in the dataset modules and in
`final_eval`, which both entry points load through the same `Trainer`, so eval cannot
disagree with training — **unless the patch was reverted**, and then nothing crashes. Check,
in about a minute:

1. **Tables agree with our config:** `python misc/check_forainet_classes.py` → `OK: … (4 classes, things=[3, 4])`.
2. **The model really is 4-class.** The eval log's module tree ends the `Semantic` head with
   `Linear(in_features=16, out_features=4)`, and it prints **`Model size = 11872109`**.
   `11872126` is exactly 17 more — one extra output of that head (16 weights + 1 bias) — and
   means a 5-class model.
3. **The weights you meant:** `Model loaded from …/<model_name>.pt:<key>`. A mistyped
   `weight_name` silently falls back to `latest` (§9).
4. **The mean excludes the ignore slot.** `Semantic Segmentation IoU` has **five** entries,
   `[ignore, low_vegetation, ground, stem_points, live_branches]`. `mIoU` must equal the
   mean of the last four, not all five. Recompute it once by hand.
5. **One report per file.** Each `Evaluation_<i>.txt` holds a single block starting with
   `Semantic Segmentation oAcc:` (don't count `Binary Semantic Segmentation oAcc:`, which
   also matches that text).

**Uneven per-class IoU is not by itself a bug.** On plot_14, `live_branches` scores 0.97
and `ground` 0.65. That's because live_branches is 68.8% of the points and ground only 4.3%,
a thin sheet at the height of low vegetation. To rule out swapped labels, check heights: in
the PLYs, `ground` sits a median 0.05–0.22 m above the lowest point of its 1 m cell, and
`live_branches` 6.8–11.9 m. The results of the first two runs are recorded in
[`progress.md`](progress.md) (2026-09-17).

## 11. The pooled report across plots — `evaluation_stats_FOR.py`

`Evaluation_<i>.txt` is **per plot**. `evaluation_stats_FOR.py` pools several plots into one
set of numbers: one confusion matrix over all their points, and one instance matching over
all their trees. That is the figure to quote as *the* result, and the one to compare
backbones with.

It needed the same 5 → 4 class fix as `final_eval`, plus four repairs before it could run at
all — see [`patches/README.md`](../patches/README.md). `misc/check_forainet_classes.py` now
guards its constants too, so a reverted patch is caught there.

**It runs on the host**, not in the container: its only `torch_points3d` import was unused
and is gone.

```powershell
cd ForAINet\PointCloudSegmentation
cmd /c "mamba run -n aifor python evaluation_stats_FOR.py <eval_run_dir> 0 1"
```

- The trailing numbers are **positions in `eval.yaml`'s `data.fold`**. `0 1` pools the two
  `_test` plots and leaves out `plot_11_val`, which training selected checkpoints on — that
  is the defensible headline. Pass no numbers to pool every plot in the folder.
- Output goes to `evaluation_total.txt` **inside the run folder, in append mode**, so
  re-running stacks blocks. Each block is headed with the timestamp, the run directory, the
  indices pooled and the fold filename behind each index, read from the run's own
  `.hydra/config.yaml` — so a stacked file stays readable.
- Expect it to take several minutes per plot: the instance part is an O(predicted × truth)
  IoU over full-resolution masks.

**Read the pooled numbers as pooled.** `oAcc` and `mIoU` weight every *point* equally, so a
larger plot pulls them; `Evaluation_<i>.txt` remains the way to see a plot that behaves
differently. The pooled mIoU must land between the per-plot values — a quick sanity check.

### Eval is deterministic — but it has two regimes

Repeating a run reproduces it **exactly**: three runs per backbone gave byte-identical
`Evaluation_<i>.txt`, every metric, both backbones. So a difference between two models is
real and not noise.

**What does change the numbers is whether `processed_0.2_test/` already existed.** The run
that *builds* that cache and the runs that *load* it give two different, each perfectly
reproducible, sets of numbers — measured on the same checkpoint:

| plot_01_test | cache built by the run | cache loaded |
|---|---|---|
| mIoU | 0.6972 | 0.6996 |
| instance recall | 0.1102 | 0.0932 |

| plot_14_test | cache built by the run | cache loaded |
|---|---|---|
| mIoU | 0.8223 | 0.8216 |
| instance recall | 0.4133 | 0.4667 |

Confirmed by moving the cache aside and re-running: the result reproduced the very first
eval (2026-09-16) to every printed digit. The mechanism inside `process_test` has not been
isolated — what matters operationally is the rule:

> **Compare only runs in the same regime.** Build the cache once, then compare
> cache-loading runs. If you delete `processed_0.2_test/` (which you must do whenever the
> raw PLYs change), re-run *every* model you intend to compare.

The numbers quoted here and in [`progress.md`](progress.md) are all cache-loading runs for
both backbones, so they are directly comparable.

---

## Quick reference — which numbers to report

For a forest-inventory result, the defensible minimum is:

1. **`mIoU without ground points`** — semantic quality where it is hard.
2. **`mMUCov` and `mMWCov` together** — instance quality for small and large trees.
3. **`mPrecision` / `mRecall`** — over- vs under-detection, rather than F1 alone.
4. **`meanPQ (things)` with its `meanSQ` and `meanRQ`** — never PQ on its own.

All four come from `Evaluation_<i>.txt`, per test plot. With only two test plots, report
them per plot rather than averaged — an average over two numbers hides more than it says.
