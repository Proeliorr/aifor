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

Hydra run directory: `${checkpoint_dir}/eval/${now:%Y-%m-%d_%H-%M-%S}`. Per test file `i`:

| File | Contents |
|---|---|
| `Evaluation_<i>.txt` | the report from §5 — **appended** |
| `Semantic_results_forEval_<i>.ply` | predicted vs GT semantic labels, full resolution |
| `Instance_Results_forEval_<i>.ply` | predicted instance ids, full resolution |
| `Instance_results_withColor_<i>.ply` | the same, random colour per instance — for looking at |
| `Instance_subsample_<i>.ply` | instance ids at subsampled resolution |
| `vote1regular_<i>.ply` | per-file voting output (`tracker_options.ply_output`) |
| `vote1regularfull.ply`, `Instance_Offset_results_forEval.ply` | run-level, **not** per-file, so they are overwritten each file |

The `_withColor` file is the one to open in a viewer; the others carry raw ids.

## 8. Running it on SegmentedForests

`conf/eval.yaml` ships pointed at the original author's machine and **cannot run as-is**.
Three keys must change:

| Key | Ships as | Needs to be |
|---|---|---|
| `checkpoint_dir` | empty | the training run's output directory (the one holding `PointGroup-PAPER.pt`) |
| `data.fold` | eleven absolute paths under `/cluster/work/igp_psr/binbin/…` | absolute paths to **your** test PLYs |
| `weight_name` | `"latest"` | `"latest"` for the final epoch, or a metric name (`miou`, `macc`) for the best checkpoint under it |

For our split that means the two plots Stage 2 marked `_test`:

```yaml
data:
  fold: ['<abs>/treeinsfused/raw/SegmentedForests/plot_01_test.ply',
         '<abs>/treeinsfused/raw/SegmentedForests/plot_14_test.ply']
```

The `_test` suffix is not decoration — ForAINet reads the split from the file name
(`name[-8:-4] == "test"`), so a renamed file changes which set it belongs to. See the
README's *Train / val / test* section.

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

---

## Quick reference — which numbers to report

For a forest-inventory result, the defensible minimum is:

1. **`mIoU without ground points`** — semantic quality where it is hard.
2. **`mMUCov` and `mMWCov` together** — instance quality for small and large trees.
3. **`mPrecision` / `mRecall`** — over- vs under-detection, rather than F1 alone.
4. **`meanPQ (things)` with its `meanSQ` and `meanRQ`** — never PQ on its own.

All four come from `Evaluation_<i>.txt`, per test plot. With only two test plots, report
them per plot rather than averaged — an average over two numbers hides more than it says.
