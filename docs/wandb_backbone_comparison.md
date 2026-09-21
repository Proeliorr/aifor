## MinkowskiEngine vs TorchSparse 1.4

Same ForAINet / PointGroup model, same training data, same 4-class scheme (`low_vegetation`, `ground`, `stem_points`, `live_branches`); the **only** difference is the sparse-convolution backbone. Every figure below is a *pooled* score — the confusion matrix and the instance matches are accumulated across plots and the metric is computed once, which is not the same as averaging the per-plot reports.

🟢 = MinkowskiEngine ahead 🟠 = TorchSparse ahead. Higher is better for every metric shown.

### Group A — held-out test plots (17 Sep)

plot_01_test + plot_14_test, pooled. Neither plot was seen in training or validation, so this is the comparison to quote. **MinkowskiEngine leads on 25 of the 27 metrics, TorchSparse on 1, 1 tied.**


**Semantic segmentation (4 classes)**

| Metric | MinkowskiEngine | TorchSparse 1.4 | Δ (Mink − TS) |
|---|---:|---:|---:|
| Overall accuracy | **0.9048** 🟢 | 0.8996 | +0.0052 |
| Mean accuracy | **0.8594** 🟢 | 0.8544 | +0.0051 |
| **mIoU** | **0.7768** 🟢 | 0.7662 | +0.0107 |
| Overall accuracy, ground excluded | **0.9101** 🟢 | 0.9040 | +0.0061 |
| Mean accuracy, ground excluded | **0.8494** 🟢 | 0.8439 | +0.0055 |
| **mIoU, ground excluded** | **0.7734** 🟢 | 0.7602 | +0.0132 |

**Per-class IoU**

| Metric | MinkowskiEngine | TorchSparse 1.4 | Δ (Mink − TS) |
|---|---:|---:|---:|
| 1 low_vegetation | **0.7171** 🟢 | 0.6913 | +0.0258 |
| 2 ground | **0.7872** 🟢 | 0.7839 | +0.0033 |
| 3 stem_points *(thing)* | **0.6952** 🟢 | 0.6849 | +0.0102 |
| 4 live_branches *(thing)* | **0.9079** 🟢 | 0.9044 | +0.0035 |

**Binary segmentation (tree / not-tree)**

| Metric | MinkowskiEngine | TorchSparse 1.4 | Δ (Mink − TS) |
|---|---:|---:|---:|
| Overall accuracy | **0.9837** 🟢 | 0.9785 | +0.0051 |
| Mean accuracy | **0.9807** 🟢 | 0.9770 | +0.0038 |
| IoU, not-tree | **0.9430** 🟢 | 0.9263 | +0.0167 |
| IoU, tree | **0.9776** 🟢 | 0.9706 | +0.0070 |
| **mIoU** | **0.9603** 🟢 | 0.9484 | +0.0119 |

**Instance segmentation (individual trees)**

| Metric | MinkowskiEngine | TorchSparse 1.4 | Δ (Mink − TS) |
|---|---:|---:|---:|
| MUCov | **0.3572** 🟢 | 0.2976 | +0.0596 |
| MWCov | **0.4194** 🟢 | 0.3694 | +0.0500 |
| Precision | **0.5055** 🟢 | 0.4935 | +0.0120 |
| Recall | **0.2383** 🟢 | 0.1959 | +0.0425 |
| **F1** | **0.3239** 🟢 | 0.2804 | +0.0435 |

**Panoptic quality**

| Metric | MinkowskiEngine | TorchSparse 1.4 | Δ (Mink − TS) |
|---|---:|---:|---:|
| RQ (things) | **0.3239** 🟢 | 0.2804 | +0.0435 |
| SQ (things) | 0.7321 | **0.7553** 🟠 | -0.0231 |
| **PQ (things)** | **0.2372** 🟢 | 0.2118 | +0.0254 |
| RQ (stuff) | 1.0000 | 1.0000 | 0 |
| SQ (stuff) | **0.9430** 🟢 | 0.9263 | +0.0167 |
| PQ (stuff) | **0.9430** 🟢 | 0.9263 | +0.0167 |
| mean PQ (all) | **0.5901** 🟢 | 0.5690 | +0.0210 |

### Group B — all three evaluation plots (18 Sep)

plot_01_test + plot_14_test + plot_11_val, pooled. plot_11 was training's validation plot, so these numbers are not a clean held-out measure — they are here for completeness. **MinkowskiEngine leads on 24 of the 27 metrics, TorchSparse on 2, 1 tied.**


**Semantic segmentation (4 classes)**

| Metric | MinkowskiEngine | TorchSparse 1.4 | Δ (Mink − TS) |
|---|---:|---:|---:|
| Overall accuracy | **0.8868** 🟢 | 0.8832 | +0.0037 |
| Mean accuracy | **0.8342** 🟢 | 0.8302 | +0.0040 |
| **mIoU** | **0.7278** 🟢 | 0.7215 | +0.0063 |
| Overall accuracy, ground excluded | **0.8891** 🟢 | 0.8844 | +0.0047 |
| Mean accuracy, ground excluded | **0.8513** 🟢 | 0.8453 | +0.0060 |
| **mIoU, ground excluded** | **0.7365** 🟢 | 0.7268 | +0.0097 |

**Per-class IoU**

| Metric | MinkowskiEngine | TorchSparse 1.4 | Δ (Mink − TS) |
|---|---:|---:|---:|
| 1 low_vegetation | **0.5926** 🟢 | 0.5779 | +0.0147 |
| 2 ground | 0.7018 | **0.7057** 🟠 | -0.0039 |
| 3 stem_points *(thing)* | **0.6974** 🟢 | 0.6869 | +0.0105 |
| 4 live_branches *(thing)* | **0.9194** 🟢 | 0.9155 | +0.0039 |

**Binary segmentation (tree / not-tree)**

| Metric | MinkowskiEngine | TorchSparse 1.4 | Δ (Mink − TS) |
|---|---:|---:|---:|
| Overall accuracy | **0.9732** 🟢 | 0.9694 | +0.0039 |
| Mean accuracy | **0.9820** 🟢 | 0.9784 | +0.0037 |
| IoU, not-tree | **0.9058** 🟢 | 0.8934 | +0.0125 |
| IoU, tree | **0.9789** 🟢 | 0.9736 | +0.0053 |
| **mIoU** | **0.9424** 🟢 | 0.9335 | +0.0089 |

**Instance segmentation (individual trees)**

| Metric | MinkowskiEngine | TorchSparse 1.4 | Δ (Mink − TS) |
|---|---:|---:|---:|
| MUCov | **0.3535** 🟢 | 0.3144 | +0.0390 |
| MWCov | **0.4041** 🟢 | 0.3702 | +0.0339 |
| Precision | **0.4771** 🟢 | 0.4737 | +0.0034 |
| Recall | **0.2332** 🟢 | 0.2009 | +0.0323 |
| **F1** | **0.3133** 🟢 | 0.2821 | +0.0311 |

**Panoptic quality**

| Metric | MinkowskiEngine | TorchSparse 1.4 | Δ (Mink − TS) |
|---|---:|---:|---:|
| RQ (things) | **0.3133** 🟢 | 0.2821 | +0.0311 |
| SQ (things) | 0.7180 | **0.7259** 🟠 | -0.0079 |
| **PQ (things)** | **0.2249** 🟢 | 0.2048 | +0.0201 |
| RQ (stuff) | 1.0000 | 1.0000 | 0 |
| SQ (stuff) | **0.9058** 🟢 | 0.8934 | +0.0125 |
| PQ (stuff) | **0.9058** 🟢 | 0.8934 | +0.0125 |
| mean PQ (all) | **0.5654** 🟢 | 0.5491 | +0.0163 |

### Reading the numbers

- **Semantic segmentation is close.** Roughly one point of mIoU separates the backbones on the held-out plots — both learned essentially the same point classifier.
- **Instance segmentation is where they part.** MinkowskiEngine recovers noticeably more trees (recall, F1, coverage), which is what drives the PQ (things) gap.
- **The only instance metric TorchSparse wins is SQ (things)** — segmentation *quality* of the instances it did find. It matches fewer trees, but the ones it matches overlap the ground truth slightly better. RQ × SQ = PQ, and the RQ gap dominates. (In Group B it also edges out `ground` IoU.)
- **RQ (stuff) is 1.0 for both** by construction: there is a single stuff region per plot and both models find it, so the stuff column really only reports its IoU.
- **These differences are not evaluation noise.** Re-running the same checkpoint through the same evaluation path reproduced the numbers exactly. They are, however, one *training* run per backbone, so what is compared is these two trained models — not the two libraries in general.
