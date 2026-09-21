# Results

I trained two models: the same ForAINet network on the same data with the same settings on
the same machine, differing **only** in the sparse-convolution library underneath —
MinkowskiEngine (the original) and TorchSparse 1.4. Both ran for 99 epochs on a rented
NVIDIA A100 80 GB. Everything below is measured; nothing is estimated.

---

## 1. Training

Each chart carries two lines, one per backbone, and the first thing to notice is how hard
they are to tell apart. The semantic loss falls steeply over the first few epochs and
then flattens; validation mIoU climbs to roughly 65 and hovers there — for **both** models,
within a fraction of a point of each other all the way to the end:

| Tracked during training (subsampled) | MinkowskiEngine | TorchSparse 1.4 |
|---|---|---|
| Final validation mIoU | 65.09 | 64.99 |
| Final test mIoU | 70.03 | 69.71 |
| Final test F1 | 0.486 | 0.468 |

That near-identity is itself a result: swapping the sparse-convolution library changed how
fast the network trained, not what it learned.

The second thing to notice is that the instance metrics — mean average precision, mean
recall, the offset and score losses — sit flat at zero for the first thirty epochs on both
lines, and only come alive at epoch 31. That is by design. A setting called
`prepare_epoch: 30` gates the entire instance branch: for thirty epochs the network learns
only to classify points, and only then does it begin grouping them into individual trees.
The side effect is expensive, and section 4 returns to it — a large part of the codebase
does not execute until thirty epochs of paid GPU time have been spent.

It also has a practical consequence visible in the curves. TorchSparse's best validation
mIoU falls at **epoch 22**, before the instance branch had switched on at all, so that
checkpoint's tree-scoring network was still untrained. The final weights were therefore used
for both models rather than each one's nominal best.

The per-epoch cost is the clearest speed result the project produced:

| Phase | MinkowskiEngine | TorchSparse 1.4 |
|---|---|---|
| Epochs 1–30 (backbone and heads only) | 1.04 s/iteration · 3 min 16 s per epoch | 1.26 s/iteration · 3 min 57 s per epoch |
| Epoch 31 onward (instance scorer active) | 13.70 s/iteration · ~44 min per epoch | 27.8–29.6 s/iteration · ~1 h 32 min per epoch |

**TorchSparse was 1.2× slower on the backbone and 2.1× slower once the scorer ran** — the
opposite of the expectation that motivated trying it. The cause is the shape of the workload
rather than a defect in either library. The published claim that TorchSparse is faster
benchmarks it against an older MinkowskiEngine on large, spatially coherent scenes; here the
hot path is the opposite, as the scorer receives hundreds of small, spatially *disjoint*
clusters in a freshly built sparse tensor on every iteration. That is close to the worst
case for the hash-based lookups TorchSparse relies on, and the per-cluster matrix
multiplications are too small to hide the cost of launching them.

One caveat on reading the charts: the figures logged during training are running averages
over subsampled cylinders, which makes them right for watching *progress* and wrong for
quoting as a *result*. Sections 2 and 3 use full-resolution evaluation instead.

---

## 2. MinkowskiEngine versus TorchSparse

Both models were evaluated on the two held-out test plots — 41,534,130 points — with the
results pooled into a single confusion matrix and a single instance matching, rather than
averaged per plot.

| Metric | MinkowskiEngine | TorchSparse 1.4 |
|---|---|---|
| Overall accuracy | **0.9048** | 0.8996 |
| Mean accuracy | **0.8594** | 0.8544 |
| **mIoU** | **0.7768** | 0.7662 |
| mIoU excluding ground | **0.7734** | 0.7602 |
| Binary mIoU (tree / not-tree) | **0.9603** | 0.9484 |
| Instance precision | **0.5055** | 0.4935 |
| Instance recall | **0.2383** | 0.1959 |
| **Instance F1** | **0.3239** | 0.2804 |
| Coverage, unweighted / weighted | **0.357 / 0.419** | 0.298 / 0.369 |
| Segmentation quality, things | 0.7321 | **0.7553** |
| **Panoptic quality, things** | **0.2372** | 0.2118 |

Per class, both models agree closely and rank the classes identically:

| Class IoU | MinkowskiEngine | TorchSparse 1.4 |
|---|---|---|
| low vegetation | **0.7171** | 0.6913 |
| ground | **0.7872** | 0.7839 |
| stem points | **0.6952** | 0.6849 |
| live branches | **0.9079** | 0.9044 |

**MinkowskiEngine leads on 25 of the 27 metrics compared.** TorchSparse wins exactly one
instance metric — *segmentation quality (things)* — and the interpretation is specific: it
matched fewer trees, but outlined the ones it did match slightly more tightly. Since
panoptic quality is recognition quality multiplied by segmentation quality, and the
recognition gap is the larger of the two, MinkowskiEngine still comes out ahead overall.

These differences are **not evaluation noise**. Three evaluations of the same checkpoint
produced byte-identical reports, for both backbones. One caveat applies: the numbers shift
slightly depending on whether the run built the preprocessing cache or loaded an existing
one — two regimes, each perfectly reproducible. Every figure above comes from cache-loading
runs for both models, so they are directly comparable.

The full set of tables — every metric the evaluation emits, and a second grouping that also
folds in the validation plot — is given in the detailed backbone comparison.

**Conclusion on the backbone swap:** it changed nothing that matters and cost speed. The
original backbone stays the default.

---

## 3. Comparison with the published results

The ForAINet paper reports its configurations in Table 4. I trained with the paper's
**basic setting**, so that is the row to compare against.

The two evaluations use different *names* for the same quantities, and it is worth showing
that they really are the same before putting them in one table. The paper reports
completeness 79.3 % and commission error 21.2 %, implying a precision of 78.8 %; combining
those as an F1 score gives 79.05 %, which is exactly the 79.0 the paper reports. So
*completeness* is recall, *commission error* is one minus precision, and *F-score* is F1 —
the same three numbers my evaluation prints.

| Metric (%) | ForAINet paper<br>(basic setting) | This work —<br>MinkowskiEngine | This work —<br>TorchSparse |
|---|---|---|---|
| Semantic — overall accuracy | **92.6** | 90.5 | 90.0 |
| Semantic — mean accuracy | 81.2 | **85.9** | 85.4 |
| Semantic — **mIoU** | 73.0 | **77.7** | 76.6 |
| Instance — completeness (recall) | **79.3** | 23.8 | 19.6 |
| Instance — omission error | **20.8** | 76.2 | 80.4 |
| Instance — commission error | **21.2** | 49.5 | 50.6 |
| Instance — **F-score** | **79.0** | 32.4 | 28.0 |
| Coverage | **77.0** | 35.7 / 41.9 | 29.8 / 36.9 |

<sub>Bold marks the better value; for the two *error* rows, lower is better. Coverage is
given as unweighted / weighted, since my evaluation distinguishes the two where the paper
reports a single figure. For context, the paper's best configuration (+ TreeMix) reaches an
F-score of 85.1 — I compare against the basic setting because that is what I trained.</sub>

Four things follow.

**The semantic half transferred, and then some.** An mIoU of 77.7 against the paper's 73.0,
and mean accuracy of 85.9 against 81.2, say that the network learned to label terrestrial
LiDAR points at least as well as it labels airborne ones. This deserves one honest caveat:
my mIoU averages over **four** classes and theirs over five, and removing a class changes
the denominator. "Comparable" is the defensible word; "better" is not.

**The instance half did not transfer.** An F-score of 32.4 against 79.0, and coverage of
roughly half. This is the central finding of the project, and it is a result rather than a
defect — the model was applied outside the conditions it was designed for, and the part that
depends most on those conditions is the part that broke.

**The most likely cause is the sensor, not the code.** ForAINet's instance branch predicts,
for every point, an offset vector toward its own tree's stem, clusters the shifted points,
and scores the resulting clusters. That geometry is natural for airborne scanning, where a
crown is well sampled from above and its stem lies directly beneath it. A terrestrial scan
inverts it: stems are densely sampled, crowns are occluded and interleaved, and adjacent
trees genuinely touch. One observation points the same way — the training-time tracker
reported test F1 around 0.47–0.48 over subsampled cylinders against roughly 0.32 once
full-resolution blocks are merged, placing part of the loss in the merge step rather than in
the network.

**Label provenance is a second candidate, and I cannot separate it from the first.**
SegmentedForests supplies semantic labels only, so the tree identifiers the instance branch
trains on and is scored against were generated by 3DFin rather than delineated by hand. The
paper's instance figures are measured against manual annotation. An unknown share of the gap
is therefore the difficulty of the data, and an unknown share is the quality of the labels;
telling them apart would need a manually delineated subset to test against, which this
project does not have. The semantic comparison above is not affected — those labels are the
dataset's own.

These are inferences, not measurements: I did not re-run ForAINet on FOR-Instance, so the
paper's numbers and mine were produced on different data by different people.

---

## 4. The six biggest challenges

**1. The label scheme had to be designed, and it was not one-to-one.** Sixteen source
classes had to become four, and one of the framework's own classes had no counterpart in the
new data at all. My first check, on a single plot, looked deceptively tidy — individual
plots use very different subsets of the sixteen values, so only a check against all fourteen
revealed the real vocabulary. I therefore made the mapping strict: an unmapped value stops
the run instead of defaulting to something plausible.

**2. A silent convention mismatch destroyed the entire instance task.** The instance
identifiers produced by 3DFin follow the opposite convention to ForAINet's, marking ground
points with the nearest tree's id. ForAINet's own filter then rejected every contaminated
instance — zero of forty-two on the plot where I found it. Nothing crashed, nothing warned;
training would have completed normally and learned instance segmentation from nothing at
all. Fixing it meant clearing the id on 46.6 % of all points.

**3. Data volume dictated the architecture of the pipeline.** 4.4 GB of input expands to
46 GB of intermediates and 29 GB of uncompressed point clouds — the format the model reads
costs exactly 37 bytes per point. Processing plot by plot and deleting each intermediate as
soon as it was consumed brought peak disk use from 108.6 GB to 34.3 GB, and exporting
compressed cut the upload from 29.2 GB to 4.3 GB. Concurrency had to be bounded by memory
rather than by CPU count, since a single plot can need nearly 40 GB of RAM.

**4. The development machine and the container disagreed, and only the container's opinion
mattered.** A YAML feature that is perfectly valid under the configuration library installed
locally is a parse error under the older version pinned inside the training image. Because
the failure happens while *reading* the config, both backbones died with a message that
named neither of them. The same shape of problem appeared twice more: the image's LiDAR
reader had no backend for the compressed format, and its NumPy version had removed functions
the framework still called. Every local check had passed.

**5. One configuration value hid three separate crashes behind thirty epochs of paid GPU
time.** Because the instance branch is gated until epoch 31, a removed NumPy alias and two
distinct TorchSparse failures each surfaced only after roughly seven hours of A100 time.
This is what made me build a pre-flight gate that runs the full chain cheaply before any
long run — and that gate itself once reported a **false pass**, because a piped command
returned the exit status of the wrong process. A gate that green-lights a broken run is
worse than no gate, so I changed it to require positive evidence from the log rather than an
exit code.

**6. Making the comparison trustworthy was a project of its own.** The script that produces
the final pooled report was still written for five classes and, as shipped, could not run at
all — I had to repair five separate faults before it produced a number. Evaluation turned
out to be perfectly deterministic, but with two distinct regimes depending on whether a
preprocessing cache already existed, which meant I could only compare runs from the same
regime. And a claim I had made earlier, that instance metrics varied between runs, I had to
retract once I actually measured it.
