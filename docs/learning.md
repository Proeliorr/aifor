

# learning.md — Machine Learning, from A to Z (a living glossary)

A personal, beginner-friendly notebook for learning how ML models are built — grounded
in **this very repo**. This project is a real, working deep-learning system (ForAINet:
it looks at a 3D LiDAR scan of a forest and labels every point, then separates it into
individual trees). That makes it a great place to see textbook concepts in real code.

**This file is meant to grow.** Whenever a new term comes up, add an entry. Keep the
style beginner-first: explain the idea in plain words, *then* point at where it lives in
this codebase, *then* link to one good resource.

### How each glossary entry is written

```
### Term
**In plain English:** what it means, with an analogy if that helps.
**In this project:** where it actually shows up (clickable file link + config key).
**Learn more:** one beginner-friendly link.
```

(If a term is pure theory and doesn't appear in this repo yet, just skip the
"In this project" line — that's fine.)

---

## Suggested path (if you're starting cold)

You don't have to read the glossary top to bottom. If you want an order, roughly:

1. [What is a model](#model) → [Features vs labels](#features-vs-labels) → [Training vs inference](#training-vs-inference)
2. [Neural network](#neural-network) → [Loss function](#loss-function) → [Gradient descent & backpropagation](#gradient-descent--backpropagation)
3. [Epoch](#epoch), [Batch](#batch--batch-size), [Optimizer / Adam](#optimizer--adam), [Learning rate](#learning-rate--lr-scheduler)
4. [Overfitting](#overfitting--generalization) → [Train/val/test split](#trainvaltest-split) → how we measure success: [Accuracy](#accuracy), [IoU / mIoU](#iou--miou)
5. What our data is: [Point cloud](#point-cloud) → [Voxelization](#voxelization--gridsampling3d) → [Data augmentation](#data-augmentation)
6. What our model actually does: [Semantic vs instance vs panoptic](#semantic-vs-instance-vs-panoptic-segmentation) → [PointGroup](#pointgroup)

Then dip into the [Resources](#resources) and try the [Hands-on](#hands-on-in-this-repo)
section to make it concrete.

---

## Glossary

### Fundamentals

#### Model
**In plain English:** a "machine" that takes some input and produces a prediction. It
has internal dials (called **parameters** or **weights**) that get tuned during
training. Before training, the dials are random and the model is useless; after
training, the dials are set so the machine gives good answers.
**In this project:** our model takes 3D points and predicts, for each point, "what is
this?" (ground, stem, branch…) and "which tree does it belong to?". It's defined in
[PointGroup3heads.py](ForAINet/PointCloudSegmentation/torch_points3d/models/panoptic/PointGroup3heads.py).
**Learn more:** [Google ML Crash Course — Framing](https://developers.google.com/machine-learning/crash-course/framing/ml-terminology)

#### Features vs labels
**In plain English:** **features** are the inputs you show the model (the "questions");
**labels** are the correct answers you want it to learn to produce. Training = showing
lots of (features → label) pairs until the model can guess the label on its own.
**In this project:** features are each point's coordinates `x, y, z` (plus extras like
height and intensity); labels are `semantic_seg` (the class) and `treeID` (which tree).
Those exact field names are produced by Stage 2 in
[pipeline/forainet_prep.py](pipeline/forainet_prep.py) and consumed by ForAINet.
**Learn more:** [Google ML Crash Course — Labels and Features](https://developers.google.com/machine-learning/crash-course/framing/ml-terminology)

#### Training vs inference
**In plain English:** **training** is the learning phase — you feed labelled examples
and the model adjusts its dials. **Inference** (a.k.a. prediction/evaluation) is using
the finished model on new, unseen data. Training is slow and done once; inference is
fast and done often.
**In this project:** training is run with
[train.py](ForAINet/PointCloudSegmentation/train.py); inference/prediction with
[eval.py](ForAINet/PointCloudSegmentation/eval.py). After inference, our Stage 2 can
convert the predicted files back to real-world coordinates (`mode=restore`).
**Learn more:** [PyTorch — training vs eval mode](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)

#### Train/val/test split
**In plain English:** you split your data into three piles. **Train** = the model
learns from it. **Validation (val)** = you peek during training to tune settings and
catch problems. **Test** = held back until the very end, to get an honest score on data
the model has truly never seen. Mixing these up is a classic way to fool yourself.
**In this project:** the dataset config
[conf/data/panoptic/treeins_set1.yaml](ForAINet/PointCloudSegmentation/conf/data/panoptic/treeins_set1.yaml)
defines which forest plots go into train / val / test.
**Learn more:** [StatQuest — Training, Validation, Testing (YouTube)](https://www.youtube.com/watch?v=fSytzGwwBVw)

#### Overfitting & generalization
**In plain English:** **overfitting** = the model memorises the training examples
instead of learning the general pattern, so it aces training but flops on new data
(like a student who memorised past exam answers). **Generalization** is the opposite —
doing well on data it's never seen, which is the whole point. Data augmentation,
validation, and not training too long all help.
**In this project:** the training augmentations in
[transforms.py](ForAINet/PointCloudSegmentation/torch_points3d/core/data_transform/transforms.py)
(random rotations, noise, flips) exist largely to fight overfitting.
**Learn more:** [Google ML Crash Course — Overfitting](https://developers.google.com/machine-learning/crash-course/overfitting/overfitting)

#### Neural network
**In plain English:** a model built from many simple units ("neurons") arranged in
layers. Each neuron multiplies its inputs by weights, adds them up, and passes the
result through an **activation function**. Stacking layers lets the network learn very
complex patterns. "Deep learning" just means a neural network with many layers.
**In this project:** ForAINet's network is a **sparse-convolution U-Net** backbone
feeding three small output "heads" — see
[PointGroup3heads.py](ForAINet/PointCloudSegmentation/torch_points3d/models/panoptic/PointGroup3heads.py).
**Learn more:** [3Blue1Brown — But what is a neural network? (YouTube)](https://www.youtube.com/watch?v=aircAruvnKk)

#### Activation function
**In plain English:** a small non-linear function applied inside each neuron (common
ones: ReLU, sigmoid). Without it, stacking layers would collapse into one big linear
formula and the network couldn't learn curved/complex patterns. It's the "spark" that
makes deep networks powerful.
**Learn more:** [3Blue1Brown neural network series](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)

#### Hyperparameter
**In plain English:** a setting *you* choose before training (not learned by the
model). Examples: how many layers, the learning rate, the batch size, how many epochs.
Tuning these well is a big part of practical ML. Contrast with **parameters/weights**,
which the model *does* learn.
**In this project:** almost every hyperparameter lives in a YAML config file under
[conf/](ForAINet/PointCloudSegmentation/conf/), e.g.
[conf/training/treeins_set1.yaml](ForAINet/PointCloudSegmentation/conf/training/treeins_set1.yaml)
(`epochs: 150`, `batch_size: 4`, `base_lr: 0.001`).
**Learn more:** [Google ML Crash Course — Hyperparameters](https://developers.google.com/machine-learning/crash-course/reducing-loss/learning-rate)

---

### Training (how a model actually learns)

#### Loss function
**In plain English:** a single number that measures how *wrong* the model's current
predictions are. Training = repeatedly nudging the dials to make this number smaller.
Low loss ≈ good predictions. Choosing the right loss for your task matters a lot.
**In this project:** ForAINet does several jobs at once, so it adds up several losses
(one for the class prediction, ones for pushing points toward their tree centre, etc.)
into one total — defined in
[panoptic_losses.py](ForAINet/PointCloudSegmentation/torch_points3d/core/losses/panoptic_losses.py).
See also [multi-task learning](#multi-task-learning).
**Learn more:** [Google ML Crash Course — Loss](https://developers.google.com/machine-learning/crash-course/descending-into-ml/training-and-loss)

#### Gradient descent & backpropagation
**In plain English:** imagine the loss as a hilly landscape and you want the lowest
valley. **Gradient descent** is walking downhill one small step at a time.
**Backpropagation** is the efficient math trick that tells each of the millions of
dials which way (and how much) to move to go downhill. Together they are the engine of
learning in neural networks.
**In this project:** you'll see `self.loss.backward()` (backprop) inside the model's
training step — the framework handles the calculus for you.
**Learn more:** [3Blue1Brown — Gradient descent (YouTube)](https://www.youtube.com/watch?v=IHZwWFHWa-w)

#### Epoch
**In plain English:** one full pass over the entire training dataset. Models usually
need many epochs (dozens to hundreds) to learn well — each pass tweaks the dials a
little more.
**In this project:** the epoch loop lives in
[trainer.py](ForAINet/PointCloudSegmentation/torch_points3d/trainer.py) (`for epoch in
range(...)`); the number of epochs is set in
[conf/training/treeins_set1.yaml](ForAINet/PointCloudSegmentation/conf/training/treeins_set1.yaml)
(`epochs: 150`).
**Learn more:** [Epoch vs batch vs iteration (article)](https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/)

#### Batch & batch size
**In plain English:** instead of showing the model one example at a time (slow) or all
of them at once (won't fit in memory), we show it in small groups called **batches**.
**Batch size** is how many examples per group. Bigger batches = smoother but more
memory-hungry.
**In this project:** `batch_size: 4` in
[conf/training/treeins_set1.yaml](ForAINet/PointCloudSegmentation/conf/training/treeins_set1.yaml)
— four forest "chunks" are processed together at a time.
**Learn more:** [Batch vs epoch (article)](https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/)

#### Optimizer / Adam
**In plain English:** the **optimizer** is the algorithm that actually applies the
downhill steps from gradient descent. **Adam** is the most popular one — it adapts the
step size automatically and usually "just works," which is why it's a great default.
**In this project:** the training config uses `Adam` with `base_lr: 0.001` (see
[conf/training/treeins_set1.yaml](ForAINet/PointCloudSegmentation/conf/training/treeins_set1.yaml)).
**Learn more:** [Adam optimizer, explained (article)](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/)

#### Learning rate & LR scheduler
**In plain English:** the **learning rate** is how big each downhill step is. Too big =
you overshoot and never settle; too small = training crawls. A **scheduler** changes it
over time — usually starting larger and shrinking it, so you cover ground fast early
then fine-tune gently at the end.
**In this project:** scheduler options live in
[conf/lr_scheduler/](ForAINet/PointCloudSegmentation/conf/lr_scheduler/) (e.g.
`cosine.yaml`, `poly_lr.yaml`, `step.yaml`) — you pick one in the config.
**Learn more:** [Learning rate (Google ML Crash Course)](https://developers.google.com/machine-learning/crash-course/reducing-loss/learning-rate)

#### Checkpoint
**In plain English:** a saved snapshot of the model's dials at a moment in training.
Checkpoints let you stop and resume, and — importantly — keep the *best* version seen
so far (since a later epoch isn't always better). The final saved model is basically the
best checkpoint.
**In this project:** handled by
[model_checkpoint.py](ForAINet/PointCloudSegmentation/torch_points3d/metrics/model_checkpoint.py);
ForAINet ships a pretrained checkpoint `PointGroup-PAPER.pt` you can download instead of
training from scratch.
**Learn more:** [PyTorch — saving & loading models](https://pytorch.org/tutorials/beginner/saving_loading_models.html)

#### GPU / CUDA & mixed precision
**In plain English:** a **GPU** is a graphics card, but it's also brilliant at the many
parallel multiplications neural nets need — often 10–100× faster than a CPU. **CUDA** is
NVIDIA's toolkit for running code on the GPU. **Mixed precision** uses lower-precision
numbers where it's safe, to go faster and use less memory with (almost) no accuracy loss.
**In this project:** ForAINet needs an NVIDIA GPU; it pins `PyTorch ...+cu111` (CUDA
11.1) and uses `torch.cuda.amp` (automatic mixed precision) inside
[trainer.py](ForAINet/PointCloudSegmentation/torch_points3d/trainer.py).
**Learn more:** [PyTorch — CUDA basics](https://pytorch.org/docs/stable/notes/cuda.html)

#### Multi-task learning
**In plain English:** training one model to do several related jobs at once, sharing
most of its "brain." It's often better than separate models because the tasks help each
other learn shared structure — but you have to balance the tasks (usually by weighting
each task's loss).
**In this project:** ForAINet's network has **three heads** doing three jobs (classify
each point, predict an offset toward the tree centre, and produce an embedding) from one
shared backbone — hence the name
[PointGroup3heads.py](ForAINet/PointCloudSegmentation/torch_points3d/models/panoptic/PointGroup3heads.py).
Their losses are added with tunable weights.
**Learn more:** [An overview of multi-task learning (Ruder, blog)](https://www.ruder.io/multi-task/)

---

### Evaluation (how we grade a model)

#### Accuracy
**In plain English:** the fraction of predictions that are correct. Simple and
intuitive — but misleading when classes are imbalanced (if 95% of points are "ground,"
a lazy model that always says "ground" scores 95% while being useless). That's why we
also use the metrics below.
**Learn more:** [Accuracy, precision, recall (Google ML Crash Course)](https://developers.google.com/machine-learning/crash-course/classification/accuracy)

#### IoU / mIoU
**In plain English:** **IoU** (Intersection over Union) measures overlap between the
predicted region and the true region: overlap ÷ combined area. 1.0 = perfect, 0 = no
overlap. **mIoU** = the *mean* IoU averaged across all classes — the standard headline
score for segmentation, and fair to rare classes.
**In this project:** computed by the metrics tracker
[panoptic_tracker_pointgroup_treeins_partseg.py](ForAINet/PointCloudSegmentation/torch_points3d/metrics/panoptic_tracker_pointgroup_treeins_partseg.py).
**Learn more:** [Intersection over Union explained (article)](https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/)

#### Precision / recall / F1
**In plain English:** **precision** = of the things I flagged, how many were right (did
I cry wolf?). **Recall** = of the things I should have flagged, how many did I catch
(did I miss any?). They trade off, so **F1** combines them into one balanced score.
**In this project:** used to score *individual tree* detection in the same tracker file
above (e.g. "did we find each tree, and not invent extra ones?").
**Learn more:** [Precision & recall (Google ML Crash Course)](https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall)

#### mAP
**In plain English:** **mean Average Precision** — the standard score for
detection/instance tasks. Roughly: for each object you predicted, is it a good enough
match (by IoU) to a real object? mAP summarises precision across many confidence
thresholds into one number. Higher is better.
**In this project:** reported for tree instances (matching predicted trees to real trees
at an IoU threshold) in the tracker file above.
**Learn more:** [mAP explained (article)](https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173)

#### Confusion matrix
**In plain English:** a table showing, for each true class, how the model actually
labelled those items — the diagonal is correct, off-diagonal cells show *what got
confused with what*. Great for seeing *which* mistakes a model makes, not just how many.
**In this project:** accumulated during evaluation by the metrics tracker to compute
per-class IoU/accuracy.
**Learn more:** [Confusion matrix (StatQuest, YouTube)](https://www.youtube.com/watch?v=Kdsp6soqA7o)

---

### Data (getting the world into the model)

#### Point cloud
**In plain English:** a set of 3D points — literally a big list of `(x, y, z)`
coordinates, often with extras like colour or laser intensity. It's how a LiDAR scanner
"sees": millions of dots in space forming the shape of trees, ground, buildings. Unlike
an image (a neat grid of pixels), a point cloud is unordered and irregular, which is why
it needs special models.
**In this project:** input plots are `.laz`/`.las` files in `SegmentedForests/
pointclouds/`; the whole pipeline exists to turn them into model-ready `.ply` files.
**Learn more:** [What is a point cloud? (article)](https://www.mdpi.com/2072-4292/12/11/1729) · [LiDAR basics (USGS)](https://www.usgs.gov/faqs/what-lidar-data-and-where-can-i-download-it)

#### Centering / normalization
**In plain English:** models learn better when inputs are in a small, consistent range
rather than huge raw numbers. **Centering** shifts the data so it starts near zero;
**normalization** more generally rescales inputs to a comfortable range. It's a
preparation step, not learning.
**In this project:** Stage 2 subtracts the minimum of each axis so all coordinates
become ≥ 0, and records the shift in a `*_offsets.yml` so it can be undone later — see
[pipeline/forainet_prep.py](pipeline/forainet_prep.py).
**Learn more:** [Why feature scaling matters (article)](https://machinelearningmastery.com/standardscaler-and-minmaxscaler-transforms-in-python/)

#### Voxelization / GridSampling3D
**In plain English:** a point cloud can have millions of irregular points. **Voxelizing**
snaps them onto a regular 3D grid of little cubes ("voxels"), keeping roughly one point
per cube. This makes the data smaller, evenly spaced, and digestible for the network —
like down-sampling a photo, but in 3D.
**In this project:** done by the `GridSampling3D` transform (e.g. a 0.2 m grid) inside
[transforms.py](ForAINet/PointCloudSegmentation/torch_points3d/core/data_transform/transforms.py).
**Learn more:** [Voxelization / voxel grids (Open3D docs)](http://www.open3d.org/docs/latest/tutorial/geometry/voxelization.html)

#### Data augmentation
**In plain English:** artificially expanding your training set by making altered copies
of your data — rotating, flipping, adding a little noise, rescaling. The model sees more
variety, so it generalises better and overfits less. The label usually stays the same
(a rotated tree is still a tree).
**In this project:** the training transforms in
[transforms.py](ForAINet/PointCloudSegmentation/torch_points3d/core/data_transform/transforms.py)
include random rotation, jitter (noise), scaling, and flips — plus fancy point-cloud
ones like elastic distortion and a "TreeMix" copy-paste augmentation.
**Learn more:** [A gentle intro to data augmentation (article)](https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/)

#### Dataloader / batching
**In plain English:** the **dataloader** is the conveyor belt that fetches examples from
disk, applies transforms, groups them into batches, and (often) does this in parallel on
several CPU workers so the GPU never waits. You rarely write this by hand — the framework
provides it.
**In this project:** built by torch-points3d's dataset classes (e.g.
[conf/data/panoptic/treeins_set1.yaml](ForAINet/PointCloudSegmentation/conf/data/panoptic/treeins_set1.yaml)
sets `num_workers`, and large plots are cut into 8 m-radius cylinders before batching).
**Learn more:** [PyTorch — Datasets & DataLoaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)

---

### This project's model (point-cloud specifics)

#### Semantic vs instance vs panoptic segmentation
**In plain English:** three levels of "labelling every point."
- **Semantic**: *what* is each point? (ground / stem / branch). All stems share one label.
- **Instance**: *which object*? (tree #1 vs tree #2) — separating individual things.
- **Panoptic**: both at once — every point gets a class *and*, if it's a "thing," an
  object ID. That's the hard, complete version, and it's exactly ForAINet's job.
**In this project:** the whole system is panoptic — output fields `semantic_seg` (class)
and `treeID` (instance). See the model
[PointGroup3heads.py](ForAINet/PointCloudSegmentation/torch_points3d/models/panoptic/PointGroup3heads.py).
**Learn more:** [Panoptic segmentation explained (article)](https://blog.roboflow.com/panoptic-segmentation/)

#### PointGroup
**In plain English:** a well-known method for instance segmentation on point clouds. The
idea: first predict, for every point, a little arrow (**offset**) pointing toward the
centre of the object it belongs to; move the points along their arrows so each object's
points bunch together; then simply **group** nearby points into instances. ForAINet
adapts this for trees.
**In this project:** it's literally the model —
[PointGroup3heads.py](ForAINet/PointCloudSegmentation/torch_points3d/models/panoptic/PointGroup3heads.py),
config name `PointGroup-PAPER`.
**Learn more:** [PointGroup paper (CVPR 2020)](https://arxiv.org/abs/2004.01658)

#### Sparse convolution + U-Net backbone
**In plain English:** a **convolution** is the core operation of vision networks — a
small filter slides over the data detecting patterns. **Sparse** convolution only does
work where points actually exist (most of 3D space is empty air), saving huge amounts of
compute. A **U-Net** is a backbone shape that shrinks the data down to capture the big
picture, then expands it back to full detail, with shortcuts across — great for
labelling every point.
**In this project:** the backbone is a sparse-conv U-Net built via
[minkowski.py](ForAINet/PointCloudSegmentation/torch_points3d/applications/minkowski.py)
(using the MinkowskiEngine library). It's the shared "body" feeding the three heads.
**Learn more:** [U-Net paper](https://arxiv.org/abs/1505.04597) · [MinkowskiEngine (sparse conv) docs](https://nvidia.github.io/MinkowskiEngine/)

#### Offset prediction
**In plain English:** one of the three heads. For each point it predicts a 3D arrow
pointing toward its object's centre. Add the arrow to the point's position and points of
the same tree pile up near a common centre — which makes them easy to group. It's the key
trick borrowed from PointGroup.
**In this project:** the `Offset` head in
[PointGroup3heads.py](ForAINet/PointCloudSegmentation/torch_points3d/models/panoptic/PointGroup3heads.py),
trained by an offset loss in
[panoptic_losses.py](ForAINet/PointCloudSegmentation/torch_points3d/core/losses/panoptic_losses.py).
**Learn more:** [PointGroup paper](https://arxiv.org/abs/2004.01658)

#### Embedding / discriminative loss
**In plain English:** an **embedding** is a short list of numbers the model invents to
describe each point, arranged so that points of the *same* object have similar numbers
and points of *different* objects have very different numbers. The **discriminative
loss** trains this: it "pulls" same-object points together and "pushes" different objects
apart. Then clustering those numbers separates the objects.
**In this project:** the `Embed` head (default 5 numbers per point) plus the
discriminative loss in
[panoptic_losses.py](ForAINet/PointCloudSegmentation/torch_points3d/core/losses/panoptic_losses.py).
**Learn more:** [Discriminative loss for instance segmentation (paper)](https://arxiv.org/abs/1708.02551)

#### Region-growing clustering
**In plain English:** a way to turn per-point predictions into whole objects: start from
a point, absorb its close neighbours, then their neighbours, and so on — like a stain
spreading — until the object is fully "grown." Do it repeatedly to find all objects. Used
after the offset step, when same-tree points are already bunched together.
**In this project:** ForAINet groups the shifted points into candidate trees using a
region-grow routine inside the model's forward pass.
**Learn more:** [Region growing (Wikipedia)](https://en.wikipedia.org/wiki/Region_growing)

#### ScoreNet
**In plain English:** after grouping produces many *candidate* objects, ScoreNet is a
small extra network that gives each candidate a confidence score ("how likely is this a
real, complete tree?"). Low-scoring junk candidates get filtered out. It's a
quality-control step.
**In this project:** the `Scorer*` modules in
[PointGroup3heads.py](ForAINet/PointCloudSegmentation/torch_points3d/models/panoptic/PointGroup3heads.py).
**Learn more:** [PointGroup paper (ScoreNet section)](https://arxiv.org/abs/2004.01658)

#### NMS (Non-Maximum Suppression)
**In plain English:** when a model proposes several overlapping detections of the *same*
object, NMS keeps the most confident one and deletes the duplicates. A clean-up step so
you report one tree, not five overlapping guesses at the same tree.
**In this project:** applied when turning scored candidates into final tree instances,
during evaluation in the tracker file.
**Learn more:** [Non-max suppression explained (article)](https://learnopencv.com/non-maximum-suppression-theory-and-implementation-in-pytorch/)

#### The 5 forest classes
**In plain English:** the specific labels this project predicts. Some are **"stuff"**
(background regions you don't count individually) and some are **"things"** (objects you
separate into instances).
**In this project:** `0 low_vegetation`, `1 ground` (stuff) and `2 stem_points`,
`3 live_branches`, `4 branches` (things — these get grouped into individual trees).
Defined in the dataset code under
[torch_points3d/datasets/](ForAINet/PointCloudSegmentation/torch_points3d/datasets/).
**Learn more:** see the repo's own [ARCHITECTURE_ANALYSIS.md](ForAINet/ARCHITECTURE_ANALYSIS.md).

---

### Tools & ecosystem

#### PyTorch
**In plain English:** the most popular open-source library for building and training
neural networks. It handles the tensor math, runs it on the GPU, and does
backpropagation automatically, so you describe *what* your model is and PyTorch works out
*how* to train it.
**In this project:** the whole ForAINet model is PyTorch (pinned to a CUDA 11.1 build).
**Learn more:** [PyTorch — 60-minute blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)

#### torch-points3d
**In plain English:** a library built *on top of* PyTorch specifically for deep learning
on 3D point clouds — it provides ready-made point-cloud models, datasets, transforms, and
training loops so you don't reinvent them.
**In this project:** ForAINet is a fork/extension of torch-points3d; the vendored
`torch_points3d/` package is the bulk of the ForAINet code.
**Learn more:** [torch-points3d docs](https://torch-points3d.readthedocs.io/)

#### Hydra config system
**In plain English:** a tool for managing settings via YAML files, and for overriding any
setting from the command line without editing files. It's why you can change an
experiment by typing `epochs=200` instead of hunting through code. Great for running many
experiment variants cleanly.
**In this project:** used **both** by ForAINet (all those `conf/*.yaml` files) *and* by
this repo's own pipeline (`conf/config.yaml`, overridden like
`forainet_prep.plots=[plot_02]`).
**Learn more:** [Hydra docs — getting started](https://hydra.cc/docs/intro/)

#### wandb / TensorBoard
**In plain English:** dashboards that record and plot what happens during training — loss
going down, metrics going up, learning-rate curves — so you can watch progress live and
compare runs. **TensorBoard** is local; **Weights & Biases (wandb)** is cloud-based.
**In this project:** ForAINet can log to either; toggled in its training config.
**Learn more:** [TensorBoard tutorial](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html) · [wandb quickstart](https://docs.wandb.ai/quickstart)

---

## Resources

### Start here (beginner, mostly free)
- **3Blue1Brown — Neural Networks** (YouTube series): the best visual intuition for how
  neural nets and gradient descent work. https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
- **StatQuest with Josh Starmer** (YouTube): friendly, no-nonsense explanations of ML
  concepts and metrics. https://www.youtube.com/c/joshstarmer
- **Google ML Crash Course**: short, practical, well-structured.
  https://developers.google.com/machine-learning/crash-course
- **Andrew Ng — Machine Learning Specialization / Deep Learning Specialization**
  (Coursera): the classic structured courses. https://www.coursera.org/specializations/machine-learning-introduction
- **fast.ai — Practical Deep Learning for Coders**: top-down, hands-on, code first.
  https://course.fast.ai/
- **PyTorch — 60-minute blitz**: your first hands-on PyTorch.
  https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html
- **d2l.ai — Dive into Deep Learning**: a free, interactive textbook with runnable code.
  https://d2l.ai/

### Go deeper (this project's territory: 3D / point clouds)
- **torch-points3d docs** — the framework ForAINet is built on.
  https://torch-points3d.readthedocs.io/
- **PointGroup** (the instance-segmentation method this model uses), CVPR 2020.
  https://arxiv.org/abs/2004.01658
- **PointNet++** and **KPConv** — foundational point-cloud networks worth knowing.
  https://arxiv.org/abs/1706.02413 · https://arxiv.org/abs/1904.08889
- **U-Net** — the backbone shape used for dense per-point labelling.
  https://arxiv.org/abs/1505.04597
- **ForAINet paper** — Xiang et al., *Automated forest inventory: analysis of
  high-density airborne LiDAR point clouds with 3D deep learning*, Remote Sensing of
  Environment, 2024. (The academic write-up of this exact model.)
- **This repo's own** [ForAINet/ARCHITECTURE_ANALYSIS.md](ForAINet/ARCHITECTURE_ANALYSIS.md)
  — a detailed narrative of the architecture, worth reading once you know the basics.

---

## Hands-on in this repo

Concrete, low-risk things to connect theory to practice (see [README.md](README.md) for
full instructions):

1. **See the pipeline without changing anything** — run Stage 2 in dry-run mode to watch
   what it *would* do:
   `cmd /c "mamba run -n aifor python forainet_prep.py forainet_prep.dry_run=true"`
2. **Read a real config** — open
   [conf/training/treeins_set1.yaml](ForAINet/PointCloudSegmentation/conf/training/treeins_set1.yaml)
   and find `epochs`, `batch_size`, and the optimizer. Match each to its glossary entry
   above.
3. **Find the epoch loop** — open
   [trainer.py](ForAINet/PointCloudSegmentation/torch_points3d/trainer.py) and locate the
   `for epoch in range(...)` line. That single loop is "training."
4. **See multi-task learning in one place** — skim
   [PointGroup3heads.py](ForAINet/PointCloudSegmentation/torch_points3d/models/panoptic/PointGroup3heads.py)
   for the three heads (`Semantic`, `Offset`, `Embed`).
5. **Read the architecture tour** — [ForAINet/ARCHITECTURE_ANALYSIS.md](ForAINet/ARCHITECTURE_ANALYSIS.md)
   once the terms above feel familiar.

---

*Keep adding entries as new terms come up — follow the template near the top of this
file so it stays consistent and easy to scan.*
