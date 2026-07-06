# misc/ — standalone tools outside the main pipeline

Tools kept here are **not part of the main pipeline**. The project's main approach is
**semantic segmentation resolved at whole-plot level**, so per-tree data is not needed
there — but these tools may be useful for other (tree-level) approaches later.

---

## `tree_splitter.py` — split plots into per-tree numpy files

After the 3DFin batch (Step 1 of the main pipeline), each plot folder holds one
instance-segmented cloud (`<plot>_tree_ID_dist_axes.las`). This tool turns it into
**one `.npy` file per tree**, ready for per-tree processing:

1. Points are **sorted by `tree_ID` ascending** (the original labels are sparse and
   non-sequential — e.g. plot_01 has 119 unique IDs spanning 0…630619; **id 0 is the
   first tree**, 0-based numbering).
2. Each distinct `tree_ID` gets a new ordinal **`2tree_ID` = 1, 2, 3, …** (so
   `tree_ID=0` → `2tree_ID=1`).
3. The cloud is split on `2tree_ID` and each tree is saved as
   `<plot>_<NNN>.npy` (zero-padded by `digits`, default 3 → `plot_01_001.npy`),
   **in the same plot folder** as the input `.las`.

### How to run

Run from the `assignment/` folder, inside the `aifor` env. The tool has its own config,
[`misc/conf/config.yaml`](conf/config.yaml); overrides keep the `tree_splitter.` prefix:

```bash
# All plots
mamba run -n aifor python misc/tree_splitter.py

# Dry run / a single plot / custom columns
mamba run -n aifor python misc/tree_splitter.py tree_splitter.dry_run=true
mamba run -n aifor python misc/tree_splitter.py tree_splitter.plots=[plot_01]
mamba run -n aifor python misc/tree_splitter.py tree_splitter.columns=[x,y,z,Class,2tree_ID]
```

### Key config keys (`tree_splitter:` section)

| Key | Meaning |
|---|---|
| `input_dir` | Root holding one subfolder per plot (the 3DFin output tree). |
| `columns` | Columns stored in each array (default `[x, y, z, Class]`, shape `[N, 4]` float64; `Class` = ground-truth semantic label). Any LAS dimension plus `tree_ID` (original label) and `2tree_ID` (the ordinal). Unknown names are skipped with a warning. |
| `digits` | Zero-padding of the file suffix (3 → `_001`). |
| `overwrite` | `false` = skip plots whose `<plot>_001.npy` already exists. |
| `plots` / `dry_run` / `continue_on_error` | Same semantics as in the main pipeline. |

Each run also creates its own Hydra dir under `outputs/<date>/<time>/` with a
`tree_splitter.log`.

---

## `Points2ForAINet.py` — generic LAS/LAZ ↔ PLY converter with offset management

The original standalone argparse tool for adapting arbitrary point clouds to ForAINet:
converts between formats, subtracts per-file coordinate offsets (min x/y/z) so all
coordinates become non-negative, stores them in a single `offset.yml` (keyed by absolute
file paths), and restores original coordinates after classification.

Its plot-preprocessing role has been **superseded by the pipeline's Stage 2**
(`forainet_prep.py` + `pipeline/forainet_prep.py`), which works per plot, fills the real
`semantic_seg`/`treeID` fields from the 3DFin output, and stores per-plot offset files.
The script is kept here for ad-hoc conversions of files outside the plot layout:

```bash
mamba run -n aifor python misc/Points2ForAINet.py --help
```
