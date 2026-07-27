"""Hydra entry point: unify semantic class labels and export the cloud + a .json sidecar.

Run inside the ``aifor`` mamba environment, for example::

    mamba run -n aifor python class_unifier.py                                # all plots
    mamba run -n aifor python class_unifier.py class_unifier.dry_run=true     # print plan only
    mamba run -n aifor python class_unifier.py class_unifier.plots=[plot_01]  # a single plot
    mamba run -n aifor python class_unifier.py class_unifier.output_format=both

Output is a ``.ply`` by default — that is the format to open in
``misc/view_split_point_cloud.ipynb``, because the viewer only draws class colours,
a legend and unique counts for INTEGER columns and a PLY keeps each column's dtype.
``output_format=npy`` writes the plain float64 matrix instead (load-anywhere, but
every label column then looks continuous); ``both`` writes the pair.
All pipeline stages share one config file, ``conf/config.yaml``; this script reads its
``class_unifier:`` section. Override any key with that prefix, e.g.
``class_unifier.unified_field=labels_v2``.

The mapping itself lives in that file's shared ``classes:`` block, which Stage 2
(``forainet_prep.py``) interpolates too — one definition, so the ``.npy`` exports and
the PLYs ForAINet trains on cannot drift apart.

Two things to know:

* **Change the scheme by editing ``conf/config.yaml``, not on the CLI.** The map's
  keys are integers, which Hydra's override parser handles badly:
  ``classes.class_map.5=1`` and ``classes.class_map={99: 0}`` both fail with
  "Key ... is not in struct"; ``++classes.class_map={99: 0}`` merges; and
  ``class_unifier.class_map={...}`` *replaces* this stage's map, severing the link to
  the shared block so Stage 2 keeps the old one and the two disagree.
* The map must cover **every** class present in the data — an unmapped value fails
  that plot instead of being silently kept or forced to a default.
"""
import os

import hydra
from hydra_zen import zen

from pipeline.class_unifier import unify_classes

# Absolute path to the conf/ directory next to this file, so the entry script can be run
# from anywhere (Hydra also accepts an absolute config_path).
CONF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conf")


@hydra.main(config_path=CONF_DIR, config_name="config", version_base="1.3")
def main(cfg):
    # config.yaml holds one section per stage; run this stage from its `class_unifier:`
    # section. zen() maps the section's keys onto unify_classes' parameters (and converts
    # OmegaConf containers to plain Python), falling back to defaults for anything absent.
    zen(unify_classes)(cfg.class_unifier)


if __name__ == "__main__":
    main()
