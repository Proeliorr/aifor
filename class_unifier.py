"""Hydra entry point: unify semantic class labels and export .npy + .json sidecars.

Run inside the ``aifor`` mamba environment, for example::

    mamba run -n aifor python class_unifier.py                                # all plots
    mamba run -n aifor python class_unifier.py class_unifier.dry_run=true     # print plan only
    mamba run -n aifor python class_unifier.py class_unifier.plots=[plot_01]  # a single plot
    mamba run -n aifor python class_unifier.py \
        "class_unifier.class_map={0: 0, 1: 2, 2: 2, 3: 3, 4: 2}"              # merge 1,4 -> 2

All pipeline stages share one config file, ``conf/config.yaml``; this script reads its
``class_unifier:`` section. Override any key with that prefix, e.g.
``class_unifier.unified_field=labels_v2``.
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
