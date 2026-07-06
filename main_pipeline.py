"""Hydra entry point: batch-run the 3DFin CLI over a folder of point clouds.

Run inside the ``aifor`` mamba environment, for example::

    mamba run -n aifor python main_pipeline.py                              # full batch (all plots)
    mamba run -n aifor python main_pipeline.py threedfin.dry_run=true       # print commands only
    mamba run -n aifor python main_pipeline.py threedfin.plots=[plot_11]    # a single plot

All pipeline stages share one config file, ``conf/config.yaml``; this script reads its
``threedfin:`` section. Override any key with that prefix, e.g.
``threedfin.pointclouds_dir=/other/path threedfin.normalize=true``.
"""
import os

import hydra
from hydra_zen import zen

from pipeline.threedfin import run_pipeline

# Absolute path to the conf/ directory next to this file, so the entry script can be run
# from anywhere (Hydra also accepts an absolute config_path).
CONF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conf")


@hydra.main(config_path=CONF_DIR, config_name="config", version_base="1.3")
def main(cfg):
    # config.yaml holds one section per stage; run 3DFin from its `threedfin:` section.
    # zen() maps the section's keys onto run_pipeline's parameters (and converts OmegaConf
    # containers to plain Python), falling back to the function defaults for anything absent.
    zen(run_pipeline)(cfg.threedfin)


if __name__ == "__main__":
    main()
