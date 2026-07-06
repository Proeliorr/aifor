"""Hydra entry point: prepare 3DFin plot clouds for ForAINet (center + PLY), or
restore classified results to original coordinates.

Run inside the ``aifor`` mamba environment, for example::

    mamba run -n aifor python forainet_prep.py                                  # preprocess all plots
    mamba run -n aifor python forainet_prep.py forainet_prep.dry_run=true       # print plan only
    mamba run -n aifor python forainet_prep.py forainet_prep.plots=[plot_02]    # a single plot
    mamba run -n aifor python forainet_prep.py forainet_prep.mode=restore \
        forainet_prep.restore_dir=/path/to/classified_plys                      # restore results

All pipeline stages share one config file, ``conf/config.yaml``; this script reads its
``forainet_prep:`` section. Override any key with that prefix, e.g.
``forainet_prep.restore_format=ply``.
"""
import os

import hydra
from hydra_zen import zen

from pipeline.forainet_prep import prep_forainet

# Absolute path to the conf/ directory next to this file, so the entry script can be run
# from anywhere (Hydra also accepts an absolute config_path).
CONF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conf")


@hydra.main(config_path=CONF_DIR, config_name="config", version_base="1.3")
def main(cfg):
    # config.yaml holds one section per stage; run this stage from its `forainet_prep:`
    # section. zen() maps the section's keys onto prep_forainet's parameters (and converts
    # OmegaConf containers to plain Python), falling back to defaults for anything absent.
    zen(prep_forainet)(cfg.forainet_prep)


if __name__ == "__main__":
    main()
