"""Hydra entry point: convert exported clouds between LAZ/LAS and PLY.

Optional stage. Stage 2 exports **LAZ** by default because a binary PLY is an
uncompressed memory dump (37.00 bytes/point measured — 29.2 GB for the 14 plots against
4.4 GB of source), and the clouds have to reach a rented GPU over a network. ForAINet
reads PLY only, so the export is expanded again by this stage, normally *inside* the
training container.

Run inside the ``aifor`` mamba environment, for example::

    mamba run -n aifor python convert.py                            # export -> ForAINet raw/
    mamba run -n aifor python convert.py convert.dry_run=true       # list, write nothing
    mamba run -n aifor python convert.py convert.plots=[plot_11]    # one plot
    mamba run -n aifor python convert.py convert.to=laz             # the other direction

Inside a container there is no Hydra; use the engine's own CLI instead, which needs
only laspy, lazrs, numpy, plyfile and PyYAML::

    python -m pipeline.convert --to ply /data/export /workspace/.../raw/SegmentedForests

All pipeline stages share one config file, ``conf/config.yaml``; this script reads its
``convert:`` section. Override any key with that prefix, e.g. ``convert.to=laz``.
"""
import os

import hydra
from hydra_zen import zen

from pipeline.convert import convert_clouds

# Absolute path to the conf/ directory next to this file, so the entry script can be run
# from anywhere (Hydra also accepts an absolute config_path).
CONF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conf")


@hydra.main(config_path=CONF_DIR, config_name="config", version_base="1.3")
def main(cfg):
    # config.yaml holds one section per stage; run the converter from its `convert:`
    # section. zen() maps the section's keys onto convert_clouds's parameters (and
    # converts OmegaConf containers to plain Python).
    zen(convert_clouds)(cfg.convert)


if __name__ == "__main__":
    main()
