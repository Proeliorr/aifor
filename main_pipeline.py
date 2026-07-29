"""Hydra entry point: run the whole pipeline, one plot at a time.

Takes the **unprocessed** plot clouds and produces the PLYs ForAINet trains on, running
3DFin and Stage 2 per plot and deleting each 3DFin ``.las`` as soon as Stage 2 has
consumed it. That keeps the transient cost to one plot's intermediate (15 GB) instead of
all fourteen at once (46 GB, measured).

Run inside the ``aifor`` mamba environment, for example::

    mamba run -n aifor python main_pipeline.py                                   # everything
    mamba run -n aifor python main_pipeline.py chain.dry_run=true                # plan only
    mamba run -n aifor python main_pipeline.py chain.plots=[plot_11]             # one plot
    mamba run -n aifor python main_pipeline.py chain.keep_intermediates=true     # keep the .las
    mamba run -n aifor python main_pipeline.py chain.stages=[threedfin]          # Stage 1 only

**Note** — this script used to run 3DFin and nothing else. That is now
``chain.stages=[threedfin]``; the individual stages also still have their own entry
scripts (``forainet_prep.py``, ``class_unifier.py``) and are unchanged.

All pipeline stages share one config file, ``conf/config.yaml``. This script reads its
``chain:`` section for what to run, and the ``threedfin:`` / ``forainet_prep:`` sections
for how to run each stage — so a chained run and a stage-by-stage run use exactly the
same settings.
"""
import os

import hydra
from omegaconf import OmegaConf

from pipeline.chain import run_chain

# Absolute path to the conf/ directory next to this file, so the entry script can be run
# from anywhere (Hydra also accepts an absolute config_path).
CONF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conf")


@hydra.main(config_path=CONF_DIR, config_name="config", version_base="1.3")
def main(cfg):
    # The other entry scripts use hydra_zen's zen(), which maps ONE config section onto a
    # function's parameters. The chain needs three sections at once (its own, plus both
    # stages'), so the sections are converted here and passed straight through. `resolve`
    # expands the ${paths.*} interpolations; the engine then sees plain Python.
    sections = OmegaConf.to_container(cfg, resolve=True)
    run_chain(
        chain=sections.get("chain"),
        threedfin=sections.get("threedfin"),
        forainet_prep=sections.get("forainet_prep"),
    )


if __name__ == "__main__":
    main()
