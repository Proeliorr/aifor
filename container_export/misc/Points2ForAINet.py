#!/usr/bin/env python3
r"""Adapt point clouds to be compatible with running ForAINet.

This is the **standalone / in-container** front-end to the pipeline's converter. It has
no Hydra dependency, so it drops into a Docker image alongside ``pipeline/`` and five
wheels (``laspy``, ``lazrs``, ``numpy``, ``plyfile``, ``PyYAML``).

Why it is only a front-end now
------------------------------
This script used to carry its own reader, writer and offset bookkeeping. All three were
worse than the pipeline's, and one was actively dangerous: its ``array_to_las`` wrote
only ``x/y/z/intensity``, so converting a training cloud to LAS **silently discarded
``semantic_seg`` and ``treeID``** — the labels ForAINet trains on. It also used a
``dummy_``/``restored_`` filename prefix and a single path-keyed ``offset.yml``, neither
of which matches the ``<plot>_offsets.yml`` the pipeline writes.

So the conversion now lives in :mod:`pipeline.convert`, which reuses the same
``_read_cloud``/``_write_las`` the restore path has always used: LAS extra-bytes
dimensions for every non-standard field, the source scales, and the CRS as a WKT VLR.
One implementation, used by the pipeline stage, by this script, and inside the container.

Centering and restoring coordinates are **not** done here — that is Stage 2's job, and
it keeps the offsets file that makes the operation reversible::

    python forainet_prep.py                                     # center + export
    python forainet_prep.py forainet_prep.mode=restore ...      # put predictions back

Examples
--------
Expand a LAZ export into the PLYs ForAINet reads (the container step)::

    python misc/Points2ForAINet.py /data/export /workspace/.../raw/SegmentedForests

The other direction, keeping the source scales and CRS from the offsets files::

    python misc/Points2ForAINet.py results/ out/ --to laz --offsets-dir SegmentedForests/ForAINet_input

A single plot, or a dry run::

    python misc/Points2ForAINet.py /data/export out/ --plots plot_11
    python misc/Points2ForAINet.py /data/export out/ --dry-run

**The file stem is preserved**, and that matters: ForAINet reads the train/val/test
split from the file NAME (``name[-7:-4] == "val"``, then ``name[-8:-4] == "test"``), so
``plot_11_val.laz`` must become ``plot_11_val.ply``. Renaming a file, or adding the
``dummy_`` prefix this script once used, quietly moves a plot between sets.
"""
import sys
from pathlib import Path

# Runnable straight from a checkout (`python misc/Points2ForAINet.py`) without the repo
# root being on PYTHONPATH -- the container copies files, it does not pip-install them.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.convert import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
