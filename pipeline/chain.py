r"""Chained driver: unprocessed clouds -> ForAINet training clouds, one plot at a time.

Why this exists
---------------
Run as separate stages, the pipeline is a relay of *complete* folders: 3DFin fills
``3DFin_output/`` with all fourteen ``.las`` files (46 GB, measured) before Stage 2 has
read a single one. Nothing needs those files once Stage 2 has consumed them, so the
whole 46 GB is dead weight the moment it is written.

This module runs the same two engines **per plot** and deletes each 3DFin ``.las`` as
soon as Stage 2 has successfully turned it into a training cloud. The transient cost
drops from all fourteen files at once to the largest single one (15.2 GB, plot_08).

3DFin is an external command-line program: it can only hand over its result as a file,
so there is no way to avoid writing that file. Consuming it immediately is the next
best thing, and gets the same outcome.

What it does *not* do
---------------------
No processing of its own. It calls :func:`pipeline.threedfin.run_pipeline` and
:func:`pipeline.forainet_prep.prep_forainet` — the very functions ``main_pipeline.py``
and ``forainet_prep.py`` call — and reads their existing config sections, so a chained
run and a stage-by-stage run cannot drift apart. Each stage keeps its own entry script;
this is an extra way to run them, not a replacement.

Two things are load-bearing
---------------------------
* **The train/val/test split is decided up front, from the SOURCE clouds.** Stage 2
  normally ranks every plot by point count and holds out the smallest, but here it only
  ever sees one plot at a time and could not rank anything. So the ranking is taken from
  the source ``.laz`` headers before 3DFin runs at all — which is sound because 3DFin
  preserves point counts exactly (verified across all 14 plots, zero delta). The result
  is handed to Stage 2 as ``split_of``.
* **Deletion happens only after Stage 2 reports that plot as succeeded.** A failure
  leaves the expensive ``.las`` exactly where it is, so the retry costs minutes rather
  than the hours of re-running 3DFin.

It is driven by ``main_pipeline.py``, which reads the ``chain:`` section of
``conf/config.yaml`` (plus ``threedfin:`` and ``forainet_prep:`` for the stages' own
settings).
"""
from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Union

from pipeline.forainet_prep import _INPUT_SUFFIX, assign_splits, prep_forainet
from pipeline.parallel import estimate_points
from pipeline.threedfin import run_pipeline

log = logging.getLogger(__name__)

# The stage names accepted in `chain.stages`, in the only order that makes sense
# (3DFin adds the tree ids Stage 2 needs).
STAGE_THREEDFIN = "threedfin"
STAGE_FORAINET_PREP = "forainet_prep"
_KNOWN_STAGES = (STAGE_THREEDFIN, STAGE_FORAINET_PREP)

# Extensions the chain considers "bulk data" and may delete once consumed. Deliberately
# narrow: 3dfin_log.txt (the record of what 3DFin did, a few KB) and the patched .ini
# survive, so a plot folder keeps its provenance after the point cloud is gone.
_CLOUD_SUFFIXES = (".las", ".laz", ".ply")


def _call(fn: Callable, section: Dict, **overrides) -> Dict[str, List[str]]:
    """Call a stage engine with its config section, plus the chain's per-plot overrides.

    The section is filtered to the parameters the engine actually accepts. hydra-zen's
    ``zen()`` does this for the standalone entry scripts; here the call is made by hand,
    so a config key the engine does not know would otherwise be a ``TypeError`` deep in
    a batch run. Unknown keys are reported once, since the usual cause is a typo.
    """
    accepted = set(inspect.signature(fn).parameters)
    unknown = sorted(set(section) - accepted - set(overrides))
    if unknown:
        log.warning("%s(): ignoring config key(s) it does not accept: %s",
                    fn.__name__, ", ".join(unknown))
    kwargs = {k: v for k, v in section.items() if k in accepted}
    kwargs.update(overrides)
    return fn(**kwargs)


def _keeps(keep_intermediates: Union[bool, Sequence[str], None], plot: str) -> bool:
    """Should this plot's 3DFin output survive?

    ``keep_intermediates`` is a small three-way switch: ``False``/``None`` deletes
    everything, ``True`` keeps everything, and a **list of plot names** keeps just those.
    The list form is what makes the diagnostic workflow practical — ``class_unifier``
    reads the 3DFin ``.las`` (it is the only thing carrying ``tree_ID``), so inspecting a
    class merge means keeping one or two of them rather than all 46 GB.
    """
    if keep_intermediates is None or isinstance(keep_intermediates, bool):
        return bool(keep_intermediates)
    return plot in {str(name) for name in keep_intermediates}


def _delete_clouds(plot_dir: Path, plot: str) -> int:
    """Remove the point-cloud files directly in ``plot_dir``; return the bytes freed.

    Each removal is guarded: a locked file (a viewer still holding it open) costs a
    warning, not the rest of the batch.
    """
    freed = 0
    if not plot_dir.is_dir():
        return 0
    for entry in sorted(plot_dir.iterdir()):
        if not entry.is_file() or entry.suffix.lower() not in _CLOUD_SUFFIXES:
            continue
        try:
            size = entry.stat().st_size
            entry.unlink()
            freed += size
            log.info("[%s] freed %.1f GB — removed %s", plot, size / 1024 ** 3, entry.name)
        except OSError as exc:
            log.warning("[%s] could not remove %s: %s", plot, entry.name, exc)
    return freed


def run_chain(
    chain: Optional[Dict] = None,
    threedfin: Optional[Dict] = None,
    forainet_prep: Optional[Dict] = None,
) -> Dict[str, List[str]]:
    """Run the pipeline plot by plot, deleting each intermediate once consumed.

    Parameters
    ----------
    chain:
        The ``chain:`` config section — ``stages``, ``keep_intermediates``, ``plots``,
        ``overwrite``, ``dry_run``, ``continue_on_error``. See ``conf/config.yaml``.
    threedfin, forainet_prep:
        The two stages' own config sections, passed through untouched apart from the
        per-plot overrides the chain has to impose (``plots``, ``split_of``, and
        ``workers=1`` for Stage 2, which has exactly one job to do at a time).

    Returns
    -------
    dict
        ``{"succeeded": [...], "failed": [...], "skipped": [...]}`` of plot names.
    """
    chain = dict(chain or {})
    threedfin = dict(threedfin or {})
    forainet_prep = dict(forainet_prep or {})

    # `null` means "all stages"; an empty list is a mistake, not a synonym for it.
    raw_stages = chain.get("stages")
    stages = list(_KNOWN_STAGES) if raw_stages is None else [str(s) for s in raw_stages]
    if not stages:
        raise ValueError("chain.stages is empty — nothing to run. Use null for every stage.")
    unknown = [s for s in stages if s not in _KNOWN_STAGES]
    if unknown:
        raise ValueError(
            f"chain.stages has unknown stage(s) {unknown}; expected any of {list(_KNOWN_STAGES)}"
        )
    run_3dfin = STAGE_THREEDFIN in stages
    run_prep = STAGE_FORAINET_PREP in stages

    keep_intermediates = chain.get("keep_intermediates", False)
    wanted = chain.get("plots")
    overwrite = bool(chain.get("overwrite", False))
    dry_run = bool(chain.get("dry_run", False))
    continue_on_error = bool(chain.get("continue_on_error", True))

    src_dir = Path(threedfin["pointclouds_dir"])
    pattern = threedfin.get("pattern") or "*.laz"
    stage1_out = Path(threedfin["output_dir"])
    prep_in = Path(forainet_prep["input_dir"])
    prep_out = Path(forainet_prep["output_dir"])
    # Stage 2 writes LAZ by default (a PLY is ~7x bigger and has to be uploaded to the
    # rented GPU), so the resume check below cannot assume an extension.
    prep_ext = str(forainet_prep.get("output_format") or "laz").lower().lstrip(".")

    if not src_dir.is_dir():
        raise NotADirectoryError(f"threedfin.pointclouds_dir does not exist: {src_dir}")
    # The chain only works if Stage 2 reads what Stage 1 writes. They are configured
    # independently (each stage must stay runnable on its own), so check rather than assume.
    if run_3dfin and run_prep and stage1_out.resolve() != prep_in.resolve():
        log.warning(
            "threedfin.output_dir (%s) and forainet_prep.input_dir (%s) differ — Stage 2 "
            "will not see what Stage 1 just wrote. Point them at the same folder.",
            stage1_out, prep_in,
        )

    # --- which plots ------------------------------------------------------
    all_sources = sorted(src_dir.glob(pattern))   # kept unfiltered: the split ranks over all
    sources = all_sources
    if wanted:
        wanted_set = {str(p) for p in wanted}
        selected = [p for p in sources if p.stem in wanted_set]
        for name in sorted(wanted_set - {p.stem for p in selected}):
            log.warning("Requested plot %r has no %s in %s", name, pattern, src_dir)
        sources = selected
    if not sources:
        log.warning("No point clouds matched %r in %s — nothing to do.", pattern, src_dir)
        return {"succeeded": [], "failed": [], "skipped": []}

    # --- the split, decided once, before any processing --------------------
    # Point counts come from the SOURCE headers (instant, no point data read). 3DFin
    # preserves them exactly, so this is the same ranking a full Stage 2 run would make
    # from its own inputs -- and it is available now, which per-plot processing requires.
    split_cfg = forainet_prep.get("split") if run_prep else None
    split_of: Dict[str, str] = {}
    if split_cfg:
        # Ranked over EVERY source cloud, not just the selected ones, so `chain.plots=[..]`
        # gives a plot the same suffix a full run would.
        counts = {p.stem: estimate_points(p) for p in all_sources}
        split_of = assign_splits(counts, split_cfg.get("n_val", 0), split_cfg.get("n_test", 0))
        by_split: Dict[str, List[str]] = {}
        for plot_name, which in sorted(split_of.items()):
            by_split.setdefault(which, []).append(plot_name)
        log.info("Split over all %d source cloud(s) by point count (smallest held out): %s",
                 len(counts), "; ".join(f"{k}={len(v)} {v}" for k, v in sorted(by_split.items())))

    if not run_prep:
        # Stage 1 alone: its .las IS the output of this run, so nothing is ever deleted.
        fate = "kept (nothing is consumed without forainet_prep)"
    elif keep_intermediates is True:
        fate = "kept"
    elif keep_intermediates and not isinstance(keep_intermediates, bool):
        fate = f"kept for {list(keep_intermediates)}, deleted for the rest"
    else:
        fate = "deleted after Stage 2 succeeds"
    log.info("Chaining %s over %d plot(s); 3DFin intermediates: %s",
             " -> ".join(stages), len(sources), fate)
    # Logged once here rather than once per plot: it says what labels Stage 2 will write,
    # which is the single most consequential setting in a training run.
    if run_prep and forainet_prep.get("class_map"):
        class_map = forainet_prep["class_map"]
        log.info("Stage 2 class map: %d source class(es) -> %s",
                 len(class_map), sorted({int(v) for v in class_map.values()}))

    succeeded: List[str] = []
    failed: List[str] = []
    skipped: List[str] = []
    freed_total = 0

    for src in sources:
        plot = src.stem
        which = split_of.get(plot)
        stage1_las = stage1_out / plot / f"{plot}{_INPUT_SUFFIX}.las"
        out_cloud = prep_out / f"{f'{plot}_{which}' if which else plot}.{prep_ext}"

        # --- resume: is this plot already done? ---------------------------
        # The last stage in the chain owns the answer, so an interrupted run picks up
        # where it stopped instead of repeating hours of 3DFin.
        final_output = out_cloud if run_prep else stage1_las
        if not overwrite and final_output.exists():
            log.info("[%s] %s already exists and overwrite=False — skipping", plot, final_output.name)
            skipped.append(plot)
            continue

        if dry_run:
            steps = []
            if run_3dfin:
                steps.append(f"3DFin {src.name} => {stage1_las.name}")
            if run_prep:
                steps.append(f"Stage 2 => {out_cloud.name} (+ {plot}_offsets.yml)")
                if not _keeps(keep_intermediates, plot):
                    steps.append(f"delete {stage1_las.name}")
            log.info("[%s] DRY RUN: %s", plot, " -> ".join(steps))

        # --- Stage 1 -------------------------------------------------------
        if run_3dfin:
            result = _call(run_pipeline, threedfin, plots=[plot], dry_run=dry_run,
                           continue_on_error=continue_on_error,
                           log_summary=False)   # the chain writes the one summary at the end
            if plot not in result["succeeded"]:
                log.error("[%s] 3DFin did not produce a result — stopping this plot here", plot)
                (skipped if plot in result["skipped"] else failed).append(plot)
                if not continue_on_error:
                    log.error("continue_on_error=False — stopping after first failure.")
                    break
                continue

        # --- Stage 2 -------------------------------------------------------
        if run_prep:
            # Skipped in a dry run when the input is not there: Stage 2 would honestly
            # report "no cloud matched", which is noise, not information — 3DFin was
            # only pretending to write it a moment ago.
            if dry_run and not stage1_las.exists():
                succeeded.append(plot)
                continue
            result = _call(
                prep_forainet, forainet_prep,
                plots=[plot],
                split_of=split_of,          # already decided; do not re-rank one plot
                overwrite=True,             # the chain settled that above, per plot
                dry_run=dry_run,
                continue_on_error=continue_on_error,
                workers=1,                  # one job — a process pool would only add latency
                log_summary=False,          # the chain writes the one summary at the end
            )
            if plot not in result["succeeded"]:
                log.error("[%s] Stage 2 did not produce %s — the 3DFin output is KEPT so "
                          "this can be retried without re-running 3DFin", plot, out_cloud.name)
                (skipped if plot in result["skipped"] else failed).append(plot)
                if not continue_on_error:
                    log.error("continue_on_error=False — stopping after first failure.")
                    break
                continue

        # --- consume the intermediate --------------------------------------
        # Only once Stage 2 has actually written the training PLY. Running Stage 1 alone
        # would delete the only thing that run produced, so that combination never deletes.
        if run_prep and not dry_run and not _keeps(keep_intermediates, plot):
            freed_total += _delete_clouds(stage1_out / plot, plot)

        succeeded.append(plot)

    # --- summary -----------------------------------------------------------
    log.info("=" * 60)
    if dry_run:
        log.info("DRY RUN (chain: %s) — would process %d plot(s); %d skipped.",
                 " -> ".join(stages), len(succeeded), len(skipped))
    else:
        log.info("chain (%s): processed %d plot(s): %d ok, %d failed, %d skipped.",
                 " -> ".join(stages), len(succeeded) + len(failed),
                 len(succeeded), len(failed), len(skipped))
        if freed_total:
            log.info("Freed %.1f GB of intermediates along the way.", freed_total / 1024 ** 3)
    if failed:
        log.warning("Failed: %s", ", ".join(failed))
    if skipped:
        log.warning("Skipped: %s", ", ".join(skipped))

    return {"succeeded": succeeded, "failed": failed, "skipped": skipped}
