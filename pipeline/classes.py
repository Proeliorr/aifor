r"""Semantic class remapping — shared by the class unifier and Stage 2.

Different datasets encode their ground truth with different class numbers. Before
such data can be fed to ForAINet the labels must be brought into ForAINet's own
scheme, and the classes with no counterpart there must disappear entirely. Both
``pipeline/class_unifier.py`` (which exports ``.npy`` for inspection) and
``pipeline/forainet_prep.py`` (which writes the PLYs ForAINet actually trains on)
need exactly the same rule, so it lives here — in **one** place, driven by the
shared ``classes:`` block of ``conf/config.yaml``.

Why a separate module: ``class_unifier`` already imports the cloud reader from
``forainet_prep``, so ``forainet_prep`` cannot import back from ``class_unifier``
without a circular import. A neutral third module both can import avoids that.

Two rules worth knowing:

* **The map must be total.** A source value the map does not mention raises. An
  unmapped label is a configuration mistake — silently keeping it would smuggle a
  bogus class into training, and silently forcing it to some default would invent
  ground truth. Whoever runs the pipeline is expected to know their label column
  and state a complete mapping.
* **``drop_value`` removes points.** Classes that have no counterpart in the target
  scheme map to ``drop_value`` (``-1`` by convention) and their points are removed
  from the output, rather than kept under an "ignore" label.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def apply_class_map(labels: np.ndarray, class_map: Dict[int, int],
                    plot: str = "?") -> np.ndarray:
    """Send ``labels`` through ``class_map`` and return the remapped array.

    The remap is vectorised: one boolean mask per map entry, applied to a copy, so
    the caller's array is never modified. Because several source values may share a
    target, merging classes needs no special handling — ``{3: 2, 10: 2}`` simply
    folds both into 2.

    The result is ``int64`` so negative sentinels (``drop_value``) are
    representable; narrow it later with :func:`to_label_dtype`, once the points
    carrying them have been dropped.

    Raises ``ValueError`` naming ``plot`` and the offending values if the map does
    not cover every value present (see the module docstring).
    """
    src = np.asarray(labels)
    if not class_map:
        raise ValueError(
            f"[{plot}] class_map is empty — every source class must be given an "
            f"explicit target (values present: {np.unique(src).tolist()})"
        )

    unified = np.empty(src.shape, dtype=np.int64)
    covered = np.zeros(src.shape, dtype=bool)
    for old, new in class_map.items():
        hit = src == int(old)
        unified[hit] = int(new)
        covered |= hit

    if not covered.all():
        missing = [int(v) for v in np.unique(src[~covered])]
        raise ValueError(
            f"[{plot}] source class(es) {missing} are not in class_map "
            f"(mapped: {sorted(int(k) for k in class_map)}). Every class present in "
            f"the data must be mapped explicitly — add them to the `classes.class_map` "
            f"block in conf/config.yaml, sending anything irrelevant to the "
            f"`classes.drop_value` sentinel."
        )
    return unified


def keep_mask(unified: np.ndarray, drop_value: Optional[int]) -> np.ndarray:
    """Boolean mask of the rows to keep (``drop_value=None`` keeps everything)."""
    unified = np.asarray(unified)
    if drop_value is None:
        return np.ones(unified.shape, dtype=bool)
    return unified != int(drop_value)


def to_label_dtype(unified: np.ndarray) -> np.ndarray:
    """Narrow remapped labels to ``uint8`` (ForAINet's dtype) when they fit.

    Call this only AFTER dropping, since a negative ``drop_value`` does not fit and
    would keep the whole column wide for the sake of points that are on their way
    out.
    """
    unified = np.asarray(unified)
    if unified.size and unified.min() >= 0 and unified.max() <= 255:
        return unified.astype(np.uint8)
    return unified


def remap_and_filter(
    labels: np.ndarray,
    class_map: Dict[int, int],
    drop_value: Optional[int],
    plot: str = "?",
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Map ``labels`` and work out which points survive — the common path.

    Returns ``(unified_kept, keep, n_dropped)``:

    * ``unified_kept`` — the remapped labels of the surviving points, already
      narrowed to ``uint8`` where possible;
    * ``keep`` — the boolean mask, so the caller can subset its *other* columns
      (coordinates, intensity, tree id) the same way;
    * ``n_dropped`` — how many points the mask removes.
    """
    unified = apply_class_map(labels, class_map, plot)
    keep = keep_mask(unified, drop_value)
    return to_label_dtype(unified[keep]), keep, int((~keep).sum())
