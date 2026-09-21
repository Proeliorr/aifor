#!/usr/bin/env python3
r"""Does the ForAINet submodule still agree with our class scheme?

Why this exists
---------------
Our semantic scheme lives in one place -- the ``classes:`` block of
``conf/config.yaml`` -- but ForAINet has to be told about it separately, by editing
its source. Those edits live in ``patches/forainet-local.patch`` and are **not** stored
by this repository: a submodule records only a commit SHA, so a plain
``git submodule update`` silently reverts every one of them, restoring the upstream
5-class tables while our config still says 4.

Nothing crashes when that happens. Training runs, the loss goes down, and the semantic
head simply has a fifth output that can never receive a label -- while ``final_eval``
divides its metrics by the wrong class count. The failure is quiet and the numbers are
wrong, which is the worst combination.

So this script re-derives what ForAINet *should* contain from our config and checks it,
by parsing the submodule's source rather than importing it (importing would need torch,
MinkowskiEngine and a GPU-shaped environment; parsing needs nothing).

Run it after any submodule operation, and before building a training image::

    python misc/check_forainet_classes.py            # exits non-zero on any mismatch
    python misc/check_forainet_classes.py --verbose  # show every value it read

The checks
----------
Three files, two numbering schemes. ForAINet stores ``semantic_seg`` 1-based on disk and
subtracts one on load, so the model works in ``0..N-1``; ``final_eval`` then adds one
back, giving ``0..N`` with **0 = ignore**. Both are checked, because mixing them up is
the easiest mistake to make here (see docs/eval_process.md).
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "conf" / "config.yaml"
FORAINET = ROOT / "ForAINet" / "PointCloudSegmentation"
SEG = FORAINET / "torch_points3d" / "datasets" / "segmentation" / "treeins_set1.py"
PAN = FORAINET / "torch_points3d" / "datasets" / "panoptic" / "treeins_set1.py"
MODEL_CFG = FORAINET / "conf" / "models" / "panoptic" / "FORpartseg_3heads.yaml"
# The pooled "final" report across plots. It repeats final_eval's 1-based constants in its
# own `if __name__` block, so it drifts independently -- hence checking it here too.
STATS = FORAINET / "evaluation_stats_FOR.py"


# --------------------------------------------------------------------------- reading
def _literal(path: Path, name: str, *, inside: Optional[str] = None):
    """Return the literal assigned to ``name``, or None if it is absent.

    Parsed with ``ast`` rather than executed: this file must run in a bare environment
    with no torch. ``inside`` restricts the search to one function body, which is how
    the two ``final_eval`` locals are reached.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    scope = tree
    if inside is not None:
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == inside:
                scope = node
                break
        else:
            return None
    for node in ast.walk(scope):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return _value_of(node.value)
    return None


def _value_of(node: ast.AST):
    """Literal behind an expression, unwrapping ``np.array([...])``-style wrappers.

    Several of these constants are numpy calls rather than bare literals
    (``SemIDforInstance = np.array([2,3])``), which ``literal_eval`` rejects outright.
    A one-argument call is unwrapped and its argument evaluated instead.
    """
    if isinstance(node, ast.Call) and node.args:
        return _value_of(node.args[0])
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError):
        return "<not a literal>"


def _color_rows(path: Path) -> Optional[int]:
    """Count the rows of the OBJECT_COLOR table (a np.asarray of literal lists)."""
    source = path.read_text(encoding="utf-8")
    match = re.search(r"OBJECT_COLOR\s*=\s*np\.asarray\(\s*(\[.*?\])\s*\)", source, re.S)
    if not match:
        return None
    body = re.sub(r"#[^\n]*", "", match.group(1))          # strip the trailing colour names
    try:
        return len(ast.literal_eval(body))
    except (ValueError, SyntaxError):
        return None


def _loads_under_omegaconf_20(path: Path) -> Optional[str]:
    """Would the TRAINING IMAGE be able to parse this yaml? Returns None if yes.

    The image has hydra-core 1.0.7, which pins omegaconf 2.0.x, whose yaml loader calls
    ``construct_object()`` on every key node. A merge key (``<<: *anchor``) is a key node
    tagged ``tag:yaml.org,2002:merge``, for which 2.0 registers no constructor -- so the
    file raises while *parsing*, before hydra reads a single setting, and both backbones
    die with an error that names neither.

    omegaconf 2.3 (this repo's ``aifor`` env) overrides ``construct_mapping`` instead and
    lets PyYAML expand merges, so a merge key looks perfectly fine on the dev machine.
    That asymmetry already cost one rented-GPU run; this reproduces 2.0's loader exactly.
    """
    def no_duplicates_constructor(loader, node, deep=False):
        mapping = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            value = loader.construct_object(value_node, deep=deep)
            if key in mapping:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping", node.start_mark,
                    "found duplicate key %s" % key, key_node.start_mark)
            mapping[key] = value
        return loader.construct_mapping(node, deep)

    class Loader20(yaml.SafeLoader):
        pass

    Loader20.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, no_duplicates_constructor)
    try:
        yaml.load(path.read_text(encoding="utf-8"), Loader=Loader20)
    except yaml.YAMLError as exc:
        return str(exc).replace("\n", " ")
    return None


def _model_block_diff(path: Path, base: str, variant: str,
                      allowed: Tuple[str, ...]) -> List[str]:
    """Keys where ``variant`` differs from ``base`` beyond the ``allowed`` overrides.

    The two blocks are a deliberate copy-paste (see the comment above PointGroup-PAPER-TS:
    a merge key cannot be used here), so something has to notice when one is edited and
    the other is not. Interpolations that name their own block -- every
    ``${models.<name>.feat_size}`` -- are rewritten to the base's name before comparing,
    since those are *supposed* to differ.
    """
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if base not in doc or variant not in doc:
        return ["%s or %s is missing from %s" % (base, variant, path.name)]

    def retarget(node):
        if isinstance(node, dict):
            return {k: retarget(v) for k, v in node.items()}
        if isinstance(node, list):
            return [retarget(v) for v in node]
        if isinstance(node, str):
            return node.replace("models.%s." % variant, "models.%s." % base)
        return node

    base_cfg, var_cfg = doc[base], retarget(doc[variant])
    problems = []
    for key in sorted(set(base_cfg) | set(var_cfg)):
        if key in allowed:
            continue
        if base_cfg.get(key, "<absent>") != var_cfg.get(key, "<absent>"):
            problems.append("%s: %s has %r, %s has %r"
                            % (key, base, base_cfg.get(key, "<absent>"),
                               variant, var_cfg.get(key, "<absent>")))
    return problems


def _yaml_value(path: Path, key: str) -> Tuple[bool, object]:
    """Find ``key:`` in a Hydra yaml without resolving its interpolations."""
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or not stripped.startswith(key + ":"):
            continue
        raw = stripped[len(key) + 1:].strip()
        return True, yaml.safe_load(raw) if raw else None
    return False, None


# --------------------------------------------------------------------------- checking
class Checker:
    def __init__(self, verbose: bool = False) -> None:
        self.failures: List[str] = []
        self.verbose = verbose

    def check(self, label: str, got, want, hint: str = "") -> None:
        ok = got == want
        if not ok:
            self.failures.append(
                "%s\n      expected: %r\n      found:    %r%s"
                % (label, want, got, ("\n      %s" % hint) if hint else "")
            )
        if self.verbose or not ok:
            print("  [%s] %-38s %r" % ("ok " if ok else "FAIL", label, got))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="print every value read, not just the failures")
    args = parser.parse_args(argv)

    for path in (CONFIG, SEG, PAN, MODEL_CFG):
        if not path.exists():
            print("cannot check: missing %s" % path)
            if "ForAINet" in str(path):
                print("  the submodule is probably not initialised: "
                      "git submodule update --init")
            return 2

    # --- what our config says ------------------------------------------------
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    classes: Dict = cfg["classes"]
    names: Dict[int, str] = {int(k): str(v) for k, v in classes["class_names"].items()}
    instance_classes = [int(c) for c in (classes.get("instance_classes") or [])]

    # class_names includes 0 = unclassified, which ForAINet does not count: its tables
    # are 0-based over the REAL classes, with unclassified reached via the -1 shift.
    real = [names[i] for i in sorted(names) if i != 0]
    n_real = len(real)
    # instance_classes uses the config's 1-based numbering; ForAINet's SemIDforInstance
    # uses the model's 0-based numbering, hence the -1.
    want_sem_for_instance = [c - 1 for c in instance_classes]
    # final_eval adds one back, so there the thing classes are the config numbers again.
    want_thing = sorted(instance_classes)
    want_stuff = sorted(set(range(1, n_real + 1)) - set(want_thing))

    print("From conf/config.yaml classes:")
    print("  %d real class(es): %s" % (n_real, ", ".join(real)))
    print("  instance (thing) classes: %s" % instance_classes)
    print()
    print("Checking the ForAINet submodule:")

    c = Checker(args.verbose)

    # --- datasets/segmentation/treeins_set1.py -------------------------------
    c.check("Treeins_NUM_CLASSES", _literal(SEG, "Treeins_NUM_CLASSES"), n_real,
            "this is dataset.num_classes -- the model head width")
    seg_labels = _literal(SEG, "INV_OBJECT_LABEL")
    c.check("INV_OBJECT_LABEL", seg_labels, dict(enumerate(real)))
    c.check("OBJECT_COLOR rows (segmentation)", _color_rows(SEG), n_real + 1,
            "one row per class plus a final row for unlabelled")

    # --- datasets/panoptic/treeins_set1.py -----------------------------------
    c.check("CLASSES_INV", _literal(PAN, "CLASSES_INV"), dict(enumerate(real)))
    c.check("OBJECT_COLOR rows (panoptic)", _color_rows(PAN), n_real + 1)
    c.check("VALID_CLASS_IDS", _literal(PAN, "VALID_CLASS_IDS"), list(range(n_real)))
    sem_for_instance = _literal(PAN, "SemIDforInstance")
    got = (list(sem_for_instance) if isinstance(sem_for_instance, (list, tuple))
           else sem_for_instance)
    c.check("SemIDforInstance", got, want_sem_for_instance,
            "0-based; equals instance_classes minus 1")

    # --- final_eval, which uses the 1-based scheme ---------------------------
    c.check("final_eval NUM_CLASSES_sem", _literal(PAN, "NUM_CLASSES_sem", inside="final_eval"),
            n_real + 1, "real classes plus the ignore class at index 0")
    c.check("final_eval NUM_CLASSES_count", _literal(PAN, "NUM_CLASSES_count", inside="final_eval"),
            n_real, "real classes only; divides mIoU in the sibling copies")
    c.check("final_eval sem_classcount", _literal(PAN, "sem_classcount", inside="final_eval"),
            list(range(1, n_real + 1)), "1-based: index 0 is the ignore class")
    c.check("final_eval thing_classes", _literal(PAN, "thing_classes", inside="final_eval"),
            want_thing)
    c.check("final_eval stuff_classes", _literal(PAN, "stuff_classes", inside="final_eval"),
            want_stuff)
    c.check("final_eval NUM_CLASSES (binary)", _literal(PAN, "NUM_CLASSES", inside="final_eval"),
            3, "unclassified / stuff / thing -- independent of the class count")

    # --- evaluation_stats_FOR.py: the pooled report across plots -------------
    # Same 1-based scheme as final_eval, declared a second time in that script's
    # `if __name__` block (which _literal sees, since it walks the whole module).
    # Left at upstream's five-class values it silently averages mIoU/mAcc over a class
    # that cannot exist and folds a phantom class 5 into "tree".
    if STATS.exists():
        hint = "evaluation_stats_FOR.py repeats final_eval's constants; keep them in step"
        c.check("stats NUM_CLASSES_sem", _literal(STATS, "NUM_CLASSES_sem"), n_real + 1, hint)
        c.check("stats sem_classcount", _literal(STATS, "sem_classcount"),
                list(range(1, n_real + 1)), hint)
        c.check("stats sem_classcount_remove_ground",
                _literal(STATS, "sem_classcount_remove_ground"),
                [c_ for c_ in range(1, n_real + 1) if names.get(c_) != "ground"],
                "every real class except ground")
        c.check("stats thing_classes", _literal(STATS, "thing_classes"), want_thing, hint)
        c.check("stats stuff_classes", _literal(STATS, "stuff_classes"), want_stuff, hint)
    else:
        print("  [ok ] %-38s %r" % ("evaluation_stats_FOR.py absent", True))

    # --- the model config: can the IMAGE even parse it? ----------------------
    parse_error = _loads_under_omegaconf_20(MODEL_CFG)
    if parse_error:
        c.failures.append(
            "%s does not parse under omegaconf 2.0 (what the image has):\n"
            "      %s\n"
            "      A YAML merge key (`<<: *anchor`) is the usual cause. It works here\n"
            "      (omegaconf 2.3) and fails in the container, before model_name is read."
            % (MODEL_CFG.name, parse_error))
        print("  [FAIL] %-38s %s" % ("parses under omegaconf 2.0", parse_error))
    elif args.verbose:
        print("  [ok ] %-38s %r" % ("parses under omegaconf 2.0", True))

    # PointGroup-PAPER-TS is a full copy of PointGroup-PAPER (it cannot use a merge key),
    # so check the copy has not drifted. `class` picks the model file, `backend` the
    # sparse library -- those two are the whole point of the variant.
    drift = _model_block_diff(MODEL_CFG, "PointGroup-PAPER", "PointGroup-PAPER-TS",
                              allowed=("class", "backend"))
    for problem in drift:
        c.failures.append("PointGroup-PAPER-TS drifted from PointGroup-PAPER\n      " + problem)
        print("  [FAIL] %-38s %s" % ("TS block matches PAPER block", problem))
    if not drift and args.verbose:
        print("  [ok ] %-38s %r" % ("TS block matches PAPER block", True))

    present, pretrained = _yaml_value(MODEL_CFG, "path_pretrained")
    if present and pretrained:
        c.failures.append(
            "path_pretrained is set to %r\n"
            "      A checkpoint trained on a different class count loads silently:\n"
            "      base_model.py uses load_state_dict_with_same_shape(strict=False),\n"
            "      which SKIPS the semantic head instead of failing. Set it to null,\n"
            "      or make sure that checkpoint has %d classes." % (pretrained, n_real)
        )
        print("  [FAIL] %-38s %r" % ("path_pretrained", pretrained))
    elif args.verbose:
        print("  [ok ] %-38s %r" % ("path_pretrained", pretrained))

    # --- report --------------------------------------------------------------
    print()
    if c.failures:
        print("%d MISMATCH(ES) -- the submodule does not match conf/config.yaml:" % len(c.failures))
        for i, failure in enumerate(c.failures, 1):
            print("  %d. %s" % (i, failure))
        print()
        print("Most likely cause: the local patch was reverted (a submodule records only")
        print("a commit SHA, so `git submodule update` restores upstream). Reapply it:")
        print("    cd ForAINet && git apply ../patches/forainet-local.patch")
        return 1

    print("OK: ForAINet agrees with conf/config.yaml (%d classes, things=%s)."
          % (n_real, want_thing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
