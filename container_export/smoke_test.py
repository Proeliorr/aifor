"""Build-time check: can this image actually convert LAZ to PLY?

Run as a RUN step in the Dockerfile so it fails on your laptop in seconds, rather than
on a rented GPU after the data has already been downloaded.

It exercises the four assumptions that are easy to get wrong on Python 3.8:

1. laspy is new enough to expose ``ExtraBytesParams`` and ``header.parse_crs`` --
   the base image ships laspy 2.0.3 (2021), which predates the latter.
2. A **LAZ backend** is installed. laspy reads .las with no help but needs lazrs or
   laszip for .laz; the base image had neither, so `laspy.read("*.laz")` would raise.
3. Extra-bytes dimensions survive a write/read cycle. `semantic_seg` and `treeID` are
   stored that way -- if they were dropped, training would silently run on zeroed
   labels.
4. The pipeline package imports without Hydra, and torch/numpy still work afterwards
   (pip can quietly move numpy while resolving laspy).
"""
import sys
import tempfile
import os

import numpy as np

print("python      %s" % sys.version.split()[0])

# --- 1 + 2: laspy with a LAZ backend ---------------------------------------
import laspy

print("laspy       %s" % laspy.__version__)
backends = [b.name for b in laspy.LazBackend if b.is_available()]
print("laz backend %s" % (backends or "NONE"))
if not backends:
    raise SystemExit(
        "FAIL: no LAZ backend. laspy cannot open .laz files.\n"
        "      Add:  pip install lazrs   (or  pip install 'laspy[lazrs]')"
    )

# --- 3: a two-point LAZ with the exact fields the export uses ---------------
header = laspy.LasHeader(point_format=6, version="1.4")
header.scales = np.array([1e-7, 1e-7, 1e-7])
header.offsets = np.zeros(3)
header.add_extra_dim(laspy.ExtraBytesParams(name="semantic_seg", type=np.uint8))
header.add_extra_dim(laspy.ExtraBytesParams(name="treeID", type=np.uint32))

las = laspy.LasData(header)
las.x = np.array([0.0, 1.5])
las.y = np.array([0.0, 2.5])
las.z = np.array([0.0, 3.5])
las.intensity = np.array([7, 9], dtype=np.uint16)
las.semantic_seg = np.array([1, 4], dtype=np.uint8)
las.treeID = np.array([0, 12345], dtype=np.uint32)

path = os.path.join(tempfile.mkdtemp(), "smoke.laz")
las.write(path)                       # exercises the compressor
back = laspy.read(path)               # exercises the decompressor

assert list(back.semantic_seg) == [1, 4], list(back.semantic_seg)
assert list(back.treeID) == [0, 12345], list(back.treeID)
assert abs(float(back.x[1]) - 1.5) < 1e-9, float(back.x[1])
back.header.parse_crs()               # used by _crs_wkt in forainet_prep
print("laz i/o     extra dims survive round trip")

# --- 4: the package imports, and nothing heavy came with it -----------------
import plyfile                        # noqa: E402

print("plyfile     %s" % getattr(plyfile, "__version__", "?"))

before = set(sys.modules)
import pipeline.convert               # noqa: E402,F401

heavy = sorted(m for m in set(sys.modules) - before
               if m.split(".")[0] in {"hydra", "hydra_zen", "omegaconf"})
if heavy:
    raise SystemExit("FAIL: converter pulled in Hydra: %s" % heavy)
print("pipeline    imports clean (no hydra/omegaconf)")

print("numpy       %s" % np.__version__)
try:
    import torch                      # noqa: E402

    # No GPU is present at build time, so only the import and the build tag are checked.
    print("torch       %s  (cuda build %s)" % (torch.__version__, torch.version.cuda))
except ModuleNotFoundError:
    raise SystemExit(
        "FAIL: torch is not installed at all. This is the base image, not the pip step "
        "above -- you are probably building FROM the wrong image."
    )
except Exception as exc:
    raise SystemExit(
        "FAIL: torch is installed but no longer imports (%s).\n"
        "      The pip step above most likely moved numpy out from under it." % exc
    )

print()
print("SMOKE TEST PASSED")
