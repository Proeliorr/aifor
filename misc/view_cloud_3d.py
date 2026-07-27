r"""Standalone 3-D point-cloud viewer — the child process of the split viewer.

``misc/view_split_point_cloud.ipynb`` shows each cloud as five fixed 2-D projections.
Its **3D view** button hands the currently filtered selection to *this* script, which
opens a real OpenGL window you can orbit, zoom and pan.

Why a separate process
----------------------
VTK (via PyVista) and tkinter each want to own the GUI event loop. Calling
``plotter.show()`` from inside a tkinter callback nests one loop inside the other:
the 2-D window freezes and the combination is fragile. Running the 3-D view as its
own process avoids the conflict entirely — the 2-D viewer stays responsive, and
Cloud A and Cloud B can each have a 3-D window open at the same time.

Usage
-----
    python view_cloud_3d.py <payload.npz>              # open the interactive window
    python view_cloud_3d.py <payload.npz> --smoke out.png   # off-screen render (tests)

The payload is written by the notebook (``CloudPanel.open_3d_view``) and is deleted
by this script once loaded, so nothing accumulates in the temp folder. It holds:

    xyz     (N, 3) float32                  the point coordinates
    rgb     (N, 3) uint8    [discrete]      one colour per point
    scalars (N,)   float32  [continuous]    raw values, coloured with viridis here
    meta    JSON string                     title, field, discrete, uniq, palette, ...

Controls in the window: left-drag = rotate, scroll = zoom, middle-drag = pan,
'r' = reset camera, 'q' = close.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# PyVista pulls in VTK, which is heavy; import it behind a friendly error so a
# missing install explains itself instead of dumping a traceback into a log file
# nobody reads.
try:
    import pyvista as pv
except ImportError:  # pragma: no cover - exercised only on a broken env
    sys.exit(
        "This 3-D view needs PyVista + VTK, which are not installed in this "
        "environment.\n\nInstall them with:\n"
        '    cmd /c "mamba run -n aifor python -m pip install --user pyvista vtk"'
    )

_BACKGROUND = "#1e1e1e"      # dark grey: bright points read better against it
_DEFAULT_POINT_SIZE = 2.0


def load_payload(path: Path) -> dict:
    """Read the .npz handed over by the notebook and delete it afterwards."""
    with np.load(path, allow_pickle=False) as data:
        payload = {
            "xyz": np.asarray(data["xyz"], dtype=np.float32),
            "rgb": np.asarray(data["rgb"]) if "rgb" in data else None,
            "scalars": np.asarray(data["scalars"]) if "scalars" in data else None,
            "meta": json.loads(str(data["meta"])),
        }
    # The parent never cleans up: whoever consumes the payload owns it.
    try:
        path.unlink()
    except OSError:
        pass  # a leftover temp file is not worth failing the viewer over
    return payload


def build_plotter(payload: dict, off_screen: bool = False):
    """Create the PyVista plotter for a payload (shared by window + smoke test)."""
    meta = payload["meta"]
    field = meta.get("field", "value")
    xyz = payload["xyz"]

    plotter = pv.Plotter(off_screen=off_screen, title=meta.get("title", "3D view"))
    plotter.set_background(_BACKGROUND)

    cloud = pv.PolyData(xyz)
    point_size = float(meta.get("point_size", _DEFAULT_POINT_SIZE))

    if payload["rgb"] is not None:
        # Discrete labels: colours are already resolved, hand VTK the RGB triples.
        cloud["rgb"] = payload["rgb"]
        plotter.add_points(cloud, scalars="rgb", rgb=True, point_size=point_size,
                           render_points_as_spheres=False)
        _add_class_legend(plotter, meta)
    else:
        # Continuous field: let PyVista map + draw a scalar bar for it.
        cloud[field] = payload["scalars"]
        plotter.add_points(cloud, scalars=field, cmap="viridis",
                           point_size=point_size, render_points_as_spheres=False,
                           scalar_bar_args={"title": field, "color": "white"})

    # Orientation helpers: a corner axes widget and a labelled bounding box.
    plotter.add_axes()
    plotter.show_bounds(location="outer", color="white", grid=False,
                        xtitle="x", ytitle="y", ztitle="z")
    plotter.camera_position = "iso"

    # Eye-dome lighting shades points by depth discontinuity, which can help read
    # 3-D structure -- but it also darkens the cloud dramatically (measured here:
    # mean brightness 104 -> 6 on a sparse selection, i.e. very nearly black), so
    # it stays OFF by default and is offered as a toggle instead.
    # NOTE: do not bind this to 'e' -- PyVista already uses 'e' (and 'q') to exit.
    _add_edl_toggle(plotter)

    subtitle = (f"{len(xyz):,} points  |  coloured by {field}\n"
                "drag rotate  |  scroll zoom  |  middle-drag pan  |  "
                "r reset  |  d depth-shading  |  q close")
    plotter.add_text(subtitle, position="upper_left", font_size=9, color="white")
    return plotter


def _add_edl_toggle(plotter) -> None:
    """Bind 'd' to switch eye-dome lighting on/off (off initially)."""
    state = {"on": False}

    def toggle():
        state["on"] = not state["on"]
        if state["on"]:
            plotter.enable_eye_dome_lighting()
        else:
            plotter.disable_eye_dome_lighting()
        plotter.render()

    try:
        plotter.add_key_event("d", toggle)
    except Exception:
        pass  # a missing shortcut must never stop the window from opening


def _add_class_legend(plotter, meta: dict) -> None:
    """Corner legend listing the discrete classes, mirroring the 2-D legend."""
    uniq = meta.get("uniq") or []
    palette = meta.get("palette") or []
    if not uniq or len(uniq) != len(palette):
        return
    # add_legend wants (label, colour) pairs; colours arrive as 0-255 RGB triples.
    entries = [[str(v), [c / 255.0 for c in rgb]] for v, rgb in zip(uniq, palette)]
    try:
        plotter.add_legend(entries, bcolor=None, size=(0.16, min(0.5, 0.035 * len(entries) + 0.05)))
    except Exception:
        # A legend is a nicety; never let it stop the window from opening.
        pass


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("payload", type=Path, help="the .npz written by the viewer")
    parser.add_argument("--smoke", type=Path, default=None,
                        help="render off-screen to this PNG instead of opening a window")
    args = parser.parse_args(argv)

    if not args.payload.is_file():
        print(f"payload not found: {args.payload}", file=sys.stderr)
        return 2

    payload = load_payload(args.payload)
    if payload["xyz"].size == 0:
        print("payload holds no points", file=sys.stderr)
        return 3

    plotter = build_plotter(payload, off_screen=args.smoke is not None)
    if args.smoke is not None:
        plotter.screenshot(str(args.smoke))
        plotter.close()
        print(f"wrote {args.smoke}")
        return 0

    plotter.show()   # blocks this process only; the 2-D viewer keeps running
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
