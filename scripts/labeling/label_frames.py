"""
Open DeepLabCut's labelling window on one folder, and keep it open until you close it.
If drawing fails, it prints the real error.

Usage (dlc environment):
    python scripts/labeling/label_frames.py cheetah-dlc/config.yaml lily_brake1_cam1
    python scripts/labeling/label_frames.py cheetah-dlc/config.yaml lily_brake1_cam1 1   # open on screen 1

Under WSL the window needs these (environment.yml sets them in the dlc env):
    QT_QPA_PLATFORM=xcb QT_XCB_GL_INTEGRATION=xcb_glx PYOPENGL_PLATFORM=glx
"""
import faulthandler
import os
import sys
import traceback

faulthandler.enable()  # prints where it died if the crash happens outside Python

from vispy.visuals import visual as vispy_visual  # noqa: E402

_original_draw = vispy_visual.Visual.draw
_printed = []


def _draw_and_report(self):
    try:
        return _original_draw(self)
    except Exception:
        if not _printed:  # the same error repeats every frame; print it once
            _printed.append(True)
            print("\n=== the real error ===", file=sys.stderr)
            traceback.print_exc()
            print("=== end ===\n", file=sys.stderr)
        raise


vispy_visual.Visual.draw = _draw_and_report

import deeplabcut  # noqa: E402

import napari  # noqa: E402

config = os.path.abspath(os.path.expanduser(sys.argv[1]))
folder = os.path.join(os.path.dirname(config), "labeled-data", sys.argv[2])
deeplabcut.label_frames(config, image_folder=folder)
viewer = napari.current_viewer()
if viewer is not None:
    # Write each point's name next to it (untick "display text" in the layer controls to hide them).
    from napari.layers import Points

    def show_names(layer):
        if isinstance(layer, Points):
            layer.text = {"string": "{label}", "size": 8, "color": "white", "anchor": "upper_left"}

    for layer in viewer.layers:
        show_names(layer)
    viewer.layers.events.inserted.connect(lambda event: show_names(event.value))

    # WSL's title bar can be unresponsive, so place the window ourselves:
    # full size, on the screen given as an optional third argument (0, 1, ...).
    from qtpy.QtGui import QGuiApplication

    screens = QGuiApplication.screens()
    for i, s in enumerate(screens):
        g = s.geometry()
        print(f"screen {i}: {g.width()} x {g.height()} at ({g.x()}, {g.y()})")
    pick = int(sys.argv[3]) if len(sys.argv) > 3 else None
    screen = screens[pick] if pick is not None and 0 <= pick < len(screens) else QGuiApplication.primaryScreen()
    win = viewer.window._qt_window
    win.showNormal()
    win.move(screen.availableGeometry().topLeft())
    win.showMaximized()
print("Close the window with File > Exit (Ctrl+Q). Save with Ctrl+S first.")
napari.run()  # keep the window open until you close it (label_frames alone returns straight away)