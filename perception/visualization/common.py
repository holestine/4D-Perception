"""Constants and helpers shared by the Rerun and MP4 visualizers."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CAR_OBJ_PATH = str(Path(__file__).resolve().parents[1] / "assets" / "car.obj")

# native extents of car.obj (length, width, height in metres); each axis is
# scaled independently to match the detected box
CAR_OBJ_NATIVE_SIZE = np.array([8.95, 3.71, 2.97], dtype=np.float32)

# classes rendered as a scaled car mesh; everything else gets a box primitive
MESH_CLASSES = {"Car", "Van", "Truck"}

_CMAP = plt.get_cmap("tab20")


def track_color(track_id):
    """Stable RGBA uint8 colour for a track ID (tab20, cycled)."""
    return (np.array(_CMAP(track_id % _CMAP.N)) * 255).astype(np.uint8)


def select_visible(detections, det_ids, show_unconfirmed_above):
    """Pick the detections worth drawing for one frame.

    Confirmed tracks are always drawn; unconfirmed detections only when they
    score above show_unconfirmed_above.

    Parameters
    ----------
    detections : perception.frame.Detections
    det_ids    : ndarray (M,)  confirmed track ID per detection (0 = none)
    show_unconfirmed_above : float

    Returns
    -------
    boxes (K, 7), track_ids (K,), names (K,)  — the visible subset
    """
    visible = (det_ids > 0) | (detections.scores > show_unconfirmed_above)
    return (
        np.asarray(detections.boxes)[visible],
        det_ids[visible],
        np.asarray(detections.names)[visible],
    )
