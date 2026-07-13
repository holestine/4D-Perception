"""Detection sources: where per-frame 3D detections come from.

A DetectionSource maps a frame to Detections in the canonical box format.
Dataset adapters call it while assembling each Frame, so swapping between
pre-computed files and live inference is a one-line change in the entry
point. Sources that need sensor data (live detectors) receive the raw
point cloud; file-based sources can ignore it.
"""

from abc import ABC, abstractmethod

from perception.frame import Detections


class DetectionSource(ABC):
    """Produces Detections for each frame of a sequence."""

    @abstractmethod
    def get(self, frame_id, points):
        """Return Detections for one frame.

        Parameters
        ----------
        frame_id : int
        points   : ndarray (N, 4)  raw [x, y, z, intensity] LiDAR points

        Returns
        -------
        Detections
        """


class OpenPCDetSource(DetectionSource):
    """Live inference with an OpenPCDet model, cached per frame.

    Works with any dataset adapter that supplies raw LiDAR points.

    Parameters
    ----------
    detector : object with detect_frame(points, frame_id) -> (boxes, scores, names),
        e.g. OpenPCDetDetector from detector.py. Duck-typed so importing this
        module never pulls in torch/OpenPCDet.
    """

    def __init__(self, detector):
        self._detector = detector
        self._cache = {}

    def get(self, frame_id, points):
        if frame_id not in self._cache:
            boxes, scores, names = self._detector.detect_frame(points, frame_id=frame_id)
            self._cache[frame_id] = Detections(boxes=boxes, scores=scores, names=names)
        return self._cache[frame_id]
