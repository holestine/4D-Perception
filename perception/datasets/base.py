"""Base interface every dataset adapter implements."""

from abc import ABC, abstractmethod

from perception.frame import Frame


class SequenceDataset(ABC):
    """A time-ordered driving sequence of sensor frames.

    Adapters own all dataset-specific I/O and conventions; each frame they
    return is a dataset-agnostic Frame with canonical-format detections.
    """

    @abstractmethod
    def __len__(self):
        """Number of frames in the sequence."""

    @abstractmethod
    def __getitem__(self, idx) -> Frame:
        """Load all sensor data and detections for frame `idx`."""
