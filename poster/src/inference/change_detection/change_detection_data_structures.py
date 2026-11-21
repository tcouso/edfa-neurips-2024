import pandas as pd
from dataclasses import dataclass
from typing import Tuple


@dataclass
class PixelChangeDetection:
    """
    Stores detected changes for a single pixel.

    Attributes
    ----------
    index : Tuple[int, int]
        The (row, column) index of the pixel.
    detected_changes : pd.Series
        The series of detected changes.
    offset : int
        The offset between actual and predicted signals.
    N : int
        The length of the detection window.
    k : float
        The threshold factor for detecting significant changes.
    """

    index: Tuple[int, int]
    detected_changes: pd.Series
    N: int
    k: float

    def __repr__(self) -> str:
        return f"PixelChangeDetection_{self.index[0]}_{self.index[1]}"

    def __str__(self) -> str:
        return (
            f"PixelChangeDetection_{self.index[0]}_{self.index[1]}"
        )