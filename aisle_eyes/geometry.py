"""ROI and bounding-box helpers."""

from __future__ import annotations

import numpy as np


def bbox_intersects_roi(
    xyxy: np.ndarray | tuple[float, ...],
    roi_xyxy: tuple[float, float, float, float],
) -> bool:
    """True if axis-aligned bbox intersects the ROI rectangle."""
    x1, y1, x2, y2 = (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]))
    rx1, ry1, rx2, ry2 = roi_xyxy
    return not (x2 < rx1 or x1 > rx2 or y2 < ry1 or y1 > ry2)


def roi_from_fractions(
    width: int,
    height: int,
    left: float,
    top: float,
    right: float,
    bottom: float,
) -> tuple[int, int, int, int]:
    """Build pixel ROI from fractional bounds in [0, 1]."""
    x1 = int(round(left * width))
    y1 = int(round(top * height))
    x2 = int(round(right * width))
    y2 = int(round(bottom * height))
    return (x1, y1, x2, y2)


def parse_roi_pixels(spec: str) -> tuple[int, int, int, int]:
    """Parse 'x1,y1,x2,y2' into integers."""
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI must be four comma-separated integers: x1,y1,x2,y2")
    return tuple(int(p) for p in parts)  # type: ignore[return-value]
