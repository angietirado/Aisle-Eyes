"""Per-track dwell time state while a person overlaps the ROI."""

from __future__ import annotations

from dataclasses import dataclass, field

from .geometry import bbox_intersects_roi


@dataclass
class TrackDwellState:
    """Cumulative dwell and completed visits for one track ID."""

    dwell_frames: int = 0
    in_roi: bool = False
    visit_start_frame: int | None = None
    last_inside_frame: int | None = None
    completed_visits_sec: list[float] = field(default_factory=list)

    def total_dwell_sec(self, fps: float) -> float:
        """Total time in ROI (completed visits plus current open segment, in seconds)."""
        return self.dwell_frames / fps


class DwellTracker:
    """Updates dwell statistics from per-frame track detections."""

    def __init__(self, fps: float) -> None:
        self.fps = fps
        self._states: dict[int, TrackDwellState] = {}

    def states(self) -> dict[int, TrackDwellState]:
        return self._states

    def update(
        self,
        frame_index: int,
        tracks: list[tuple[int, tuple[float, float, float, float]]],
        roi_xyxy: tuple[float, float, float, float],
    ) -> None:
        for tid, box in tracks:
            inside = bbox_intersects_roi(box, roi_xyxy)
            st = self._states.setdefault(tid, TrackDwellState())
            if inside:
                if not st.in_roi:
                    st.in_roi = True
                    st.visit_start_frame = frame_index
                st.dwell_frames += 1
                st.last_inside_frame = frame_index
            else:
                if st.in_roi:
                    self._close_visit(st)

    def _close_visit(self, st: TrackDwellState) -> None:
        st.in_roi = False
        if st.visit_start_frame is not None and st.last_inside_frame is not None:
            dur_frames = st.last_inside_frame - st.visit_start_frame + 1
            st.completed_visits_sec.append(dur_frames / self.fps)
        st.visit_start_frame = None
        st.last_inside_frame = None

    def finalize_open_visits(self) -> None:
        for st in self._states.values():
            if st.in_roi:
                self._close_visit(st)
