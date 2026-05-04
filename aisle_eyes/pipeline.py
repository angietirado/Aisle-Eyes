"""Run YOLO tracking + ROI dwell analysis on a video file."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
from ultralytics import YOLO

from .dwell import DwellTracker
from .geometry import roi_from_fractions


def _unpack_tracks(result) -> list[tuple[int, tuple[float, float, float, float]]]:
    if result.boxes is None or len(result.boxes) == 0:
        return []
    ids = result.boxes.id
    if ids is None:
        return []
    xyxy = result.boxes.xyxy.cpu().numpy()
    ids_np = ids.cpu().numpy().astype(int)
    out: list[tuple[int, tuple[float, float, float, float]]] = []
    for i in range(len(ids_np)):
        box = tuple(float(x) for x in xyxy[i])
        out.append((int(ids_np[i]), box))
    return out


def _draw_roi(frame: np.ndarray, roi: tuple[int, int, int, int]) -> None:
    x1, y1, x2, y2 = roi
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
    cv2.putText(
        frame,
        "Display ROI",
        (x1, max(0, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 200, 255),
        2,
        cv2.LINE_AA,
    )


def _draw_track_overlay(
    frame: np.ndarray,
    tid: int,
    box: tuple[float, float, float, float],
    dwell_sec: float,
    inside: bool,
) -> None:
    x1, y1, x2, y2 = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
    color = (0, 255, 128) if inside else (180, 180, 180)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"ID {tid}"
    cv2.putText(
        frame,
        label,
        (x1, max(0, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
        cv2.LINE_AA,
    )
    if inside:
        timer = f"Dwell: {dwell_sec:.1f}s"
        ty = min(frame.shape[0] - 8, y2 + 22)
        cv2.putText(
            frame,
            timer,
            (x1, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )


def run_pipeline(
    input_path: str | Path,
    output_video: str | Path,
    summary_csv: str | Path | None,
    model_name: str = "yolov8n.pt",
    roi_pixels: tuple[int, int, int, int] | None = None,
    roi_fractions: tuple[float, float, float, float] | None = None,
    device: str | None = None,
) -> Path:
    """
    Process video: YOLO person tracking, ROI dwell timers, annotated MP4 + optional CSV.

    roi_pixels: (x1, y1, x2, y2) in pixels. If None, roi_fractions must be set
    (left, top, right, bottom) in [0,1].
    """
    input_path = Path(input_path)
    output_video = Path(output_video)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    probe = cv2.VideoCapture(str(input_path))
    if not probe.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")
    fps = probe.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    probe.release()

    if roi_pixels is not None:
        roi = roi_pixels
    elif roi_fractions is not None:
        lf, tp, rt, bt = roi_fractions
        roi = roi_from_fractions(w, h, lf, tp, rt, bt)
    else:
        roi = roi_from_fractions(w, h, 0.32, 0.22, 0.68, 0.88)

    model = YOLO(model_name)
    tracker = DwellTracker(fps=float(fps))
    roi_f = tuple(float(x) for x in roi)

    writer: cv2.VideoWriter | None = None
    frame_index = 0

    kwargs = {"stream": True, "persist": True, "classes": [0], "verbose": False}
    if device:
        kwargs["device"] = device

    for result in model.track(source=str(input_path), **kwargs):
        frame = result.orig_img
        if frame is None:
            continue
        if writer is None:
            out_h, out_w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                str(output_video), fourcc, float(fps), (out_w, out_h)
            )
            if not writer.isOpened():
                raise RuntimeError(f"Could not open VideoWriter for {output_video}")

        tracks = _unpack_tracks(result)
        tracker.update(frame_index, tracks, roi_f)

        vis = frame.copy()
        _draw_roi(vis, roi)
        states = tracker.states()

        for tid, box in tracks:
            st = states.get(tid)
            inside = st.in_roi if st else False
            dwell_sec = st.total_dwell_sec(float(fps)) if st else 0.0
            _draw_track_overlay(vis, tid, box, dwell_sec, inside)

        writer.write(vis)
        frame_index += 1

    if writer is not None:
        writer.release()

    tracker.finalize_open_visits()

    if summary_csv:
        summary_csv = Path(summary_csv)
        _write_summary_csv(summary_csv, tracker, float(fps))

    return output_video


def _write_summary_csv(path: Path, tracker: DwellTracker, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "track_id",
                "visit_count",
                "total_dwell_sec",
                "avg_visit_sec",
                "visits_sec_json",
            ]
        )
        for tid in sorted(tracker.states().keys()):
            st = tracker.states()[tid]
            visits = st.completed_visits_sec
            total = sum(visits)
            if total <= 0 and st.dwell_frames <= 0:
                continue
            avg = total / len(visits) if visits else 0.0
            w.writerow(
                [
                    tid,
                    len(visits),
                    round(total, 3),
                    round(avg, 3),
                    str([round(v, 3) for v in visits]),
                ]
            )
