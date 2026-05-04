"""CLI: python -m aisle_eyes"""

from __future__ import annotations

import argparse
from pathlib import Path

from .geometry import parse_roi_pixels
from .pipeline import run_pipeline


def main() -> None:
    p = argparse.ArgumentParser(
        description="AisleEyes: YOLO person tracking + display ROI dwell times."
    )
    p.add_argument(
        "-i",
        "--input",
        type=Path,
        default=None,
        help="Path to your video file (MP4, AVI, MOV, …). Must exist on disk.",
    )
    p.add_argument(
        "--pick",
        action="store_true",
        help="Open a file dialog to choose the input video (no -i needed).",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output annotated video (default: <input>_aisle_eyes.mp4)",
    )
    p.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Write dwell summary CSV (default: next to output with .csv)",
    )
    p.add_argument(
        "--no-csv",
        action="store_true",
        help="Do not write a summary CSV",
    )
    p.add_argument(
        "--roi",
        type=str,
        default=None,
        help="ROI in pixels: x1,y1,x2,y2",
    )
    p.add_argument(
        "--roi-frac",
        type=str,
        default=None,
        help="ROI as fractions: left,top,right,bottom in [0,1] e.g. 0.3,0.2,0.7,0.9",
    )
    p.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Ultralytics YOLO weights (default: yolov8n.pt)",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="torch device, e.g. cuda, cpu, mps",
    )
    args = p.parse_args()

    if args.pick:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        picked = filedialog.askopenfilename(
            title="AisleEyes — select input video",
            filetypes=[
                ("Video", "*.mp4 *.avi *.mov *.mkv *.webm"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
        if not picked:
            raise SystemExit("No file selected.")
        inp = Path(picked).expanduser().resolve()
    elif args.input is not None:
        inp = args.input.expanduser().resolve()
    else:
        raise SystemExit(
            "You must pass a real video path with -i, or use --pick.\n"
            "Example (use your actual file):\n"
            '  python -m aisle_eyes -i "C:\\Users\\YourName\\Videos\\my_clip.mp4" -o out.mp4\n'
            "Example (file dialog):\n"
            "  python -m aisle_eyes --pick -o out.mp4"
        )

    if not inp.is_file():
        raise SystemExit(
            f"Input video not found:\n  {inp}\n\n"
            "-i must point to a file that exists. \"your_video.mp4\" was only an example name.\n"
            "Use the full path to your clip, or run with --pick to browse.\n"
            "Example:\n"
            f'  python -m aisle_eyes -i "{Path.cwd() / "my_recording.mp4"}" -o out.mp4'
        )
    out = args.output
    if out is None:
        out = inp.with_name(f"{inp.stem}_aisle_eyes{inp.suffix}")
    else:
        out = out.expanduser().resolve()

    summary = None if args.no_csv else args.summary
    if summary is None and not args.no_csv:
        summary = out.with_suffix(".csv")

    roi_pixels = None
    roi_frac = None
    if args.roi:
        roi_pixels = parse_roi_pixels(args.roi)
    elif args.roi_frac:
        parts = [float(x.strip()) for x in args.roi_frac.split(",")]
        if len(parts) != 4:
            raise SystemExit("--roi-frac needs four values: left,top,right,bottom")
        roi_frac = tuple(parts)

    run_pipeline(
        input_path=inp,
        output_video=out,
        summary_csv=summary,
        model_name=args.model,
        roi_pixels=roi_pixels,
        roi_fractions=roi_frac,
        device=args.device,
    )
    print(f"Wrote: {out}")
    if summary:
        print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
