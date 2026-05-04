# Aisle-Eyes

Aisle-Eyes is an SDET 102 computer vision project that uses Ultralytics YOLO to detect and track people in a video, measure how long each person stays inside a target display area (ROI), and export both an annotated output video and dwell-time summary CSV.

## What the project does

- Detects `person` objects in each frame with YOLO.
- Tracks people across frames with persistent track IDs.
- Measures dwell time when each tracked person overlaps a display ROI.
- Saves:
  - an annotated video with boxes, IDs, ROI, and timer overlay
  - a CSV report with per-person visit and dwell statistics

## How to run the code

1. Open a terminal in the project folder.
2. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Run the app:

   ```bash
   python -m aisle_eyes -i "path/to/input_video.mp4" -o output_annotated.mp4
   ```

4. Optional ROI controls:

   ```bash
   # Pixel ROI
   python -m aisle_eyes -i input.mp4 -o out.mp4 --roi 120,80,520,400

   # Fractional ROI (left,top,right,bottom)
   python -m aisle_eyes -i input.mp4 -o out.mp4 --roi-frac 0.3,0.2,0.7,0.85
   ```

5. Optional file picker:

   ```bash
   python -m aisle_eyes --pick -o out.mp4
   ```

## What files are included

- `aisle_eyes/__main__.py`: command-line entry point.
- `aisle_eyes/pipeline.py`: end-to-end video processing pipeline.
- `aisle_eyes/dwell.py`: dwell-time tracking state and visit aggregation.
- `aisle_eyes/geometry.py`: ROI and box intersection helpers.
- `AisleEyes_Colab.ipynb`: Colab notebook workflow.
- `requirements.txt`: required Python packages.
- `.gitignore`: ignores generated outputs and large artifacts.

## Required libraries and setup steps

- Python 3.10+ recommended.
- Required packages in `requirements.txt`:
  - `ultralytics`
  - `opencv-python`
  - `numpy`
  - `lap` (tracker dependency)

On first run, YOLO weights (for example `yolov8n.pt`) may download automatically if not already present.

## Where input data comes from

- User-provided prerecorded video files (`.mp4`, `.avi`, `.mov`, etc.).
- Videos can come from:
  - phone/laptop recordings
  - store/security-style footage (if permission is granted)
  - public pedestrian datasets or licensed online clips

## Limitations and special notes

- Dwell metrics depend on camera angle, ROI placement, lighting, and crowd density.
- Occlusion and heavy overlap can cause track ID switches and timing errors.
- Fast movement through ROI may produce very short visits.
- CSV track IDs are internal tracker IDs, not person identity.
- Keep file paths quoted on Windows when folders contain spaces.
