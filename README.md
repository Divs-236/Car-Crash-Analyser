# Motion_v2 — Dashcam Crash Detector

A computer-vision pipeline that analyses dashcam footage and flags two kinds of crashes:

- **Between-car crash**: two other vehicles collide in front of the camera.
- **Recording-vehicle crash**: the camera car itself is hit or hits something.

It combines **YOLOv8** vehicle detection, **DeepSORT** tracking and **Farneback optical flow**, then looks at four motion signals to decide whether a crash occurred. It produces an annotated video plus diagnostic plots.

## About this project

This is a **course project** built for a **Signals and Systems** class. The crash detection treats quantities extracted from video (bounding-box width, box overlap, lateral position and optical-flow magnitude) as time-domain signals, and applies signal-processing techniques to them: smoothing with a Savitzky–Golay filter, differentiation to obtain rates of change, and threshold-based event detection.

## How it works

1. **Detect**: YOLOv8 finds cars, buses and trucks (COCO classes 2, 5, 7) in the lower two-thirds of each frame.
2. **Track**: DeepSORT assigns each vehicle a persistent ID.
3. **Extract signals** per vehicle / per pair / per frame:

| Signal | What it measures | Used for |
|--------|------------------|----------|
| 1. Bounding-box width | Growth in box width, a proxy for closing distance | Detecting vehicles rapidly approaching |
| 2. Inter-vehicle IoU | Box overlap between two tracked vehicles | Detecting contact between cars |
| 3. Lateral position | Horizontal centre of each vehicle | Spotting swerving |
| 4. Optical flow | Mean motion magnitude across the whole frame | Detecting a violent jolt of the camera (recording-vehicle crash) |

4. **Decide**:
   - **Between-car crash**: flagged when, for 3 consecutive samples, the smoothed closing rate is above `3.0` px/sample **and** IoU is above `0.07`.
   - **Recording-vehicle crash**: flagged when smoothed optical flow is above `5.0` for 2 consecutive frames. The most likely "target" vehicle is the one whose box grew most, near the frame centre, in the 100 samples before the event.

Detection and tracking run on every 2nd frame for speed; optical flow runs on every frame. Skipped frames reuse the last annotated frame so the output video keeps its original length.

On the live preview, boxes are coloured by threat level:
🟢 green = normal, 🟠 orange = closing moderately fast, 🔴 red = very close or closing quickly.

## Requirements

- Python 3.9+
- A YOLOv8 weights file (`yolov8n.pt`); Ultralytics downloads it automatically if missing

```bash
pip install opencv-python ultralytics deep-sort-realtime matplotlib numpy scipy
```

## Usage

1. Put your video in the project folder (the script currently reads `test5.mp4`).
2. To use a different clip, edit this line in `detect.py`:
   ```python
   cap = cv.VideoCapture("test5.mp4")
   ```
3. Run:
   ```bash
   python detect.py
   ```
4. Press **`q`** in the preview window to stop early.

A summary is printed to the console when the video ends.

## Outputs

| File | Description |
|------|-------------|
| `output_annotated.mp4` | Video with boxes, IDs, vehicle count and frame counter |
| `signal1_width.png` | Bounding-box width over time per vehicle |
| `signal2_iou.png` | Inter-vehicle IoU |
| `signal3_centers.png` | Lateral position over time |
| `signal4_record_flow.png` | Raw vs smoothed optical flow with threshold |
| `crash_between_<A>_<B>.png` | Detailed breakdown if a between-car crash is flagged |
| `crash_record.png` | Detailed breakdown if a recording-vehicle crash is flagged |

## Tuning

The main thresholds live in `detect.py`:

| Parameter | Default | Effect |
|-----------|---------|--------|
| YOLO `conf` | `0.3` | Minimum detection confidence |
| DeepSORT `n_init` / `max_age` | `10` / `20` | Samples to confirm a track / to drop a lost one |
| Closing-rate threshold | `3.0` px/sample | Sensitivity of between-car detection |
| IoU threshold | `0.07` | Required overlap for between-car detection |
| Optical-flow threshold | `5.0` | Sensitivity of recording-vehicle detection |
| Consecutive samples | `3` (pair) / `2` (flow) | Robustness against single-frame noise |

## Project structure

```
Car Crash Analyser/
├── detect.py            # main script
├── yolov8n.pt           # YOLOv8 nano weights
├── test*.mp4            # sample dashcam clips
├── output_annotated.mp4 # generated output
├── signal*.png          # generated plots
├── LICENSE
└── README.md
```

## Limitations

- Thresholds are tuned on a small set of clips and may need adjusting for other cameras, resolutions or frame rates.
- Bounding-box width is only a rough proxy for distance; it can be thrown off by turns, lane changes or detector jitter.
- The recording-vehicle detector relies on whole-frame optical flow, so heavy camera shake or bumpy roads can cause false positives.
- Only the first flagged event of each type is reported.

## License

Released under the [MIT License](LICENSE).
