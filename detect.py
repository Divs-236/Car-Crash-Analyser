"""
Motion-Dashcam crash Analyser
 
Detects two kinds of crashes in a video:
  1. Between-car crash : two other vehicles collide in front of the camera.
  2. Recording-vehicle crash : the camera car itself is hit / hits something.
 
Pipeline:
  YOLOv8 (vehicle detection) -> DeepSORT (ID tracking) -> four signals:
    Signal 1 - bbox width growth   (proxy for closing distance)
    Signal 2 - pairwise box IoU    (overlap between two vehicles)
    Signal 3 - lateral position    (swerving indicator)
    Signal 4 - global optical flow (violent camera motion = recording car crash)
"""

import cv2 as cv
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

model = YOLO("yolov8n.pt")  #pretrained object detector
cap = cv.VideoCapture("test5.mp4")  #video of interest
tracker = DeepSort(max_age=20, n_init=10)  #Used for tracking objects, objects needs to be seen for 10 frames continuously to be tracked, 
                                           #tracking is dropped if object is not visible after 20 frames

track_signals = {}  #track_id -> {"widths": [...], "centers": [...]}
pair_iou      = {}  # (idA, idB) -> [IoU per sample]

def compute_iou(boxA, boxB):
    """IoU of two [x1, y1, x2, y2] boxes, with a size-similarity pre-filter."""
    wA = boxA[2] - boxA[0]
    wB = boxB[2] - boxB[0]
    if wA <= 0 or wB <= 0:
        return 0.0
    if min(wA, wB) / max(wA, wB) < 0.2:
        return 0.0

    # Ignore pairs where one box is much smaller than the other (e.g. a far car
    # "behind" a near one). Overlap between very different sizes is just perspective,
    # not a collision.

    xA = max(boxA[0], boxB[0]);  yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]);  yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


def get_box_color(track_id, track_signals, frame_width):
    """Picks the on-screen box colour (BGR) from how fast / how close a vehicle is."""
    signals = track_signals.get(track_id, {})
    widths  = signals.get("widths", [])

    live_closing_rate = widths[-1] - widths[-2] if len(widths) >= 2 else 0
    current_width     = widths[-1] if widths else 0

    if current_width > frame_width * 0.45 or live_closing_rate > 9:
        return (0, 0, 255)  #Red: very close
    if live_closing_rate > 3:
        return (0, 165, 255)  #Orange: Closing Moderately fast  
    return (0, 255, 0)  #Green: Relatively safe    


def smooth(arr):
    """Savitzky-Golay smoothing to reduce detector jitter."""
    n = len(arr)
    if n < 4:
        return np.array(arr, dtype=float)
    win = min(7, n if n % 2 != 0 else n - 1)
    return savgol_filter(arr, window_length=win, polyorder=1)


""" VIDEO I/O SETUP     """


if not cap.isOpened():
    print("Video Not loaded")

total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
fps          = cap.get(cv.CAP_PROP_FPS)
frame_count  = 0

#Writer for the annotated output video.
w_frame = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
h_frame = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fourcc  = cv.VideoWriter_fourcc(*"mp4v")
out_writer = cv.VideoWriter("output_annotated.mp4", fourcc, fps, (w_frame, h_frame))

prev_gray = None  #previous grayscale frame, needed for optical flow
flow_signal = []  #mean optical-flow magnitude
confirmed_boxes = {}  #track_id -> box in full-frame coordinates
last_frame = None  #last annotated frame, re-used on skipped frames

while True:
    ret, frame = cap.read()
    if not ret:
        print("Video Ended")
        break

    frame_count += 1

    #Signal 4: global optical flow (computed on EVERY frame)
    #A sudden spike in average motion across the whole image means the camera
    #itself was jolted, i.e. the recording vehicle was hit

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    if prev_gray is not None:
        flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        magnitude, _ = cv.cartToPolar(flow[..., 0], flow[..., 1])
        flow_signal.append(float(np.mean(magnitude)))
    else:
        flow_signal.append(0.0)
    prev_gray = gray


    #Frame skipping: run detection/tracking on every 2nd frame only
    #Odd frames are written out unchanged so the output video keeps the original length and fps.
    if frame_count % 2 != 0:
        if last_frame is not None:
            out_writer.write(last_frame)
        else:
            out_writer.write(frame)
        continue


    #Region of interest: drop the top third (sky / buildings)
    h, w = frame.shape[:2]
    crop_start = h // 3
    roi = frame[crop_start:h, 0:w]

    #Vehicle detection with YOLO
    results    = model(roi, conf=0.3, imgsz=1280, verbose=False)
    detections = results[0].boxes

    vehicle_count = 0
    ds_input = []  #detections formatted for DeepSORT

    for box in detections:
        class_id   = int(box.cls[0])
        confidence = float(box.conf[0])
        if class_id not in [2, 5, 7]:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        ds_input.append(([x1, y1, x2 - x1, y2 - y1], confidence, class_id))

    tracks = tracker.update_tracks(ds_input, frame=roi)

    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 0:
            continue

        track_id = track.track_id
        det = track.get_det_supplementary()
        if det is not None:
            x1, y1, bw, bh = map(int, det[0])
            x2 = x1 + bw
        else:
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            bw = x2 - x1

        y1 += crop_start
        y2 = y1 + (bw if det is None else bh)

        det2 = track.get_det_supplementary()
        if det2 is not None:
            x1, y1, bw, bh = map(int, det2[0])
            x2 = x1 + bw
            y2 = y1 + bh
            y1 += crop_start
            y2 += crop_start
        else:
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            bw = x2 - x1
            y1 += crop_start
            y2 += crop_start

        center = (x1 + x2) // 2

        #Signals 1 & 3: store width and lateral position per track
        if track_id not in track_signals:
            track_signals[track_id] = {"widths": [], "centers": []}
        track_signals[track_id]["widths"].append(bw)
        track_signals[track_id]["centers"].append(center)

        vehicle_count += 1

        color = get_box_color(track_id, track_signals, w)
        cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"#{track_id}"
        (tw, th), _ = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv.putText(frame, label, (x1 + 2, y1 - 3),
                   cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv.LINE_AA)

    #HUD overlay: vehicle count and frame counter
    cv.putText(frame, f"Vehicles: {vehicle_count}", (10, 60),
               cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)
    cv.putText(frame, f"Frame: {frame_count}/{total_frames}", (10, 90),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv.LINE_AA)


    #Signal 2: pairwise IoU between every pair of visible vehicles
    confirmed_boxes = {}
    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 0:
            continue
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        confirmed_boxes[track.track_id] = [x1, y1 + crop_start, x2, y2 + crop_start]

    ids = list(confirmed_boxes.keys())
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            idA = ids[i];  idB = ids[j]
            pair_key = (min(idA, idB), max(idA, idB))
            iou = compute_iou(confirmed_boxes[idA], confirmed_boxes[idB])
            if pair_key not in pair_iou:
                pair_iou[pair_key] = []
            pair_iou[pair_key].append(iou)

    #Save and write the annotated frame; press 'q' to stop earl
    last_frame = frame.copy()
    out_writer.write(frame)

    cv.imshow("Crash Detector", frame)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out_writer.release()
cv.destroyAllWindows()

#Keep only tracks seen long enough (>= 10 samples) to be plotted meaningfully
valid_tracks = {tid: sig for tid, sig in track_signals.items()
                if len(sig["widths"]) >= 10}

#Between-car crash detection
#A pair is flagged when, for 3 consecutive samples, at least one car's box is
#growing quickly (closing rate > 3 px/sample) AND the two boxes overlap (IoU > 0.07)
crashed_pair   = None
crashed_sample = -1

for (idA, idB), iou_vals in pair_iou.items():
    sigA = track_signals.get(idA, {})
    sigB = track_signals.get(idB, {})
    wA   = sigA.get("widths", [])
    wB   = sigB.get("widths", [])
    if len(wA) < 10 or len(wB) < 10 or len(iou_vals) < 5:
        continue

    #Smooth the width signals, then differentiate to get closing rate (px/sample)
    s_wA = smooth(wA)
    s_wB = smooth(wB)
    cr_A = np.diff(s_wA)
    cr_B = np.diff(s_wB)

    #Trim everything to a common length so the arrays line up
    iou_arr = np.array(iou_vals)
    min_len = min(len(cr_A), len(cr_B), len(iou_arr))
    cr_A    = cr_A[:min_len]
    cr_B    = cr_B[:min_len]
    iou_arr = iou_arr[:min_len]

    cr_combined = np.maximum(cr_A, cr_B)

    #Trim everything to a common length so the arrays line u
    consecutive = 0
    for i in range(min_len):
        if cr_combined[i] > 3.0 and iou_arr[i] > 0.07:
            consecutive += 1
        else:
            consecutive = 0
        if consecutive >= 3:
            crashed_pair   = (idA, idB)
            crashed_sample = i
            break
    if crashed_pair:
        break


#Flagged when smoothed mean flow > 5.0 for 2 consecutive frames.
record_crash   = False
record_sample  = -1
smooth_flow = None

if len(flow_signal) > 10:
    smooth_flow = smooth(flow_signal)
    consecutive = 0
    for i, val in enumerate(smooth_flow):
        if val > 5.0:
            consecutive += 1
        else:
            consecutive = 0
        if consecutive >= 2:
            record_crash  = True
            record_sample = i
            break


#Identify which vehicle the recording car hit
# Score each track by how much its box grew in the 100 samples before the crash,
# weighted by how close it was to the centre of the frame (i.e. directly ahead).
record_target = None
if record_crash:
    yolo_sample = record_sample // 2
    frame_mid_x = w_frame // 2
    best_score  = -1
    for track_id, signals in track_signals.items():
        widths  = signals["widths"]
        centers = signals["centers"]
        if len(widths) < 3:
            continue

        #Look-back window: up to 100 samples before the crash
        lb_start = max(0, yolo_sample - 100)
        lb_end   = max(1, yolo_sample)
        w_window = widths[lb_start:lb_end]
        c_window = centers[lb_start:lb_end]
        if not w_window:
            continue
        width_growth = w_window[-1] - w_window[0]
        avg_center   = sum(c_window) / len(c_window)
        center_score = 1.0 / (1.0 + abs(avg_center - frame_mid_x))
        score = width_growth * center_score
        if score > best_score:
            best_score = score
            record_target = track_id

if crashed_pair:
    print(f"  Between-car crash : FLAGGED — Cars #{crashed_pair[0]} & #{crashed_pair[1]} at sample {crashed_sample}")
else:
    print(f"  Between-car crash : None detected")
print(f"  Recording Vehicle vehicle crash : {'FLAGGED at sample ' + str(record_sample) if record_crash else 'None detected'}")
print(f"  Recording Vehicle crash target  : {'Car #' + str(record_target) if record_target else 'None detected'}")


#Signal 1: bounding-box width over time for every valid track.
#A rapidly rising line = a vehicle getting closer.
plt.figure(figsize=(12, 4))
for track_id, signals in valid_tracks.items():
    plt.plot(signals["widths"], label=f"Car #{track_id}")
plt.xlabel("YOLO Sample")
plt.ylabel("Bounding Box Width (px)")
plt.title("Signal 1 — Closing Distance Proxy (bbox width over time)")
plt.legend(fontsize=7)
plt.tight_layout()
plt.savefig("signal1_width.png")
plt.show()


#Signal 2: inter-vehicle IoU.
#If a crash was found, show only that pair; otherwise show pairs whose overlap
#ever exceeded the threshold (excluding pairs involving the recording-crash target).
plt.figure(figsize=(12, 4))
iou_plotted = 0

if crashed_pair:
    iou_vals = pair_iou[crashed_pair]
    plt.plot(iou_vals, color="red", linewidth=1.5,
             label=f"Cars #{crashed_pair[0]} & #{crashed_pair[1]} (CRASHED)")
    plt.axvline(crashed_sample, color="red", linestyle="--",
                label=f"Crash @ sample {crashed_sample}")
    iou_plotted += 1
else:
    record_pairs = set()
    if record_target:
        for key in pair_iou:
            if record_target in key:
                record_pairs.add(key)
    for key, iou_vals in pair_iou.items():
        if key in record_pairs:
            continue
        if len(iou_vals) < 5 or max(iou_vals) < 0.07:
            continue
        plt.plot(iou_vals, label=f"Cars #{key[0]} & #{key[1]}")
        iou_plotted += 1

if iou_plotted == 0:
    plt.text(0.5, 0.5,
             "No inter-vehicle IoU exceeded threshold" +
             ("\n(recording vehicle involved — IoU not applicable)" if record_crash else ""),
             ha="center", va="center", transform=plt.gca().transAxes, fontsize=10)

plt.axhline(0.07, color="gray", linestyle="--", alpha=0.7,
            label=f"Threshold ({0.07})")
plt.xlabel("Pair Sample")
plt.ylabel("IoU")
plt.title("Signal 2 — Inter-vehicle IoU (crashed pair only / size-filtered)")
plt.legend(fontsize=7)
plt.tight_layout()
plt.savefig("signal2_iou.png")
plt.show()



#Signal 3: horizontal position of each vehicle. Sudden sideways jumps suggest swerving.
plt.figure(figsize=(12, 4))
for track_id, signals in valid_tracks.items():
    if len(signals["centers"]) >= 10:
        plt.plot(signals["centers"], label=f"Car #{track_id}")
plt.axhline(w_frame // 2, color="white", linestyle="--", alpha=0.5,
            label="Frame centre")
plt.xlabel("YOLO Sample")
plt.ylabel("Center-X (px)")
plt.title("Signal 3 — Lateral Position over time (swerving indicator)")
plt.legend(fontsize=7)
plt.tight_layout()
plt.savefig("signal3_centers.png")
plt.show()


#Signal 4: raw vs smoothed optical flow with the detection threshold
if smooth_flow is not None:
    plt.figure(figsize=(12, 4))
    plt.plot(flow_signal, alpha=0.4, label="Raw flow")
    plt.plot(smooth_flow,            label="Smoothed flow")
    plt.axhline(5.0, color="red", linestyle="--", alpha=0.7,
                label=f"Threshold ({5.0})")
    if record_crash:
        plt.axvline(record_sample, color="red", linestyle="--",
                    label=f"Crash @ frame {record_sample}")
    plt.xlabel("Frame")
    plt.ylabel("Mean Optical Flow Magnitude")
    plt.title("Signal 4 — Recording Vehicle Optical Flow")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig("signal4_record_flow.png")
    plt.show()


#Detailed 3-panel breakdown for the between-car crash:
#widths, combined closing rate, and IoU, all aligned on the flagged sample
if crashed_pair:
    idA, idB   = crashed_pair
    wA  = track_signals[idA]["widths"]
    wB  = track_signals[idB]["widths"]
    iou_vals   = pair_iou[crashed_pair]
    s_wA = smooth(wA);  s_wB = smooth(wB)
    cr_A = np.diff(s_wA);  cr_B = np.diff(s_wB)
    iou_arr = np.array(iou_vals)
    min_len = min(len(cr_A), len(cr_B), len(iou_arr))
    cr_combined = np.maximum(cr_A[:min_len], cr_B[:min_len])

    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    fig.suptitle(f"Between-Car Crash — Cars #{idA} & #{idB}  (flagged @ sample {crashed_sample})",
                 fontweight="bold")

    axes[0].plot(wA, alpha=0.35, color="steelblue",  label=f"Raw #{idA}")
    axes[0].plot(s_wA,           color="steelblue",  label=f"Smooth #{idA}", linewidth=1.5)
    axes[0].plot(wB, alpha=0.35, color="darkorange", label=f"Raw #{idB}")
    axes[0].plot(s_wB,           color="darkorange", label=f"Smooth #{idB}", linewidth=1.5)
    axes[0].axvline(crashed_sample, color="red", linestyle="--", label="Crash")
    axes[0].set_ylabel("Width (px)")
    axes[0].set_title("Closing Distance (both cars)")
    axes[0].legend(fontsize=7)

    axes[1].plot(cr_combined, color="orange", label="Combined closing rate")
    axes[1].axhline(3.0, color="orange", linestyle="--", alpha=0.6,
                    label=f"Threshold ({3.0})")
    axes[1].axvline(crashed_sample, color="red", linestyle="--")
    axes[1].set_ylabel("px / sample")
    axes[1].set_title("Closing Rate (max of both cars)")
    axes[1].legend(fontsize=7)

    axes[2].plot(iou_arr[:min_len], color="purple", label="IoU")
    axes[2].axhline(0.07, color="purple", linestyle="--", alpha=0.6,
                    label=f"Threshold ({0.07})")
    axes[2].axvline(crashed_sample, color="red", linestyle="--", label="Crash")
    axes[2].set_xlabel("Pair Sample")
    axes[2].set_ylabel("IoU")
    axes[2].set_title("Inter-vehicle IoU")
    axes[2].legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(f"crash_between_{idA}_{idB}.png")
    plt.show()


#2-panel breakdown for the recording-vehicle crash:
#optical flow on top, the suspected target car's width below
if record_crash and smooth_flow is not None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    fig.suptitle(f"Recording Vehicle Crash — flow flagged @ frame {record_sample}"
                 + (f", target: Car #{record_target}" if record_target else ""),
                 fontweight="bold")

    axes[0].plot(flow_signal, alpha=0.4, label="Raw flow")
    axes[0].plot(smooth_flow,            label="Smoothed flow")
    axes[0].axhline(5.0, color="red", linestyle="--",
                    label=f"Threshold ({5.0})")
    axes[0].axvline(record_sample, color="red", linestyle="--",
                    label=f"Crash @ {record_sample}")
    axes[0].set_ylabel("Mean Magnitude")
    axes[0].set_title("Recording Vehicle Optical Flow")
    axes[0].legend(fontsize=7)

    if record_target and record_target in track_signals:
        w = track_signals[record_target]["widths"]
        frame_indices = [i * 2 for i in range(len(w))]
        axes[1].plot(frame_indices, w, label=f"Car #{record_target} width")
        axes[1].axvline(record_sample, color="red", linestyle="--",
                        label=f"Crash @ frame {record_sample}")
        axes[1].set_xlabel("Frame")
        axes[1].set_ylabel("Width (px)")
        axes[1].set_title(f"Recording Vehicle Target — Car #{record_target} Closing Distance")
        axes[1].legend(fontsize=7)
    else:
        axes[1].set_title("Recording Vehicle target not identified")

    plt.tight_layout()
    plt.savefig("crash_record.png")
    plt.show()