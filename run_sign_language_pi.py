"""
run_sign_language_pi.py — Real-time sign language detection for Raspberry Pi 4.

Optimized differences vs run_sign_language.py:
  - Uses TFLite interpreter (tflite-runtime) instead of full TensorFlow
  - Camera capture runs on a background thread so MediaPipe + inference
    never block on frame acquisition
  - Default camera resolution reduced to 480×360 (cuts MediaPipe CPU usage
    by ~40% vs 1280×720 with negligible accuracy impact on landmarks)
  - MediaPipe runs in IMAGE mode with explicit frame skipping so the CPU
    is not saturated (process every Nth frame, N configurable)
  - No ONNX/DirectML dependency (not available on Pi)
  - --no-display flag suppresses the OpenCV window for headless SSH use

Hardware target:
  Raspberry Pi 4 (any RAM), ARM Cortex-A72, Raspberry Pi OS (64-bit)
  Tested with: tflite-runtime 2.14, mediapipe 0.10, opencv-python-headless 4.x

Usage:
  python run_sign_language_pi.py
  python run_sign_language_pi.py --model pi_models/classifier_int8.tflite --no-display
  python run_sign_language_pi.py --source /dev/video0 --width 320 --height 240
"""

import argparse
import collections
import json
import queue
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Sign language detection — Pi 4 optimized")
    p.add_argument("--model",  default="pi_models/classifier_fp16.tflite",
                   help="Path to TFLite classifier model")
    p.add_argument("--task",   default="mediapipe_models/hand_landmarker.task",
                   help="Path to MediaPipe hand_landmarker.task bundle")
    p.add_argument("--labels", default="karsl_words_labels.json",
                   help="Labels JSON file")
    p.add_argument("--source", default="0",
                   help="Camera index (int) or video file path")
    p.add_argument("--width",  type=int, default=480, help="Capture width")
    p.add_argument("--height", type=int, default=360, help="Capture height")
    p.add_argument("--skip-frames", type=int, default=1,
                   help="Run MediaPipe on every Nth frame (1 = every frame, 2 = every other, …). "
                        "Increase on slower Pi models.")
    p.add_argument("--num-threads", type=int, default=4,
                   help="TFLite interpreter threads (Pi 4 has 4 cores)")
    p.add_argument("--confidence-threshold", type=float, default=0.50,
                   help="Minimum softmax confidence to commit a prediction")
    p.add_argument("--smoothing-window", type=int, default=10,
                   help="How many consecutive identical predictions before accepting a word")
    p.add_argument("--no-display", action="store_true",
                   help="Suppress the OpenCV window (for headless SSH operation)")
    p.add_argument("--test", action="store_true",
                   help="Run for --test-frames frames then exit")
    p.add_argument("--test-frames", type=int, default=60)
    return p.parse_args()

# ---------------------------------------------------------------------------
# MediaPipe setup
# ---------------------------------------------------------------------------

SEQUENCE_LENGTH = 30
FEATURE_LENGTH  = 126  # 21 landmarks × 3 coords × 2 hands

def build_landmarker(task_path: str):
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    base_options = mp_python.BaseOptions(model_asset_path=task_path)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,   # simpler than VIDEO on Pi; no timestamp tracking
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def extract_keypoints(result) -> np.ndarray:
    """Convert a MediaPipe HandLandmarkerResult into a (126,) normalized float32 vector."""
    left  = np.zeros(63, dtype=np.float32)
    right = np.zeros(63, dtype=np.float32)

    for i, hand_landmarks in enumerate(result.hand_landmarks):
        handedness_label = result.handedness[i][0].category_name  # "Left" or "Right"

        pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32)

        # Wrist-center translation invariance
        pts -= pts[0]

        # Scale invariance: normalize by max 2-D distance from wrist
        scale = np.max(np.linalg.norm(pts[:, :2], axis=1))
        if scale > 1e-6:
            pts /= scale

        flat = pts.flatten()
        if handedness_label == "Left":
            left = flat
        else:
            right = flat

    return np.concatenate([left, right])

# ---------------------------------------------------------------------------
# TFLite classifier wrapper
# ---------------------------------------------------------------------------

def build_interpreter(model_path: str, num_threads: int):
    try:
        import tflite_runtime.interpreter as tflite
        interp = tflite.Interpreter(model_path=model_path, num_threads=num_threads)
    except ImportError:
        import tensorflow as tf
        interp = tf.lite.Interpreter(model_path=model_path, num_threads=num_threads)
    interp.allocate_tensors()
    return interp


def run_classifier(interpreter, sequence: np.ndarray) -> np.ndarray:
    """Run a single (30, 126) sequence through the TFLite interpreter; return (10,) probs."""
    inp_idx = interpreter.get_input_details()[0]["index"]
    out_idx = interpreter.get_output_details()[0]["index"]
    interpreter.set_tensor(inp_idx, sequence[np.newaxis].astype(np.float32))
    interpreter.invoke()
    return interpreter.get_tensor(out_idx)[0]

# ---------------------------------------------------------------------------
# Camera thread
# ---------------------------------------------------------------------------

class CameraReader(threading.Thread):
    """Reads frames from the camera in a background thread.

    The main loop always gets the latest frame without blocking on capture.
    """
    def __init__(self, source, width, height):
        super().__init__(daemon=True)
        self._source = source
        self._width  = width
        self._height = height
        self._q      = queue.Queue(maxsize=2)
        self._stop_evt = threading.Event()

    def run(self):
        src = int(self._source) if str(self._source).isdigit() else self._source
        cap = cv2.VideoCapture(src)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open camera source: {self._source}", file=sys.stderr)
            self._stop_evt.set()
            return

        while not self._stop_evt.is_set():
            ok, frame = cap.read()
            if not ok:
                break
            # Drop old frames to avoid latency accumulation
            if self._q.full():
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    pass
            self._q.put(frame)
        cap.release()

    def read(self):
        """Return the latest frame or None if not yet available."""
        try:
            return self._q.get(timeout=0.1)
        except queue.Empty:
            return None

    def stop(self):
        self._stop_evt.set()

# ---------------------------------------------------------------------------
# Probability bar overlay
# ---------------------------------------------------------------------------

def draw_prob_bars(frame, probs, labels, top_n=5):
    h, w = frame.shape[:2]
    bar_w  = 200
    bar_h  = 18
    x_off  = w - bar_w - 10
    y_start = 10

    top_idx = np.argsort(probs)[::-1][:top_n]
    for rank, idx in enumerate(top_idx):
        label = labels[idx] if labels else str(idx)
        conf  = probs[idx]
        y = y_start + rank * (bar_h + 4)

        # Background
        cv2.rectangle(frame, (x_off, y), (x_off + bar_w, y + bar_h), (50, 50, 50), -1)
        # Fill
        fill = int(conf * bar_w)
        color = (0, 200, 0) if rank == 0 else (0, 120, 200)
        cv2.rectangle(frame, (x_off, y), (x_off + fill, y + bar_h), color, -1)
        # Text
        cv2.putText(frame, f"{label}: {conf*100:.1f}%",
                    (x_off + 4, y + bar_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ── Load labels ────────────────────────────────────────────────────────
    labels = None
    if Path(args.labels).is_file():
        with open(args.labels) as f:
            meta = json.load(f)
        # Support both {"actions": [...]} and plain list
        labels = meta.get("actions", meta) if isinstance(meta, dict) else meta
        print(f"[labels] {len(labels)} classes: {labels}")
    else:
        print(f"[warn]  Labels file not found: {args.labels}")

    # ── Build models ───────────────────────────────────────────────────────
    print(f"[model]  Loading {args.model} ...")
    interpreter = build_interpreter(args.model, args.num_threads)
    print(f"[model]  TFLite interpreter ready  (threads={args.num_threads})")

    print(f"[mp]    Building hand landmarker from {args.task} ...")
    landmarker = build_landmarker(args.task)
    print(f"[mp]    Hand landmarker ready")

    # ── State ──────────────────────────────────────────────────────────────
    sequence     = collections.deque(maxlen=SEQUENCE_LENGTH)
    predictions  = collections.deque(maxlen=args.smoothing_window)
    sentence     = []
    last_probs   = np.zeros(len(labels) if labels else 10, dtype=np.float32)
    frame_count  = 0
    fps_t0       = time.perf_counter()
    fps_display  = 0.0

    # ── Camera thread ──────────────────────────────────────────────────────
    camera = CameraReader(args.source, args.width, args.height)
    camera.start()
    print(f"[cam]   Capture started  ({args.width}×{args.height}  source={args.source})")
    print("        Press 'q' to quit, 'c' to clear sentence.\n")

    try:
        while True:
            frame = camera.read()
            if frame is None:
                continue

            frame_count += 1

            # ── FPS tracking ───────────────────────────────────────────────
            if frame_count % 30 == 0:
                fps_display = 30 / (time.perf_counter() - fps_t0)
                fps_t0 = time.perf_counter()

            # ── MediaPipe landmark extraction (every Nth frame) ────────────
            if frame_count % args.skip_frames == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                import mediapipe as mp
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect(mp_image)
                keypoints = extract_keypoints(result)
                sequence.append(keypoints)

            # ── Classifier ────────────────────────────────────────────────
            if len(sequence) == SEQUENCE_LENGTH:
                seq_arr = np.array(sequence, dtype=np.float32)
                probs = run_classifier(interpreter, seq_arr)
                last_probs = probs
                pred_class = int(np.argmax(probs))
                max_conf   = float(probs[pred_class])

                predictions.append(pred_class)

                # Accept prediction only when the window is unanimous and confident
                if (len(set(predictions)) == 1 and
                        max_conf >= args.confidence_threshold and
                        len(predictions) == args.smoothing_window):
                    pred_label = labels[pred_class] if labels else str(pred_class)
                    if not sentence or sentence[-1] != pred_label:
                        sentence.append(pred_label)
                        print(f"  → {pred_label}  ({max_conf*100:.1f}%)")
                    sentence = sentence[-5:]  # keep last 5 words

            # ── Display ───────────────────────────────────────────────────
            if not args.no_display:
                overlay = frame.copy()

                # Sentence bar
                sentence_text = " ".join(sentence)
                cv2.rectangle(overlay, (0, 0), (frame.shape[1], 40), (50, 50, 50), -1)
                cv2.putText(overlay, sentence_text, (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

                # Sequence fill indicator
                fill_frac = len(sequence) / SEQUENCE_LENGTH
                cv2.rectangle(overlay, (0, frame.shape[0]-6), (int(fill_frac * frame.shape[1]), frame.shape[0]),
                              (0, 200, 0) if fill_frac == 1.0 else (0, 150, 200), -1)

                # FPS
                cv2.putText(overlay, f"FPS:{fps_display:.1f}", (10, frame.shape[0] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 0), 1, cv2.LINE_AA)

                draw_prob_bars(overlay, last_probs, labels)
                cv2.imshow("Sign Language — Pi", overlay)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("c"):
                    sentence.clear()
                    predictions.clear()
                    sequence.clear()
                    print("  [cleared]")

            if args.test and frame_count >= args.test_frames:
                print(f"[test]  Processed {frame_count} frames. Exiting.")
                break

    finally:
        camera.stop()
        if not args.no_display:
            cv2.destroyAllWindows()
        landmarker.close()
        print("Done.")


if __name__ == "__main__":
    main()
