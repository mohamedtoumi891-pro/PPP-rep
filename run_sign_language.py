import argparse
import json
import os
import time
import urllib.request
from pathlib import Path

# Suppress TF noise before importing TF
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

import cv2
import numpy as np

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

# TF is only needed for CPU mode — imported lazily to avoid slowing startup in GPU mode
_tf = None

def _get_tf():
    global _tf
    if _tf is None:
        import tensorflow as tf  # noqa: PLC0415
        _tf = tf
    return _tf

from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.image import Image, ImageFormat

MODEL_URLS = {
    'hand_landmarker.task': 'https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task',
}

DEFAULT_MODEL = 'karsl_words_action_v2.h5'
DEFAULT_LABELS = 'karsl_words_labels.json'


def download_model_files(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in MODEL_URLS.items():
        target = model_dir / filename
        if target.exists():
            continue
        print(f'Downloading {filename}...')
        urllib.request.urlretrieve(url, target)
        print(f'Downloaded {filename} to {target}')


def build_landmarkers(model_dir: Path):
    hand_options = mp_vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_dir / 'hand_landmarker.task')),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.HandLandmarker.create_from_options(hand_options)


def build_image(frame: np.ndarray) -> Image:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image(image_format=ImageFormat.SRGB, data=rgb)


def _landmarks_to_array(landmarks, expected_count, include_visibility=False):
    if not landmarks or not hasattr(landmarks, '__len__'):
        return None
    landmarks = landmarks[0] if isinstance(landmarks[0], list) or hasattr(landmarks[0], '__len__') else landmarks
    if not landmarks or len(landmarks) != expected_count:
        return None

    if include_visibility:
        coords = [[
            res.x if res.x is not None else 0.0,
            res.y if res.y is not None else 0.0,
            res.z if res.z is not None else 0.0,
            res.visibility if res.visibility is not None else 0.0,
        ] for res in landmarks]
    else:
        coords = [[
            res.x if res.x is not None else 0.0,
            res.y if res.y is not None else 0.0,
            res.z if res.z is not None else 0.0,
        ] for res in landmarks]
    return np.array(coords, dtype=np.float32).flatten()


def normalize_hand_keypoints(values):
    points = values.reshape(21, 3).astype(np.float32)
    if np.count_nonzero(points) == 0:
        return values

    points = points - points[0]
    scale = float(np.max(np.linalg.norm(points[:, :2], axis=1)))
    if scale < 1e-6:
        return points.flatten()
    return (points / scale).flatten()


def extract_keypoints(hand_result):
    lh = np.zeros(21 * 3, dtype=np.float32)
    rh = np.zeros(21 * 3, dtype=np.float32)
    if hand_result and getattr(hand_result, 'hand_landmarks', None):
        handedness_list = hand_result.handedness or []
        for hand_landmarks, handedness in zip(hand_result.hand_landmarks, handedness_list):
            values = _landmarks_to_array([hand_landmarks], 21)
            if values is None:
                continue
            label = None
            if handedness:
                category = handedness[0]
                label = getattr(category, 'category_name', None) or getattr(category, 'display_name', None)
            if label and label.lower() == 'left':
                lh = values
            elif label and label.lower() == 'right':
                rh = values
            elif np.count_nonzero(lh) == 0:
                lh = values
            elif np.count_nonzero(rh) == 0:
                rh = values

    return np.concatenate([normalize_hand_keypoints(lh), normalize_hand_keypoints(rh)])


def draw_keypoints(image, hand_result):
    height, width, _ = image.shape

    def draw_landmark_list(landmarks, color, radius=2):
        for lm in landmarks:
            if lm.x is None or lm.y is None:
                continue
            x = int(min(max(lm.x * width, 0), width - 1))
            y = int(min(max(lm.y * height, 0), height - 1))
            cv2.circle(image, (x, y), radius, color, -1)

    if hand_result and getattr(hand_result, 'hand_landmarks', None):
        for landmarks in hand_result.hand_landmarks:
            draw_landmark_list(landmarks, (255, 0, 0), radius=3)


def build_colors(count):
    base_colors = [
        (245, 117, 16),
        (117, 245, 16),
        (16, 117, 245),
        (245, 16, 117),
        (16, 245, 207),
        (207, 16, 245),
        (245, 207, 16),
        (117, 16, 245),
        (16, 245, 117),
        (245, 117, 207),
    ]
    return [base_colors[index % len(base_colors)] for index in range(count)]


def load_actions(labels_path):
    if not labels_path:
        raise ValueError('No labels file specified. Use --labels to provide a JSON labels file.')

    with open(labels_path, 'r', encoding='utf-8') as labels_file:
        metadata = json.load(labels_file)

    actions = metadata.get('actions')
    if not actions:
        raise ValueError(f'No actions found in labels file: {labels_path}')
    return actions


def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        y1 = 60 + num * 32
        y2 = 88 + num * 32
        cv2.rectangle(output_frame, (0, y1), (int(prob * 180), y2), colors[num], -1)
        cv2.putText(
            output_frame,
            actions[num],
            (0, y2 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return output_frame


def convert_h5_to_onnx(h5_path: str, onnx_path: str) -> None:
    try:
        import tf2onnx
        import tf2onnx.convert
    except ImportError as e:
        raise ImportError(f'Missing package for ONNX conversion: {e}. Install: pip install tf2onnx') from e

    tf = _get_tf()
    print(f'Converting {h5_path} to ONNX (this runs once)...')
    model = tf.keras.models.load_model(h5_path, compile=False)
    input_sig = [tf.TensorSpec(model.input_shape, tf.float32, name='input')]
    tf2onnx.convert.from_keras(model, input_signature=input_sig, opset=13, output_path=onnx_path)
    print(f'ONNX model saved to {onnx_path}')


def _select_gpu_providers():
    available = set(ort.get_available_providers())
    preferred = ['CUDAExecutionProvider', 'DmlExecutionProvider']
    return [p for p in preferred if p in available] + ['CPUExecutionProvider']


def load_inference_engine(model_path: str, use_gpu: bool):
    """Returns ('onnx', session) or ('tf', keras_model)."""
    if use_gpu:
        if not ONNX_AVAILABLE:
            print('WARNING: onnxruntime not installed. Run: pip install onnxruntime-directml')
            print('Falling back to CPU (TensorFlow).')
        else:
            onnx_path = model_path.replace('.h5', '.onnx')
            if not os.path.exists(onnx_path):
                convert_h5_to_onnx(model_path, onnx_path)

            providers = _select_gpu_providers()
            session = ort.InferenceSession(onnx_path, providers=providers)
            active = session.get_providers()
            if 'CUDAExecutionProvider' in active:
                print('GPU inference: RTX 4060 via CUDA')
            elif 'DmlExecutionProvider' in active:
                print('GPU inference: RTX 4060 via DirectML')
            else:
                print('GPU providers unavailable — running on CPU via ONNX Runtime')
            return ('onnx', session)

    tf = _get_tf()
    model = tf.keras.models.load_model(model_path, compile=False)
    print('Loaded TensorFlow model (CPU)')
    return ('tf', model)


def get_output_size(engine):
    mode, obj = engine
    if mode == 'onnx':
        return obj.get_outputs()[0].shape[-1]
    return obj.output_shape[-1]


def run_inference(engine, sequence: np.ndarray) -> np.ndarray:
    mode, obj = engine
    data = np.expand_dims(sequence, axis=0).astype(np.float32)
    if mode == 'onnx':
        return obj.run(None, {obj.get_inputs()[0].name: data})[0][0]
    return obj.predict(data, verbose=0)[0]


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run sign language detection using MediaPipe Tasks and a Keras model.')
    parser.add_argument('--source', type=str, default='0', help='Video source: camera index or path to video file.')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL, help='Path to the trained Keras .h5 model.')
    parser.add_argument('--labels', type=str, default=DEFAULT_LABELS, help='Path to a JSON labels file.')
    parser.add_argument('--model-dir', type=str, default='mediapipe_models', help='Directory for MediaPipe task bundles.')
    parser.add_argument('--gpu', action='store_true', help='Use RTX 4060 GPU via ONNX Runtime + CUDA.')
    parser.add_argument('--test', action='store_true', help='Run a quick test for a few frames without user input.')
    parser.add_argument('--test-frames', type=int, default=30, help='Number of frames to process in test mode.')
    parser.add_argument('--no-display', action='store_true', help='Do not show the OpenCV window.')
    return parser.parse_args()


def open_video_source(source):
    try:
        source_id = int(source)
    except ValueError:
        source_id = source
    cap = cv2.VideoCapture(source_id)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video source: {source}')
    return cap


def main():
    args = parse_arguments()

    model_path = args.model
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f'Model not found: {model_path}\n'
            f'Default model is {DEFAULT_MODEL}. Make sure it exists in the project directory.'
        )

    print(f'Loading model: {model_path}')
    engine = load_inference_engine(model_path, args.gpu)

    actions = load_actions(args.labels)
    out_size = get_output_size(engine)
    if out_size != len(actions):
        raise ValueError(
            f'Model outputs {out_size} classes but {len(actions)} labels loaded from {args.labels}. '
            'Use the labels JSON that was created with this model.'
        )

    colors = build_colors(len(actions))

    model_dir = Path(args.model_dir)
    download_model_files(model_dir)
    hand_landmarker = build_landmarkers(model_dir)

    sequence = []
    sentence = []
    predictions = []
    threshold = 0.5

    print(f'Opening camera (source={args.source})...')
    cap = open_video_source(args.source)
    print('Camera opened. Press Q to quit.')
    frame_count = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            mp_image = build_image(frame)
            timestamp_ms = int(time.time() * 1000)

            hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

            draw_keypoints(frame, hand_result)

            keypoints = extract_keypoints(hand_result)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = run_inference(engine, np.array(sequence))
                predictions.append(np.argmax(res))

                if np.unique(predictions[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > threshold:
                        if len(sentence) == 0 or actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])

                    if len(sentence) > 5:
                        sentence = sentence[-5:]

                frame = prob_viz(res, actions, frame, colors)

            cv2.rectangle(frame, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(
                frame,
                ' '.join(sentence),
                (3, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if not args.no_display:
                cv2.imshow('Sign Language Detection', frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            frame_count += 1
            if args.test and frame_count >= args.test_frames:
                print(f'Test mode completed after {frame_count} frames.')
                break

    finally:
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        hand_landmarker.close()


if __name__ == '__main__':
    main()
