import argparse
import json
import random
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Bidirectional, Conv1D, Dense, Dropout, GlobalAveragePooling1D, Input, LSTM
from tensorflow.keras.models import Model

from run_sign_language import build_image, build_landmarkers, download_model_files, extract_keypoints


SELECTED_SIGNS = [
    {'id': '0171', 'arabic': r'\u064a\u0628\u0646\u064a', 'english': 'build'},
    {'id': '0172', 'arabic': r'\u064a\u0643\u0633\u0631', 'english': 'break'},
    {'id': '0173', 'arabic': r'\u064a\u0645\u0634\u064a', 'english': 'walk'},
    {'id': '0174', 'arabic': r'\u064a\u062d\u0628', 'english': 'love'},
    {'id': '0175', 'arabic': r'\u064a\u0643\u0631\u0647', 'english': 'hate'},
    {'id': '0178', 'arabic': r'\u064a\u0632\u0631\u0639', 'english': 'plant'},
    {'id': '0181', 'arabic': r'\u064a\u0641\u0643\u0631', 'english': 'think'},
    {'id': '0182', 'arabic': r'\u064a\u0633\u0627\u0639\u062f', 'english': 'help'},
    {'id': '0185', 'arabic': r'\u064a\u062e\u062a\u0627\u0631', 'english': 'choose'},
    {'id': '0186', 'arabic': r'\u064a\u0646\u0627\u062f\u064a', 'english': 'call'},
]

SEQUENCE_LENGTH = 30
FEATURE_LENGTH = 21 * 3 * 2
CACHE_VERSION = 2
AUTOTUNE = tf.data.AUTOTUNE


def log(message=''):
    print(message, flush=True)


def configure_tensorflow_devices(force_cpu=False, gpu_index=0, mixed_precision=False, xla=False):
    if force_cpu:
        tf.config.set_visible_devices([], 'GPU')
        log('TensorFlow device: CPU forced by --force-cpu')
        return '/CPU:0'

    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        log('TensorFlow device: no GPU detected, training will use CPU.')
        log('If you expected GPU training, check that your TensorFlow install has GPU support.')
        return '/CPU:0'

    if gpu_index < 0 or gpu_index >= len(gpus):
        raise ValueError(f'--gpu-index must be between 0 and {len(gpus) - 1}, got {gpu_index}')

    selected_gpu = gpus[gpu_index]
    tf.config.set_visible_devices(selected_gpu, 'GPU')
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as error:
            log(f'Could not enable memory growth for {gpu.name}: {error}')
        except ValueError as error:
            log(f'Memory growth not available for {gpu.name}: {error}')

    logical_gpus = tf.config.list_logical_devices('GPU')
    if mixed_precision:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        log('Mixed precision enabled: mixed_float16')
    else:
        tf.keras.mixed_precision.set_global_policy('float32')
        log('Mixed precision disabled: float32')

    tf.config.optimizer.set_jit(bool(xla))
    log(f'XLA JIT enabled: {bool(xla)}')
    log(f'TensorFlow device: GPU enabled ({selected_gpu.name})')
    log(f'Logical GPUs available: {len(logical_gpus)}')
    return '/GPU:0'


def query_nvidia_gpu():
    nvidia_smi = shutil.which('nvidia-smi')
    if not nvidia_smi:
        return None

    command = [
        nvidia_smi,
        '--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu',
        '--format=csv,noheader,nounits',
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=3)
    except (OSError, subprocess.SubprocessError):
        return None

    first_line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ''
    parts = [part.strip() for part in first_line.split(',')]
    if len(parts) != 5:
        return None

    name, utilization, memory_used, memory_total, temperature = parts
    return {
        'name': name,
        'utilization': utilization,
        'memory_used': memory_used,
        'memory_total': memory_total,
        'temperature': temperature,
    }


class ConsoleTrainingMonitor(Callback):
    def __init__(self, batch_interval=10, show_gpu_stats=True):
        super().__init__()
        self.batch_interval = batch_interval
        self.show_gpu_stats = show_gpu_stats
        self.best_val_accuracy = 0.0
        self.epoch_started_at = None

    def on_train_begin(self, logs=None):
        log('\nTraining monitor')
        log('Epoch | loss   | acc    | val_loss | val_acc | best_val_acc | lr       | time')
        log('------|--------|--------|----------|---------|--------------|----------|------')

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_started_at = time.time()
        log(f'Starting epoch {epoch + 1}/{self.params.get("epochs", "?")}')

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        steps = self.params.get('steps')
        is_interval = (batch + 1) % self.batch_interval == 0
        is_last_batch = steps is not None and (batch + 1) >= steps
        if not is_interval and not is_last_batch:
            return

        if steps:
            progress = f'{batch + 1}/{steps}'
        else:
            progress = str(batch + 1)
        log(
            f'  batch {progress} - '
            f'loss: {logs.get("loss", 0.0):.4f} - '
            f'acc: {logs.get("sparse_categorical_accuracy", 0.0):.4f}'
        )

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_accuracy = logs.get('val_sparse_categorical_accuracy')
        if val_accuracy is not None:
            self.best_val_accuracy = max(self.best_val_accuracy, float(val_accuracy))

        elapsed = time.time() - self.epoch_started_at if self.epoch_started_at else 0.0
        lr = tf.keras.backend.get_value(self.model.optimizer.learning_rate)
        log(
            f'{epoch + 1:>5} | '
            f'{logs.get("loss", 0.0):.4f} | '
            f'{logs.get("sparse_categorical_accuracy", 0.0):.4f} | '
            f'{logs.get("val_loss", 0.0):.4f}   | '
            f'{logs.get("val_sparse_categorical_accuracy", 0.0):.4f}  | '
            f'{self.best_val_accuracy:.4f}       | '
            f'{float(lr):.6f} | '
            f'{elapsed:.1f}s'
        )
        if self.show_gpu_stats:
            gpu = query_nvidia_gpu()
            if gpu:
                log(
                    f'  GPU {gpu["name"]}: {gpu["utilization"]}% util, '
                    f'{gpu["memory_used"]}/{gpu["memory_total"]} MB, {gpu["temperature"]}C'
                )


def read_sampled_frames(video_path, sequence_length, crop_index=0, crop_count=1):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {video_path}')

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        frame_indexes = list(range(sequence_length))
    else:
        margin = max(0, int(frame_count * 0.08))
        usable_start = margin
        usable_end = max(usable_start, frame_count - 1 - margin)
        if crop_count > 1:
            offset = int(round((crop_index / (crop_count - 1) - 0.5) * frame_count * 0.14))
            usable_start = min(max(0, usable_start + offset), frame_count - 1)
            usable_end = min(max(usable_start, usable_end + offset), frame_count - 1)
        frame_indexes = np.linspace(usable_start, usable_end, sequence_length).astype(int).tolist()

    frames = []
    for frame_index in frame_indexes:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ret, frame = cap.read()
        if not ret:
            frame = frames[-1] if frames else np.zeros((480, 640, 3), dtype=np.uint8)
        frames.append(frame)

    cap.release()
    return frames


def extract_video_sequence(video_path, hand_landmarker, sequence_index, crop_index=0, crop_count=1):
    sequence = []
    timestamp_base = sequence_index * SEQUENCE_LENGTH * 33
    frames = read_sampled_frames(video_path, SEQUENCE_LENGTH, crop_index, crop_count)
    for index, frame in enumerate(frames):
        mp_image = build_image(frame)
        hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_base + index * 33)
        keypoints = extract_keypoints(hand_result)
        if keypoints.shape[0] != FEATURE_LENGTH:
            raise ValueError(f'Expected {FEATURE_LENGTH} keypoints, got {keypoints.shape[0]} for {video_path}')
        sequence.append(keypoints)
    return np.asarray(sequence, dtype=np.float32)


def load_cached_dataset(cache_path, crops_per_video):
    if not cache_path.exists():
        return None

    log(f'Loading landmark cache from {cache_path}...')
    cached = np.load(cache_path, allow_pickle=True)
    version = int(cached['version']) if 'version' in cached else 1
    cached_crops = int(cached['crops_per_video']) if 'crops_per_video' in cached else 1
    if version == CACHE_VERSION and cached_crops == crops_per_video:
        log(f'Loaded cached dataset: {cached["X"].shape[0]} samples')
        return cached['X'], cached['y']

    X = cached['X'] if 'X' in cached else None
    y = cached['y'] if 'y' in cached else None
    if X is not None and y is not None and X.shape[1:] == (SEQUENCE_LENGTH, FEATURE_LENGTH) and cached_crops == crops_per_video:
        log(f'Loaded compatible legacy cache: {X.shape[0]} samples')
        return X, y

    log('Ignoring old landmark cache because preprocessing settings changed.')
    return None


def build_dataset(data_dir, cache_path, model_dir, crops_per_video):
    data_dir = Path(data_dir)
    cache_path = Path(cache_path)
    log('Preparing dataset...')
    cached = load_cached_dataset(cache_path, crops_per_video)
    if cached is not None:
        return cached

    log('No usable cache found. Extracting hand landmarks from videos...')
    download_model_files(Path(model_dir))
    hand_landmarker = build_landmarkers(Path(model_dir))
    sequences = []
    labels = []

    try:
        for label_index, sign in enumerate(SELECTED_SIGNS):
            sign_dir = data_dir / sign['id']
            video_paths = sorted(sign_dir.glob('*.mp4'))
            if not video_paths:
                raise FileNotFoundError(f'No .mp4 files found in {sign_dir}')

            log(f"Extracting {sign['id']} {sign['english']} ({len(video_paths)} videos)")
            total_sequences = len(video_paths) * crops_per_video
            completed_sequences = 0
            for video_number, video_path in enumerate(video_paths, start=1):
                log(f'  video {video_number}/{len(video_paths)}: {video_path.name}')
                for crop_index in range(crops_per_video):
                    sequence = extract_video_sequence(
                        video_path,
                        hand_landmarker,
                        len(sequences),
                        crop_index,
                        crops_per_video,
                    )
                    sequences.append(sequence)
                    labels.append(label_index)
                    completed_sequences += 1
                    log(f'    crop {crop_index + 1}/{crops_per_video} ({completed_sequences}/{total_sequences})')
    finally:
        hand_landmarker.close()

    X = np.asarray(sequences, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    log(f'Saving landmark cache to {cache_path}...')
    np.savez_compressed(cache_path, X=X, y=y, version=CACHE_VERSION, crops_per_video=crops_per_video)
    return X, y


def train_validation_split(X, y, validation_fraction=0.2, seed=42):
    rng = random.Random(seed)
    train_indexes = []
    validation_indexes = []

    for label in sorted(set(y.tolist())):
        indexes = [index for index, value in enumerate(y.tolist()) if value == label]
        rng.shuffle(indexes)
        validation_count = max(1, int(round(len(indexes) * validation_fraction)))
        validation_indexes.extend(indexes[:validation_count])
        train_indexes.extend(indexes[validation_count:])

    rng.shuffle(train_indexes)
    rng.shuffle(validation_indexes)
    return X[train_indexes], X[validation_indexes], y[train_indexes], y[validation_indexes]


def augment_sequences(X, y, copies=2, seed=42):
    if copies <= 0:
        return X, y

    rng = np.random.default_rng(seed)
    augmented = [X]
    augmented_labels = [y]

    for _ in range(copies):
        noisy = X.copy()
        noisy += rng.normal(0.0, 0.015, size=noisy.shape).astype(np.float32)
        noisy *= rng.normal(1.0, 0.04, size=(len(noisy), 1, 1)).astype(np.float32)
        temporal_shift = rng.integers(-2, 3, size=len(noisy))
        for index, shift in enumerate(temporal_shift):
            if shift != 0:
                noisy[index] = np.roll(noisy[index], int(shift), axis=0)
        augmented.append(noisy)
        augmented_labels.append(y.copy())

    return np.concatenate(augmented, axis=0), np.concatenate(augmented_labels, axis=0)


def build_training_dataset(X, y, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.shuffle(buffer_size=len(X), reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.cache()
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


def build_validation_dataset(X, y, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.cache()
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


def build_model(class_count, model_size='large', architecture='conv'):
    model_sizes = {
        'small': {
            'conv_filters': 96,
            'lstm_1': 96,
            'lstm_2': 64,
            'dense_1': 96,
            'dense_2': 48,
        },
        'large': {
            'conv_filters': 160,
            'lstm_1': 160,
            'lstm_2': 128,
            'dense_1': 160,
            'dense_2': 80,
        },
    }
    if model_size not in model_sizes:
        raise ValueError(f'Unknown model size: {model_size}')
    if architecture not in {'conv', 'lstm'}:
        raise ValueError(f'Unknown architecture: {architecture}')
    size = model_sizes[model_size]

    inputs = Input(shape=(SEQUENCE_LENGTH, FEATURE_LENGTH))

    if architecture == 'conv':
        x = Conv1D(size['conv_filters'], kernel_size=3, padding='same', activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.15)(x)
        x = Conv1D(size['conv_filters'], kernel_size=5, padding='same', dilation_rate=2, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Conv1D(size['conv_filters'] * 2, kernel_size=3, padding='same', dilation_rate=4, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.25)(x)
        x = GlobalAveragePooling1D()(x)
    else:
        x = Conv1D(size['conv_filters'], kernel_size=3, padding='same', activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.15)(x)
        x = Bidirectional(LSTM(size['lstm_1'], return_sequences=True))(x)
        x = Dropout(0.25)(x)
        x = Bidirectional(LSTM(size['lstm_2']))(x)

    x = Dense(size['dense_1'], activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(size['dense_2'], activation='relu')(x)
    outputs = Dense(class_count, activation='softmax', dtype='float32')(x)

    model = Model(inputs=inputs, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
    return model


def write_labels(labels_path):
    labels_path = Path(labels_path)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'actions': [sign['english'] for sign in SELECTED_SIGNS],
        'signs': SELECTED_SIGNS,
        'sequence_length': SEQUENCE_LENGTH,
        'feature_length': FEATURE_LENGTH,
        'cache_version': CACHE_VERSION,
        'preprocessing': 'wrist-centered hand landmarks, scaled by hand radius',
    }
    labels_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding='utf-8')


def parse_args():
    parser = argparse.ArgumentParser(description='Train a 10-word KARSL sign-language model from local MP4 videos.')
    parser.add_argument('--data-dir', default=r'..\train', help='Folder containing 0171, 0172, ... video folders.')
    parser.add_argument('--model-dir', default='mediapipe_models', help='MediaPipe model bundle directory.')
    parser.add_argument('--cache', default=r'data\karsl_words_sequences.npz', help='Landmark cache path.')
    parser.add_argument('--model-out', default='karsl_words_action_v2.h5', help='Output .h5 model path.')
    parser.add_argument('--labels-out', default='karsl_words_labels.json', help='Output labels JSON path.')
    parser.add_argument('--epochs', type=int, default=120, help='Maximum training epochs.')
    parser.add_argument('--batch-size', type=int, default=64, help='Training batch size.')
    parser.add_argument('--batch-log-interval', type=int, default=10, help='Print training progress every N batches.')
    parser.add_argument('--crops-per-video', type=int, default=1, help='Temporal crops extracted from each video.')
    parser.add_argument('--augment-copies', type=int, default=2, help='Extra noisy training copies per sequence.')
    parser.add_argument('--model-size', choices=['small', 'large'], default='large', help='Model capacity.')
    parser.add_argument('--gpu-index', type=int, default=0, help='GPU index to use when multiple GPUs are available.')
    parser.add_argument('--mixed-precision', action='store_true', help='Enable mixed_float16 GPU training.')
    parser.add_argument('--xla', action='store_true', help='Enable TensorFlow XLA JIT compilation.')
    parser.add_argument('--no-gpu-stats', action='store_true', help='Do not print nvidia-smi GPU stats after epochs.')
    parser.add_argument('--force-cpu', action='store_true', help='Disable GPU and train on CPU.')
    return parser.parse_args()


def main():
    args = parse_args()
    log('Starting KARSL word training...')
    training_device = configure_tensorflow_devices(
        force_cpu=args.force_cpu,
        gpu_index=args.gpu_index,
        mixed_precision=args.mixed_precision,
        xla=args.xla,
    )
    write_labels(args.labels_out)
    log(f'Wrote labels metadata to {args.labels_out}')

    X, y = build_dataset(args.data_dir, args.cache, args.model_dir, args.crops_per_video)
    X_train, X_validation, y_train, y_validation = train_validation_split(X, y)
    log('Split dataset into training and validation sets.')
    X_train, y_train = augment_sequences(X_train, y_train, args.augment_copies)
    log(f'Applied augmentation: {args.augment_copies} extra copies per training sequence.')

    log('\nDataset ready')
    log(f'Classes: {len(SELECTED_SIGNS)}')
    log(f'Training samples: {len(X_train)}')
    log(f'Validation samples: {len(X_validation)}')
    log(f'Sequence shape: {X_train.shape[1:]}')
    log(f'Batch size: {args.batch_size}')
    log(f'Model size: {args.model_size}')

    log(f'Building model on {training_device}...')
    with tf.device(training_device):
        train_dataset = build_training_dataset(X_train, y_train, args.batch_size)
        validation_dataset = build_validation_dataset(X_validation, y_validation, args.batch_size)
        model = build_model(len(SELECTED_SIGNS), args.model_size)
        model.summary(print_fn=log)
        callbacks = [
            ConsoleTrainingMonitor(
                batch_interval=max(1, args.batch_log_interval),
                show_gpu_stats=not args.no_gpu_stats,
            ),
            EarlyStopping(monitor='val_sparse_categorical_accuracy', patience=18, mode='max', restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-5),
            ModelCheckpoint(
                args.model_out,
                monitor='val_sparse_categorical_accuracy',
                mode='max',
                save_best_only=True,
                verbose=1,
            ),
        ]

        log('Starting model.fit. This script trains once; it is not programmed to restart automatically.')
        model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=args.epochs,
            callbacks=callbacks,
            verbose=0,
        )

        log('Training finished. Loading the best saved model for final evaluation...')
        best_model = tf.keras.models.load_model(args.model_out)
        loss, sparse_accuracy = best_model.evaluate(validation_dataset, verbose=0)

    log(f'Validation loss: {loss:.4f}')
    log(f'Validation sparse accuracy: {sparse_accuracy:.4f}')
    log(f'Best model saved to {args.model_out}')
    log(f'Saved labels to {args.labels_out}')
    log('Training script complete. Exiting now.')


if __name__ == '__main__':
    main()
