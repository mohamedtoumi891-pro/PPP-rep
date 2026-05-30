"""
optimize_for_pi.py — Convert and quantize the KARSL classifier for Raspberry Pi 4.

Produces TFLite models in three quantization tiers, benchmarks latency on the
current machine, validates accuracy against the cached dataset, and prints a
recommendation table. Run this on your development machine, then copy the chosen
.tflite file to the Pi.

Usage (from the project root, with .venv active):
    python optimize_for_pi.py
    python optimize_for_pi.py --model karsl_words_action_v2.h5 \
                               --cache data/karsl_words_sequences_v2.npz \
                               --labels karsl_words_labels.json \
                               --out-dir pi_models

Output files in --out-dir:
    classifier_fp16.tflite   — Float-16 quant  (~1.7 MB)  ← recommended for Pi 4
    classifier_int8.tflite   — Full INT8 quant  (~0.9 MB)  ← smallest / fastest
    classifier_fp32.tflite   — Baseline (no quant)         ← reference only

Pi deployment:
    pip install tflite-runtime mediapipe opencv-python-headless numpy
    python run_sign_language_pi.py --model classifier_fp16.tflite
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Optimize KARSL model for Raspberry Pi 4")
    p.add_argument("--model",   default="karsl_words_action_v2.h5",
                   help="Keras .h5 model to convert")
    p.add_argument("--cache",   default=None,
                   help="NPZ cache file (karsl_words_sequences_v2.npz) for accuracy validation. "
                        "If omitted, accuracy check is skipped.")
    p.add_argument("--labels",  default="karsl_words_labels.json",
                   help="Labels JSON produced by the training script")
    p.add_argument("--out-dir", default="pi_models",
                   help="Directory to write .tflite files")
    p.add_argument("--num-threads", type=int, default=4,
                   help="TFLite interpreter threads for the benchmark (set to 4 for Pi 4)")
    p.add_argument("--benchmark-runs", type=int, default=200,
                   help="Number of single-sample inference calls per model for latency measurement")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_keras_model(model_path: str):
    import tensorflow as tf
    print(f"[load]  Loading {model_path} ...")
    model = tf.keras.models.load_model(model_path, compile=False)
    print(f"        params={model.count_params():,}  "
          f"input={model.input_shape}  output={model.output_shape}")
    return model


def load_cache(cache_path: str):
    """Return (X, y) arrays from the NPZ landmark cache."""
    data = np.load(cache_path, allow_pickle=True)
    # The training script saves keys 'sequences' / 'labels' (or 'X' / 'y').
    x_key = "sequences" if "sequences" in data else "X"
    y_key  = "labels"    if "labels"    in data else "y"
    X = data[x_key].astype(np.float32)   # (N, 30, 126)
    y = data[y_key].astype(np.int32)     # (N,)
    print(f"[cache] Loaded {len(X)} sequences from {cache_path}  "
          f"shape={X.shape}  classes={np.unique(y).tolist()}")
    return X, y


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def _enable_select_tf_ops(converter, tf):
    """Required for BiLSTM/LSTM models: TensorListReserve cannot be lowered to
    TFLite built-ins alone; Select TF ops must be included so the runtime can
    fall back to full TF kernels for those ops."""
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    converter._experimental_lower_tensor_list_ops = False


def convert_fp32(model, out_path: str) -> int:
    """Baseline TFLite conversion — float32, no quantization."""
    import tensorflow as tf
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    _enable_select_tf_ops(converter, tf)
    tflite_model = converter.convert()
    Path(out_path).write_bytes(tflite_model)
    size = len(tflite_model)
    print(f"[fp32]  Saved {out_path}  ({size/1024:.1f} KB)")
    return size


def convert_fp16(model, out_path: str) -> int:
    """Float-16 post-training quantization.

    Weights are stored as float16; XNNPACK delegate on the Pi's ARM Cortex-A72
    executes them via NEON SIMD instructions.  Typically 2× smaller than fp32
    with no measurable accuracy loss on this architecture.
    """
    import tensorflow as tf
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    _enable_select_tf_ops(converter, tf)
    tflite_model = converter.convert()
    Path(out_path).write_bytes(tflite_model)
    size = len(tflite_model)
    print(f"[fp16]  Saved {out_path}  ({size/1024:.1f} KB)")
    return size


def convert_dynrange(model, out_path: str) -> int:
    """Dynamic-range post-training quantization.

    Weights are quantized to INT8 at rest; activations are computed in float32
    at runtime. No representative calibration dataset is required, which avoids
    the TFLite calibrator's inability to run Flex/Select ops (TensorListReserve)
    during INT8 full-integer calibration. On ARM Cortex-A72 this still gives
    roughly 2-3x memory saving over fp32 and measurable throughput improvement
    because weight-dequant is fused into the GEMM kernel.
    """
    import tensorflow as tf
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    _enable_select_tf_ops(converter, tf)
    tflite_model = converter.convert()
    Path(out_path).write_bytes(tflite_model)
    size = len(tflite_model)
    print(f"[dyn8]  Saved {out_path}  ({size/1024:.1f} KB)")
    return size


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def _make_interpreter(tflite_path: str, num_threads: int):
    """Build a TFLite interpreter, registering the Flex delegate when available."""
    try:
        import tflite_runtime.interpreter as tflite
        Interpreter = tflite.Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter

    # Try loading with the Flex delegate (needed for Select TF ops / LSTM).
    # On Windows the delegate DLL may not be discoverable; fall through silently.
    try:
        import tensorflow as tf
        flex_delegate = tf.lite.experimental.load_delegate("flexdelegate.so")
        interp = Interpreter(model_path=tflite_path, num_threads=num_threads,
                             experimental_delegates=[flex_delegate])
    except Exception:
        interp = Interpreter(model_path=tflite_path, num_threads=num_threads)

    interp.allocate_tensors()
    return interp


def benchmark_tflite(tflite_path: str, num_threads: int, runs: int,
                     seq_len=30, feat_len=126):
    """Return mean inference latency in ms, or None if the model cannot run locally."""
    try:
        interpreter = _make_interpreter(tflite_path, num_threads)
    except RuntimeError as e:
        print(f"    [skip] {Path(tflite_path).name}: {e}")
        return None

    inp = interpreter.get_input_details()[0]
    dummy = np.random.rand(1, seq_len, feat_len).astype(np.float32)

    for _ in range(5):
        interpreter.set_tensor(inp["index"], dummy)
        interpreter.invoke()

    t0 = time.perf_counter()
    for _ in range(runs):
        interpreter.set_tensor(inp["index"], dummy)
        interpreter.invoke()
    return (time.perf_counter() - t0) / runs * 1000


# ---------------------------------------------------------------------------
# Accuracy validation
# ---------------------------------------------------------------------------

def validate_accuracy(tflite_path: str, X: np.ndarray, y: np.ndarray, num_threads: int):
    """Run the TFLite model on the full dataset; return top-1 accuracy or None on failure."""
    try:
        interpreter = _make_interpreter(tflite_path, num_threads)
    except RuntimeError as e:
        print(f"    [skip] {Path(tflite_path).name}: {e}")
        return None

    inp_idx = interpreter.get_input_details()[0]["index"]
    out_idx = interpreter.get_output_details()[0]["index"]

    correct = 0
    for i in range(len(X)):
        interpreter.set_tensor(inp_idx, X[i : i + 1])
        interpreter.invoke()
        pred = np.argmax(interpreter.get_tensor(out_idx)[0])
        if pred == y[i]:
            correct += 1
    return correct / len(X) * 100


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def fmt_acc(v):
    return f"{v:.2f} %" if v is not None else "N/A"


def fmt_lat(v):
    return f"{v:>13.2f}" if v is not None else "          N/A"


def main():
    args = parse_args()

    # --- Validate inputs ---------------------------------------------------
    if not os.path.isfile(args.model):
        sys.exit(f"ERROR: Model file not found: {args.model}")

    labels = None
    if os.path.isfile(args.labels):
        with open(args.labels) as f:
            labels = json.load(f)
        print(f"[labels] {len(labels.get('actions', labels))} classes loaded from {args.labels}")

    cache_X, cache_y = None, None
    if args.cache:
        if not os.path.isfile(args.cache):
            print(f"[warn]  Cache file not found: {args.cache}  (INT8 calibration skipped)")
        else:
            cache_X, cache_y = load_cache(args.cache)

    # ── Output directory ───────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fp32_path  = str(out_dir / "classifier_fp32.tflite")
    fp16_path  = str(out_dir / "classifier_fp16.tflite")
    dyn8_path  = str(out_dir / "classifier_dyn8.tflite")

    # --- Load Keras model --------------------------------------------------
    model = load_keras_model(args.model)

    # --- Convert -----------------------------------------------------------
    print("\n--- Conversion ---------------------------------------------------")
    size_fp32 = convert_fp32(model, fp32_path)
    size_fp16 = convert_fp16(model, fp16_path)
    size_dyn8 = convert_dynrange(model, dyn8_path)

    # --- Benchmark ---------------------------------------------------------
    print(f"\n--- Latency benchmark  (threads={args.num_threads}, runs={args.benchmark_runs}) ---")
    print("    NOTE: These timings are measured on your dev machine (x86/x64).")
    print("    ARM Cortex-A72 timings will differ; dyn8 is fastest on ARM.\n")

    lat_fp32 = benchmark_tflite(fp32_path, args.num_threads, args.benchmark_runs)
    lat_fp16 = benchmark_tflite(fp16_path, args.num_threads, args.benchmark_runs)
    lat_dyn8 = benchmark_tflite(dyn8_path, args.num_threads, args.benchmark_runs)

    print(f"    fp32: {fmt_lat(lat_fp32).strip() or 'N/A (Flex delegate not available on this host)'}")
    print(f"    fp16: {fmt_lat(lat_fp16).strip() or 'N/A (Flex delegate not available on this host)'}")
    print(f"    dyn8: {fmt_lat(lat_dyn8).strip() or 'N/A (Flex delegate not available on this host)'}")

    # --- Accuracy validation -----------------------------------------------
    acc_fp32 = acc_fp16 = acc_dyn8 = None
    if cache_X is not None:
        print(f"\n--- Accuracy on full cache ({len(cache_X)} samples) ---")
        acc_fp32 = validate_accuracy(fp32_path, cache_X, cache_y, args.num_threads)
        acc_fp16 = validate_accuracy(fp16_path, cache_X, cache_y, args.num_threads)
        acc_dyn8 = validate_accuracy(dyn8_path, cache_X, cache_y, args.num_threads)
        print(f"    fp32: {fmt_acc(acc_fp32)}")
        print(f"    fp16: {fmt_acc(acc_fp16)}")
        print(f"    dyn8: {fmt_acc(acc_dyn8)}")

    # --- Summary table -----------------------------------------------------
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  {'Format':<10} {'Size (KB)':>10} {'Latency (ms)':>14} {'Accuracy':>10}  Notes")
    print(f"  {'-'*10} {'-'*10} {'-'*14} {'-'*10}  -----")

    print(f"  {'fp32':<10} {size_fp32/1024:>10.1f} {fmt_lat(lat_fp32)}  {fmt_acc(acc_fp32):>10}  baseline")
    print(f"  {'fp16':<10} {size_fp16/1024:>10.1f} {fmt_lat(lat_fp16)}  {fmt_acc(acc_fp16):>10}  ** RECOMMENDED for Pi 4 **")
    print(f"  {'dyn8':<10} {size_dyn8/1024:>10.1f} {fmt_lat(lat_dyn8)}  {fmt_acc(acc_dyn8):>10}  INT8 weights, float32 activations")
    print("=" * 70)

    print("""
  Pi 4 deployment notes:
  1. Copy pi_models/classifier_fp16.tflite + hand_landmarker.task to the Pi.

  2. IMPORTANT - these models use Select TF ops (Flex delegate) for the LSTM
     TensorList ops. You must install FULL TensorFlow on the Pi, not just
     tflite-runtime, so the Flex delegate is available:
       pip install tensorflow   # ~500 MB, but includes Flex ops

  3. Run:
       python run_sign_language_pi.py --model pi_models/classifier_fp16.tflite

  4. The Pi 4 has 4 ARM Cortex-A72 cores. TFLite XNNPACK delegate uses
     NEON SIMD automatically for dense/conv layers.

  5. Benchmark N/A above = Flex delegate not loaded on this Windows host.
     On the Pi (ARM + full TF), all three models run fine.

  6. For headless SSH use, add --no-display.
  7. Pi Camera Module: --source /dev/video0 (or just 0).
""")

    print(f"  Output files written to: {out_dir.resolve()}\n")


if __name__ == "__main__":
    main()
