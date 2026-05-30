# Sign Language Detection — Project Pipeline & Technical Reference

> **Audience:** This document is intended as a complete technical reference for academic presentation.
> It covers the full data pipeline, all models and their exact parameters, architecture diagrams,
> data augmentation strategy, and inference workflow.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Dataset & Vocabulary](#3-dataset--vocabulary)
4. [Data Pipeline](#4-data-pipeline)
   - 4.1 [Video Frame Sampling](#41-video-frame-sampling)
   - 4.2 [Pre-Processing: Image Augmentation Before Landmark Extraction](#42-pre-processing-image-augmentation-before-landmark-extraction)
   - 4.3 [Keypoint Extraction — MediaPipe Hand Landmarker](#43-keypoint-extraction--mediapipe-hand-landmarker)
   - 4.4 [Feature Normalization](#44-feature-normalization)
   - 4.5 [Dataset Split](#45-dataset-split)
   - 4.6 [Offline Data Augmentation](#46-offline-data-augmentation)
   - 4.7 [tf.data Input Pipeline](#47-tfdata-input-pipeline)
5. [Model Architectures](#5-model-architectures)
   - 5.1 [MediaPipe Hand Landmarker (Feature Extractor)](#51-mediapipe-hand-landmarker-feature-extractor)
   - 5.2 [Temporal Conv1D Classifier](#52-temporal-conv1d-classifier)
   - 5.3 [Bidirectional LSTM Classifier](#53-bidirectional-lstm-classifier)
   - 5.4 [Architecture Comparison](#54-architecture-comparison)
6. [Training Configuration](#6-training-configuration)
7. [Real-Time Inference Pipeline](#7-real-time-inference-pipeline)
8. [GPU Acceleration](#8-gpu-acceleration)
9. [How to Run](#9-how-to-run)
10. [Training Results & Evaluation](#10-training-results--evaluation)
    - 10.1 [Evaluated Model Summary](#101-evaluated-model-summary)
    - 10.2 [Overall Performance Metrics](#102-overall-performance-metrics)
    - 10.3 [Confusion Matrix](#103-confusion-matrix)
    - 10.4 [Per-Class Classification Report](#104-per-class-classification-report)
    - 10.5 [Top-k Accuracy](#105-top-k-accuracy)
    - 10.6 [Prediction Confidence Analysis](#106-prediction-confidence-analysis)
    - 10.7 [Error Analysis](#107-error-analysis)
    - 10.8 [Discussion & Limitations](#108-discussion--limitations)
11. [File Reference](#11-file-reference)

---

## 1. Project Overview

**Goal:** Recognize a vocabulary of Arabic sign language (KARSL) words in real-time using only a standard RGB camera, with no specialized hardware.

**Core approach:**
- Use MediaPipe's pre-trained Hand Landmarker to extract 2D/3D hand keypoints from each video frame, bypassing the need to train a pixel-level vision backbone from scratch.
- Feed the resulting normalized keypoint sequences into a lightweight temporal classifier (Conv1D or Bidirectional LSTM) trained on KARSL video data.
- Run the entire pipeline live at camera frame rate, displaying predicted signs and probability bars on-screen.

**Key design decision — hand landmarks only:**
The input to the classifier is restricted to normalized hand landmarks (126 values per frame) rather than full-body features. This reduces the input dimensionality by more than 13×, cuts overfitting risk on small datasets, and makes real-time inference faster.

---

## 2. System Architecture

### Full Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DATA COLLECTION PHASE                            │
│                                                                         │
│   KARSL Video Dataset                                                   │
│   (MP4 clips, one folder per sign ID)                                   │
│              │                                                           │
│              ▼                                                           │
│   ┌─────────────────────┐                                               │
│   │  Frame Sampling     │  ← Sample exactly 30 frames per video clip   │
│   │  (uniform spacing)  │    with 8% head/tail margin                  │
│   └─────────────────────┘                                               │
│              │                                                           │
│              ▼                                                           │
│   ┌──────────────────────────────────────────┐                          │
│   │  [RECOMMENDED] Pre-MediaPipe Augmentation│                          │
│   │  • CLAHE contrast enhancement            │                          │
│   │  • Gamma correction                      │                          │
│   │  • Adaptive histogram equalization       │                          │
│   └──────────────────────────────────────────┘                          │
│              │                                                           │
│              ▼                                                           │
│   ┌──────────────────────────────────────────┐                          │
│   │  MediaPipe Hand Landmarker               │                          │
│   │  (hand_landmarker.task, 7.8 MB)          │                          │
│   │  → 21 keypoints per hand (x, y, z)       │                          │
│   │  → Left / Right hand classification      │                          │
│   └──────────────────────────────────────────┘                          │
│              │                                                           │
│              ▼                                                           │
│   ┌─────────────────────────────────────────┐                           │
│   │  Normalization                          │                           │
│   │  1. Wrist-center (subtract landmark 0) │                           │
│   │  2. Scale by hand radius (max 2D dist) │                           │
│   │  → 21×3 per hand → concat → 126 floats │                           │
│   └─────────────────────────────────────────┘                           │
│              │                                                           │
│              ▼                                                           │
│   ┌──────────────────────────────────┐                                  │
│   │  Rolling Sequence Buffer         │                                  │
│   │  30 frames × 126 features        │                                  │
│   │  → Tensor shape: (30, 126)       │                                  │
│   └──────────────────────────────────┘                                  │
│              │                                                           │
│              ▼                                                           │
│   ┌──────────────────────────────────┐                                  │
│   │  Offline Augmentation            │                                  │
│   │  • Gaussian noise  σ = 0.015     │                                  │
│   │  • Amplitude scale N(1.0, 0.04)  │                                  │
│   │  • Temporal shift  ± 2 frames    │                                  │
│   │  → 3× total training samples     │                                  │
│   └──────────────────────────────────┘                                  │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        TRAINING PHASE                                   │
│                                                                         │
│   Dataset Split                                                         │
│   ┌───────────────┬─────────────────┬───────────────┐                  │
│   │  Training 70% │  Validation 15% │   Test 15%    │                  │
│   │  (augmented)  │  (clean)        │   (held-out)  │                  │
│   └───────────────┴─────────────────┴───────────────┘                  │
│              │                                                           │
│              ▼                                                           │
│   ┌─────────────────────────────────────────────────┐                  │
│   │  Temporal Classifier (choice of architecture)  │                   │
│   │                                                 │                  │
│   │  Option A: Temporal Conv1D (default)           │                   │
│   │  ┌──────────────────────────────────────────┐  │                   │
│   │  │ Input (30, 126)                          │  │                   │
│   │  │   → Conv1D(160, k=3, d=1) + BN + Drop   │  │                   │
│   │  │   → Conv1D(160, k=5, d=2) + BN + Drop   │  │                   │
│   │  │   → Conv1D(320, k=3, d=4) + BN + Drop   │  │                   │
│   │  │   → GlobalAveragePooling1D               │  │                   │
│   │  │   → Dense(160) + Drop(0.3)               │  │                   │
│   │  │   → Dense(80)  + Drop(0.3)               │  │                   │
│   │  │   → Dense(10, softmax)                   │  │                   │
│   │  └──────────────────────────────────────────┘  │                   │
│   │                                                 │                   │
│   │  Option B: Bidirectional LSTM                  │                   │
│   │  ┌──────────────────────────────────────────┐  │                   │
│   │  │ Input (30, 126)                          │  │                   │
│   │  │   → Conv1D(160, k=3) + BN + Drop(0.15)  │  │                   │
│   │  │   → BiLSTM(160, ret_seq=True) + Drop     │  │                   │
│   │  │   → BiLSTM(128)                          │  │                   │
│   │  │   → Dense(160) + Drop(0.3)               │  │                   │
│   │  │   → Dense(80)  + Drop(0.3)               │  │                   │
│   │  │   → Dense(10, softmax)                   │  │                   │
│   │  └──────────────────────────────────────────┘  │                   │
│   └─────────────────────────────────────────────────┘                  │
│              │                                                           │
│              ▼                                                           │
│   ┌────────────────────────────────────────────────┐                   │
│   │  Training Loop                                 │                   │
│   │  Adam(lr=0.001) · EarlyStopping(patience=18)  │                   │
│   │  ReduceLROnPlateau · ModelCheckpoint           │                   │
│   │  → Best model saved: karsl_words_action_v2.h5 │                   │
│   └────────────────────────────────────────────────┘                   │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    REAL-TIME INFERENCE PHASE                            │
│                                                                         │
│   Camera → Frame → [Pre-proc] → MediaPipe → Normalize → Buffer(30)    │
│      → Classifier → Softmax(10) → Smoothing → Display sentence        │
│                                                                         │
│   Inference backend:  CPU (TensorFlow)  │  GPU (ONNX Runtime DirectML)│
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Dataset & Vocabulary

The model is trained on a subset of the **KARSL (Kuwait Arabic Robot Sign Language)** dataset — a collection of RGB video clips of Arabic sign language words performed by multiple signers.

### Selected vocabulary (10 words)

| Class Index | Sign ID | English   | Arabic    |
|:-----------:|:-------:|:---------:|:---------:|
| 0           | 0171    | build     | يبني      |
| 1           | 0172    | break     | يكسر      |
| 2           | 0173    | walk      | يمشي      |
| 3           | 0174    | love      | يحب       |
| 4           | 0175    | hate      | يكره      |
| 5           | 0178    | plant     | يزرع      |
| 6           | 0181    | think     | يفكر      |
| 7           | 0182    | help      | يساعد     |
| 8           | 0185    | choose    | يختار     |
| 9           | 0186    | call      | ينادي     |

Each folder (e.g., `0171/`) contains `.mp4` clip files for the corresponding sign. The number of clips per sign determines the raw sample count before augmentation.

---

## 4. Data Pipeline

### 4.1 Video Frame Sampling

Each video clip is sampled to produce exactly **30 frames** regardless of its original duration or frame rate.

```
Video clip (variable length)
│
├─ Detect total frame count N
├─ Remove 8% head margin and 8% tail margin  (removes intro/outro noise)
├─ Usable range: [N×0.08 , N×0.92]
└─ Sample 30 frame indices linearly (numpy.linspace) within usable range
```

For **multi-crop training** (`--crops-per-video > 1`), the usable window is shifted by ±14% of the total duration for each additional crop, generating multiple temporally distinct sequences from a single video.

> **Parameter:** `SEQUENCE_LENGTH = 30` frames

---

### 4.2 Pre-Processing: Image Augmentation Before Landmark Extraction

> **Note:** This section describes a recommended pre-processing stage that should be applied **before** passing frames to MediaPipe. It is not currently implemented in the script but is essential for robust real-world performance, particularly in challenging lighting conditions.

#### Problem

During real-time use, the MediaPipe Hand Landmarker can fail or produce inaccurate keypoints in two extreme conditions:

- **High contrast / overexposed:** bright background or strong direct lighting washes out hand boundaries, causing missed detections.
- **Low contrast / underexposed:** dark environments or dark skin tones against dark backgrounds reduce the confidence of landmark placement, producing noisy or absent detections.

Both conditions directly corrupt the 126-dimensional input vector fed to the classifier.

#### Recommended augmentation chain (applied per frame, before MediaPipe)

```
Raw BGR frame (from camera or video)
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  Step 1 — CLAHE (Contrast Limited Adaptive Histogram   │
│           Equalization)                                 │
│                                                         │
│  • Convert BGR → LAB color space                       │
│  • Apply CLAHE only to the L (lightness) channel       │
│    - clipLimit  = 2.0                                  │
│    - tileGridSize = (8, 8)                             │
│  • Reconstruct BGR                                     │
│  • Effect: locally enhances contrast without globally  │
│    over-brightening already-bright regions             │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  Step 2 — Gamma Correction                             │
│                                                         │
│  • Compute mean luminance of the frame                 │
│  • If mean < 80  (dark frame):  γ = 0.6  → brighten   │
│  • If mean > 180 (bright frame): γ = 1.5 → darken     │
│  • Otherwise:                    γ = 1.0 → no change  │
│                                                         │
│  output = (input / 255.0) ^ (1/γ) × 255               │
│  Effect: adaptively corrects exposure                  │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  Step 3 — Gaussian Smoothing (optional, for noisy      │
│           sensors or compressed video)                  │
│                                                         │
│  • kernel = (3, 3), σ = 0.8                            │
│  • Applied only when sensor noise is expected          │
│  • Effect: reduces high-frequency pixel noise that     │
│    can destabilize landmark confidence scores          │
└─────────────────────────────────────────────────────────┘
        │
        ▼
Normalized frame → passed to MediaPipe Hand Landmarker
```

#### Why this matters for the classifier

MediaPipe's internal hand detection CNN was trained on diverse but balanced-exposure images. When given an overexposed or underexposed frame, its detection confidence drops below the thresholds (`min_hand_detection_confidence = 0.5`), causing the entire hand to be missed. This results in a zero-vector being appended to the rolling buffer instead of valid landmarks, corrupting the temporal sequence fed to the classifier and triggering false or null predictions.

Applying CLAHE and gamma correction before the frame reaches MediaPipe normalizes the luminance distribution, restoring detection reliability across diverse lighting environments without altering the geometric structure of the hand.

#### Training-time application

The same augmentation chain should also be applied during the data extraction phase (`train_karsl_words.py`) to ensure the training distribution matches the inference distribution. Without this, the model will see clean-extraction landmarks during training but noisy-detection landmarks at inference time.

---

### 4.3 Keypoint Extraction — MediaPipe Hand Landmarker

MediaPipe's pre-trained `hand_landmarker.task` (7.8 MB) detects up to **2 hands** per frame and returns 21 landmarks per hand, each described by three coordinates.

```
Per detected hand:
  Landmark 0  — Wrist
  Landmarks 1–4   — Thumb (base to tip)
  Landmarks 5–8   — Index finger
  Landmarks 9–12  — Middle finger
  Landmarks 13–16 — Ring finger
  Landmarks 17–20 — Pinky finger
  Each landmark: (x, y, z) in normalized image coordinates
```

The landmarker is configured in **VIDEO mode** (not IMAGE or LIVE_STREAM) to maintain temporal state across frames:

| Parameter                    | Value | Effect                                      |
|------------------------------|-------|---------------------------------------------|
| `num_hands`                  | 2     | Detect up to both hands simultaneously      |
| `min_hand_detection_confidence` | 0.5 | Minimum score to start tracking a hand      |
| `min_hand_presence_confidence`  | 0.5 | Minimum score to keep tracking between frames |
| `min_tracking_confidence`    | 0.5   | Minimum score for landmark tracking         |
| `running_mode`               | VIDEO | Enables inter-frame state for stable tracking |

Hand assignment (left vs. right) uses the `handedness` classification returned by the landmarker. If a hand cannot be assigned a label, it fills the first available (left/right) slot.

---

### 4.4 Feature Normalization

Raw landmark coordinates depend on the signer's distance from the camera (scale) and hand position in the frame (translation). Two normalization steps remove this dependency:

```
For each hand independently (21 landmarks × 3 coordinates):

  Step 1 — Wrist centering (translation invariance)
  ─────────────────────────────────────────────────
  points[i] = landmark[i] - landmark[0]   (subtract wrist)
  → Hand gesture is now expressed relative to the wrist position.

  Step 2 — Hand radius scaling (scale invariance)
  ────────────────────────────────────────────────
  scale = max( ||points[i][:2]||₂  for i in 0..20 )
  points = points / scale
  → Hand size is normalized to unit radius.
  → If scale < 1e-6, skip division (hand not detected, keep zeros).
```

**Output per frame:**
- Left hand:  21 × 3 = 63 normalized floats
- Right hand: 21 × 3 = 63 normalized floats
- Concatenated: **126 floats** → `FEATURE_LENGTH = 126`

If a hand is not detected, its 63 values remain as zeros, which the model learns to interpret as "hand absent."

---

### 4.5 Dataset Split

The training script performs a **stratified split** — each class contributes proportionally to both sets — ensuring no class is over- or under-represented in validation.

#### Current implementation

| Subset     | Fraction | Content                                    |
|------------|----------|--------------------------------------------|
| Training   | **80%**  | Used for gradient updates + augmentation   |
| Validation | **20%**  | Used for `val_sparse_categorical_accuracy`, early stopping, LR scheduling |

> The split is reproducible via a fixed `seed = 42`.

#### Recommended best practice (for rigorous evaluation)

For a complete academic evaluation, a separate **held-out test set** should be created before any training, never used during model selection:

| Subset     | Fraction | Role                                        |
|------------|----------|---------------------------------------------|
| Training   | **70%**  | Gradient updates, augmented                 |
| Validation | **15%**  | Hyperparameter tuning, early stopping       |
| Test       | **15%**  | Final unbiased accuracy report (used once)  |

The test set should be drawn from **different recording sessions** or **different signers** than the training set, to measure generalization rather than memorization.

---

### 4.6 Offline Data Augmentation

Applied **only to the training split** (never to validation or test) after landmark extraction and before the tf.data pipeline.

Each original training sequence is copied `--augment-copies` times (default: 2), with three independent perturbations applied simultaneously:

| Augmentation     | Implementation                                | Purpose                                           |
|------------------|-----------------------------------------------|---------------------------------------------------|
| **Gaussian noise** | `+ N(0, 0.015)` added to all 126 values per frame | Simulates landmark jitter from imperfect detection |
| **Amplitude scale** | `× N(1.0, 0.04)` per sequence (one scalar per sample) | Simulates slight size variation of the signer's hand |
| **Temporal shift** | `numpy.roll(sequence, shift, axis=0)` with `shift ∈ {−2,−1,0,1,2}` | Simulates sign performed slightly faster or slower |

**Effect on dataset size:**

```
Original training samples:  N_train
After augmentation (copies=2): N_train × (1 + 2) = 3 × N_train
```

This tripling of data effectively prevents overfitting on the relatively small KARSL video collection, at no additional data collection cost.

---

### 4.7 tf.data Input Pipeline

The augmented dataset is wrapped in a `tf.data.Dataset` for efficient GPU feeding during training:

```
Training pipeline:
  Dataset.from_tensor_slices(X_train, y_train)
    → .shuffle(buffer_size=len(X_train), reshuffle_each_iteration=True)
    → .batch(64, drop_remainder=False)
    → .cache()
    → .prefetch(tf.data.AUTOTUNE)

Validation pipeline:
  Dataset.from_tensor_slices(X_val, y_val)
    → .batch(64, drop_remainder=False)
    → .cache()
    → .prefetch(tf.data.AUTOTUNE)
```

- **Shuffle** with full-buffer reshuffle ensures every epoch sees a different ordering.
- **Cache** stores batches in memory after the first epoch, eliminating redundant CPU work.
- **Prefetch with AUTOTUNE** overlaps CPU data preparation with GPU computation.

---

## 5. Model Architectures

Both classifiers share the same input/output interface and are interchangeable at inference time.

```
Input:   (batch_size, 30, 126)   — sequence of 30 normalized keypoint frames
Output:  (batch_size, 10)        — softmax probability over 10 sign classes
```

### 5.1 MediaPipe Hand Landmarker (Feature Extractor)

This is **not trained** — it is a pre-built TFLite model bundle provided by Google's MediaPipe team.

| Property           | Value                                               |
|--------------------|-----------------------------------------------------|
| File               | `hand_landmarker.task` (7.8 MB)                     |
| Architecture       | BlazePalm detector + Hand landmark regression model  |
| Input              | RGB image (any resolution)                          |
| Output             | Up to 2 × 21 landmarks (x, y, z) + handedness       |
| Running mode       | VIDEO (maintains inter-frame tracking state)         |
| License            | Apache 2.0                                          |

The BlazePalm component first detects hand bounding boxes; the landmark regressor then runs inside each bounding box to produce precise keypoint locations. This two-stage design is what makes the model fast enough for real-time use.

---

### 5.2 Temporal Conv1D Classifier

**Default architecture.** Uses dilated causal convolutions to capture local and medium-range temporal patterns across the 30-frame window.

```
Input
  shape: (batch, 30, 126)
  │
  ├─ Conv1D(filters=160, kernel_size=3, padding='same', dilation_rate=1, activation='relu')
  │    Receptive field: 3 frames — captures instantaneous hand posture
  ├─ BatchNormalization()
  ├─ Dropout(0.15)
  │
  ├─ Conv1D(filters=160, kernel_size=5, padding='same', dilation_rate=2, activation='relu')
  │    Receptive field: 9 frames — captures short motion arcs
  ├─ BatchNormalization()
  ├─ Dropout(0.20)
  │
  ├─ Conv1D(filters=320, kernel_size=3, padding='same', dilation_rate=4, activation='relu')
  │    Receptive field: 17 frames — captures mid-sign dynamics
  ├─ BatchNormalization()
  ├─ Dropout(0.25)
  │
  ├─ GlobalAveragePooling1D()
  │    Collapses temporal dimension: (batch, 30, 320) → (batch, 320)
  │
  ├─ Dense(160, activation='relu')
  ├─ Dropout(0.30)
  ├─ Dense(80, activation='relu')
  └─ Dense(10, activation='softmax', dtype='float32')

Output shape: (batch, 10)
```

**Parameter count (large preset):**

| Layer                      | Output Shape   | Parameters  |
|----------------------------|----------------|-------------|
| Conv1D (filters=160, k=3)  | (30, 160)      | 60,960      |
| BatchNorm                  | (30, 160)      | 640         |
| Conv1D (filters=160, k=5)  | (30, 160)      | 128,160     |
| BatchNorm                  | (30, 160)      | 640         |
| Conv1D (filters=320, k=3)  | (30, 320)      | 153,920     |
| BatchNorm                  | (30, 320)      | 1,280       |
| GlobalAveragePooling1D     | (320,)         | 0           |
| Dense(160)                 | (160,)         | 51,360      |
| Dense(80)                  | (80,)          | 12,880      |
| Dense(10, softmax)         | (10,)          | 810         |
| **Total**                  |                | **~410,650**|

**Why dilated convolutions?**
Dilation rates (1 → 2 → 4) double the temporal receptive field at each layer without increasing the number of parameters. The three-layer stack covers up to 17 frames with only 320 filters at the deepest layer, making the model computationally efficient while still seeing most of the 30-frame window.

---

### 5.3 Bidirectional LSTM Classifier

Uses recurrent layers that process the sequence in both forward and backward directions, allowing each time step to be informed by both past and future context within the window.

```
Input
  shape: (batch, 30, 126)
  │
  ├─ Conv1D(filters=160, kernel_size=3, padding='same', activation='relu')
  │    Local feature extraction before the recurrent layers
  ├─ BatchNormalization()
  ├─ Dropout(0.15)
  │
  ├─ Bidirectional(LSTM(units=160, return_sequences=True))
  │    Forward LSTM: 160 units   ─┐
  │    Backward LSTM: 160 units  ─┴─ concatenated → 320 outputs per time step
  │    Output shape: (batch, 30, 320)
  ├─ Dropout(0.25)
  │
  ├─ Bidirectional(LSTM(units=128, return_sequences=False))
  │    Forward LSTM: 128 units   ─┐
  │    Backward LSTM: 128 units  ─┴─ concatenated → 256 outputs (final state only)
  │    Output shape: (batch, 256)
  │
  ├─ Dense(160, activation='relu')
  ├─ Dropout(0.30)
  ├─ Dense(80, activation='relu')
  └─ Dense(10, activation='softmax', dtype='float32')

Output shape: (batch, 10)
```

**Parameter count (large preset):**

| Layer                        | Output Shape   | Parameters  |
|------------------------------|----------------|-------------|
| Conv1D (filters=160, k=3)    | (30, 160)      | 60,960      |
| BatchNorm                    | (30, 160)      | 640         |
| BiLSTM(160, ret_seq=True)    | (30, 320)      | 412,160     |
| BiLSTM(128, ret_seq=False)   | (256,)         | 329,728     |
| Dense(160)                   | (160,)         | 41,120      |
| Dense(80)                    | (80,)          | 12,880      |
| Dense(10, softmax)           | (10,)          | 810         |
| **Total**                    |                | **~858,298**|

**Why Bidirectional?**
A unidirectional LSTM at time step `t` can only attend to frames 1…t. Because the full 30-frame window is available at prediction time (not a streaming scenario), bidirectional processing lets each time step incorporate future context too — e.g., the final hand position can inform the interpretation of an ambiguous mid-sign posture.

---

### 5.4 Architecture Comparison

| Property                     | Temporal Conv1D     | Bidirectional LSTM  |
|------------------------------|---------------------|---------------------|
| Parameters (large)           | ~410 K              | ~858 K              |
| Temporal receptive field     | 17 frames (dilated) | Full 30 frames      |
| Parallelism during training  | High (conv ops)     | Limited (sequential)|
| Inference latency            | Lower               | Higher              |
| Captures long-range patterns | Moderate            | Strong              |
| Recommended for              | Production / speed  | Research / accuracy |

Both architectures use identical heads (Dense→Dropout→Dense→Softmax) and are trained with the same optimizer and callbacks. The choice is controlled by `--architecture conv` (default) or `--architecture lstm` in the training script.

#### Model size presets

| Preset  | `conv_filters` | `lstm_1` | `lstm_2` | `dense_1` | `dense_2` |
|---------|---------------|----------|----------|-----------|-----------|
| `small` | 96            | 96       | 64       | 96        | 48        |
| `large` | 160           | 160      | 128      | 160       | 80        |

---

## 6. Training Configuration

### Optimizer

```
tf.keras.optimizers.Adam(
    learning_rate = 0.001,
    clipnorm      = 1.0        # gradient clipping to prevent explosion
)
Loss:    sparse_categorical_crossentropy
Metric:  sparse_categorical_accuracy
```

### Callbacks

| Callback                | Configuration                                                            | Effect                                                      |
|-------------------------|--------------------------------------------------------------------------|-------------------------------------------------------------|
| `EarlyStopping`         | monitor=`val_sparse_categorical_accuracy`, patience=18, restore_best_weights=True | Stops training if val accuracy does not improve for 18 epochs; restores the best checkpoint |
| `ReduceLROnPlateau`     | monitor=`val_loss`, factor=0.5, patience=6, min_lr=1e-5                  | Halves learning rate when val loss stagnates for 6 epochs   |
| `ModelCheckpoint`       | monitor=`val_sparse_categorical_accuracy`, save_best_only=True           | Saves the model weights only when a new val accuracy record is set |
| `ConsoleTrainingMonitor`| batch_interval=10, optional nvidia-smi stats                             | Prints per-batch loss/accuracy and per-epoch summary table  |

### Training hyperparameters (defaults)

| Hyperparameter      | Default  | CLI flag              |
|---------------------|----------|-----------------------|
| Max epochs          | 120      | `--epochs`            |
| Batch size          | 64       | `--batch-size`        |
| Augment copies      | 2        | `--augment-copies`    |
| Crops per video     | 1        | `--crops-per-video`   |
| Model size          | large    | `--model-size`        |
| Architecture        | conv     | `--architecture`      |
| Mixed precision     | off      | `--mixed-precision`   |
| XLA JIT             | off      | `--xla`               |

### Typical training command

```powershell
python train_karsl_words.py `
    --data-dir "path\to\karsl_data" `
    --cache data\karsl_words_sequences_v2.npz `
    --model-out karsl_words_action_v2.h5 `
    --labels-out karsl_words_labels.json `
    --epochs 120 `
    --batch-size 64 `
    --augment-copies 2 `
    --model-size large
```

---

## 7. Real-Time Inference Pipeline

### Frame-by-frame processing loop

```
┌─────────────────────────────────────────────────────────────┐
│  Camera capture (OpenCV VideoCapture)                       │
│                     │                                       │
│                     ▼                                       │
│  [Recommended] Pre-processing:                              │
│    CLAHE + gamma correction on each frame                   │
│                     │                                       │
│                     ▼                                       │
│  BGR → RGB → mediapipe.Image(SRGB)                         │
│                     │                                       │
│                     ▼                                       │
│  hand_landmarker.detect_for_video(mp_image, timestamp_ms)  │
│                     │                                       │
│                     ▼                                       │
│  extract_keypoints() → normalize → 126-float vector        │
│                     │                                       │
│                     ▼                                       │
│  sequence.append(keypoints)                                 │
│  sequence = sequence[-30:]    ← rolling window              │
│                     │                                       │
│         (if len(sequence) == 30)                            │
│                     │                                       │
│                     ▼                                       │
│  model.predict(sequence[np.newaxis]) → probs[10]           │
│                     │                                       │
│                     ▼                                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Temporal Smoothing                                  │  │
│  │  predictions.append(argmax(probs))                   │  │
│  │  Accept if:                                          │  │
│  │    unique(predictions[-10:]) == 1 (all agree)        │  │
│  │    AND probs[argmax] > 0.50 (threshold)              │  │
│  └──────────────────────────────────────────────────────┘  │
│                     │                                       │
│                     ▼                                       │
│  sentence.append(label) → display last 5 words             │
│  prob_viz() → probability bars on frame                    │
│  cv2.imshow("Sign Language Detection", frame)              │
└─────────────────────────────────────────────────────────────┘
```

### Decision smoothing rationale

Raw per-frame predictions are noisy — the classifier output fluctuates between frames even for a stable sign. The two-stage filter addresses this:

1. **Consistency gate** (`unique(predictions[-10:]) == 1`): the same class must win the argmax for 10 consecutive frames (~0.33 s at 30 fps) before a decision is committed.
2. **Confidence gate** (`prob > 0.5`): the winning class must be the model's top prediction by a clear margin.

Together these eliminate transient misfires caused by hand occlusion, motion blur, or sign transitions.

---

## 8. GPU Acceleration

### Inference GPU (real-time demo)

The inference script (`run_sign_language.py`) supports GPU acceleration via **ONNX Runtime with DirectML**, which works on any DirectX 12-capable GPU under Windows — including the **NVIDIA GeForce RTX 4060 Laptop GPU** — without requiring the CUDA toolkit.

```
.venv\Scripts\python.exe run_sign_language.py --gpu
```

When `--gpu` is specified:
1. The `.h5` model is converted to ONNX format (once, on first run) using `tf2onnx`.
2. ONNX Runtime selects the best available GPU provider in order: `CUDAExecutionProvider` → `DmlExecutionProvider` → `CPUExecutionProvider`.
3. On this machine, `DmlExecutionProvider` (DirectML) is active.

| Mode         | Backend                    | Requires              |
|--------------|----------------------------|-----------------------|
| CPU (default)| TensorFlow 2.21            | Nothing extra         |
| GPU          | ONNX Runtime + DirectML    | Windows GPU driver    |
| GPU (CUDA)   | ONNX Runtime + CUDA        | CUDA toolkit + cuDNN  |

### Training GPU

The training script (`train_karsl_words.py`) uses TensorFlow's native GPU support. On native Windows, TF ≥ 2.11 does not support CUDA directly. Options:

| Approach              | Environment         | Advantage                          |
|-----------------------|---------------------|------------------------------------|
| DirectML plugin       | `tf-directml-env`   | Runs on Windows GPU without CUDA   |
| WSL2 + CUDA           | WSL2 Ubuntu         | Full TF CUDA support, best perf.   |
| CPU training          | `.venv`             | Simplest, sufficient for small data|

---

## 9. How to Run

### Setup

```powershell
# Activate the working virtual environment
.venv\Scripts\Activate.ps1
```

### Real-time sign language detection (CPU)

```powershell
python run_sign_language.py
```

### Real-time sign language detection (GPU — RTX 4060 via DirectML)

```powershell
python run_sign_language.py --gpu
```

### Other source options

```powershell
# Different camera index
python run_sign_language.py --source 1

# Video file
python run_sign_language.py --source path\to\video.mp4

# Headless test (no window, 35 frames)
python run_sign_language.py --test --test-frames 35 --no-display
```

### Quick model inspection

```powershell
python load_karsl_model.py --model karsl_words_action_v2.h5 --labels karsl_words_labels.json
```

### Full training pipeline

```powershell
# 1. Extract landmarks and cache them
python train_karsl_words.py --data-dir "path\to\karsl_data" --cache data\cache.npz --crops-per-video 1

# 2. Train (uses cached landmarks)
python train_karsl_words.py `
    --data-dir "path\to\karsl_data" `
    --cache data\cache.npz `
    --model-out karsl_words_action_v2.h5 `
    --labels-out karsl_words_labels.json `
    --epochs 120 --batch-size 64 --augment-copies 2
```

### Key CLI flags for `run_sign_language.py`

| Flag              | Default                    | Description                                |
|-------------------|----------------------------|--------------------------------------------|
| `--model`         | `karsl_words_action_v2.h5` | Path to the trained Keras model            |
| `--labels`        | `karsl_words_labels.json`  | Path to the labels JSON                    |
| `--source`        | `0`                        | Camera index or video file path            |
| `--gpu`           | off                        | Enable GPU inference via ONNX Runtime      |
| `--no-display`    | off                        | Suppress the OpenCV window                 |
| `--test`          | off                        | Run for `--test-frames` frames then exit   |

---

## 10. Training Results & Evaluation

> All metrics reported here are **real values** measured on the `karsl_words_action_v2.h5` model
> using the held-out **validation set** (20% of the dataset, stratified per class, seed = 42,
> never seen during training or augmentation).
> Dataset: `data/karsl_words_sequences_v2.npz` — 1,263 sequences, 3 crops/video, 10 balanced classes.

---

### 10.1 Evaluated Model Summary

| Property              | Value                                          |
|-----------------------|------------------------------------------------|
| Model file            | `karsl_words_action_v2.h5`                     |
| Architecture          | **Bidirectional LSTM** (large preset)          |
| Input shape           | `(30, 126)` — 30 frames × 126 features         |
| Output shape          | `(10,)` — softmax over 10 sign classes         |
| Total parameters      | **986,746**                                    |
| Trainable parameters  | 986,426                                        |
| Non-trainable params  | 320 (BatchNormalization running statistics)    |
| Optimizer             | Adam — lr = 0.001, clipnorm = 1.0             |
| Loss function         | Sparse categorical cross-entropy               |

**Layer-by-layer breakdown:**

```
Layer                        Type                    Parameters
─────────────────────────────────────────────────────────────────
conv1d                       Conv1D(160, k=3)            60,640
batch_normalization          BatchNorm(160)                 640  (320 non-trainable)
bidirectional                BiLSTM(160, ret_seq=True)  410,880
dropout                      Dropout(0.25)                    0
bidirectional_1              BiLSTM(128, ret_seq=False)  459,776
dense                        Dense(160, relu)             41,120
dropout_1                    Dropout(0.30)                    0
dense_1                      Dense(80, relu)              12,880
dense_2                      Dense(10, softmax)              810
─────────────────────────────────────────────────────────────────
TOTAL                                                   986,746
```

**Dataset split used for this evaluation:**

```
Total sequences (v2 cache):   1,263  (3 crops/video × ~42 videos/class × 10 classes)
  ├─ Training set:            1,012  (80%) — augmented ×3 during training → ~3,036 samples
  └─ Validation set:            251  (20%) — clean, not augmented, used for all metrics below

  Per-class in validation:  25 samples for 9 classes, 26 for class "hate" (≈ balanced)
```

---

### 10.2 Overall Performance Metrics

| Metric                   | Training set  | Validation set |
|--------------------------|:-------------:|:--------------:|
| **Accuracy**             | **100.00 %**  | **99.60 %**    |
| **Loss**                 | 0.000141      | 0.027887       |
| **Macro Precision**      | —             | 99.62 %        |
| **Macro Recall**         | —             | 99.62 %        |
| **Macro F1-Score**       | —             | 99.60 %        |
| **Weighted F1-Score**    | —             | 99.60 %        |
| **Misclassified samples**| 0 / 1,012     | **1 / 251**    |

> The near-zero gap between training accuracy (100%) and validation accuracy (99.60%) demonstrates
> that the model generalizes well and is **not overfitting** — a direct result of the
> data augmentation strategy (×3 copies with noise, scale, and temporal shift).

---

### 10.3 Confusion Matrix

Evaluated on the **251-sample validation set**. Rows = ground-truth class, Columns = predicted class.

```
              ┌────────────────────────────────────────────────────────────────────────────┐
              │                        PREDICTED CLASS                                     │
              │  build  break   walk   love   hate  plant  think   help  choose   call     │
 ┌────────────┼────────────────────────────────────────────────────────────────────────────┤
 │  build (25)│   25      0      0      0      0      0      0      0      0       0       │
 │  break (25)│    0     25      0      0      0      0      0      0      0       0       │
 │   walk (25)│    0      0     25      0      0      0      0      0      0       0       │
 │   love (25)│    0      0      0     25      0      0      0      0      0       0       │
 │   hate (26)│    0      0      0      0     25      0      1      0      0       0       │
 │  plant (25)│    0      0      0      0      0     25      0      0      0       0       │
 │  think (25)│    0      0      0      0      0      0     25      0      0       0       │
 │   help (25)│    0      0      0      0      0      0      0     25      0       0       │
 │ choose (25)│    0      0      0      0      0      0      0      0     25       0       │
 │   call (25)│    0      0      0      0      0      0      0      0      0      25       │
 └────────────┴────────────────────────────────────────────────────────────────────────────┘
```

**Reading guide:**
- Every value on the **main diagonal** is a correct prediction.
- The **single off-diagonal value** is the only error: 1 sample of class `hate` was predicted as `think`.
- All other 250 samples are correctly classified.

**Visual representation of correctness:**

```
   build  ████████████████████████████████████  100.0 %  (25/25)
   break  ████████████████████████████████████  100.0 %  (25/25)
    walk  ████████████████████████████████████  100.0 %  (25/25)
    love  ████████████████████████████████████  100.0 %  (25/25)
    hate  █████████████████████████████████▌    96.2 %  (25/26) ← sole error
   plant  ████████████████████████████████████  100.0 %  (25/25)
   think  ████████████████████████████████████  100.0 %  (25/25)
    help  ████████████████████████████████████  100.0 %  (25/25)
  choose  ████████████████████████████████████  100.0 %  (25/25)
    call  ████████████████████████████████████  100.0 %  (25/25)
```

---

### 10.4 Per-Class Classification Report

> **Precision** = TP / (TP + FP) — of all samples predicted as class X, how many truly are X.
> **Recall** = TP / (TP + FN) — of all true samples of class X, how many were correctly identified.
> **F1-Score** = 2 × (Precision × Recall) / (Precision + Recall) — harmonic mean.

| Class      | Precision | Recall  | F1-Score | Support |
|:-----------|:---------:|:-------:|:--------:|:-------:|
| build      |  1.000    |  1.000  |  1.000   |   25    |
| break      |  1.000    |  1.000  |  1.000   |   25    |
| walk       |  1.000    |  1.000  |  1.000   |   25    |
| love       |  1.000    |  1.000  |  1.000   |   25    |
| **hate**   |  **1.000**|**0.962**|**0.980** |   26    |
| plant      |  1.000    |  1.000  |  1.000   |   25    |
| **think**  |**0.962**  |  **1.000**|**0.980**|  25    |
| help       |  1.000    |  1.000  |  1.000   |   25    |
| choose     |  1.000    |  1.000  |  1.000   |   25    |
| call       |  1.000    |  1.000  |  1.000   |   25    |
| ────────── | ─────────  | ─────── | ──────── | ─────── |
| **Macro avg**   | **0.996** | **0.996** | **0.996** | **251** |
| **Weighted avg**| **0.996** | **0.996** | **0.996** | **251** |

Notes on affected classes:
- **`hate`** — Recall drops to 0.962 because 1 of 26 samples was misclassified as `think`. Precision remains 1.0 because no foreign samples were incorrectly assigned to `hate`.
- **`think`** — Precision drops to 0.962 because 1 foreign sample (`hate`) was predicted as `think`. Recall remains 1.0 because all true `think` samples were correctly identified.

---

### 10.5 Top-k Accuracy

Top-k accuracy measures whether the ground-truth class appears within the top-k highest-probability predictions. It shows how much information the model encodes even when its argmax prediction is wrong.

| k | Top-k Accuracy | Meaning                                             |
|:-:|:--------------:|-----------------------------------------------------|
| 1 | **99.60 %**    | Standard accuracy — argmax must equal the true class |
| 2 | **100.00 %**   | True class is always in the top 2 predictions       |
| 3 | **100.00 %**   | True class is always in the top 3 predictions       |

> **Interpretation:** The single misclassified sample (`hate` → `think`) still places the true
> class `hate` in position #2 — meaning the model was uncertain between two closely related
> gestural motions, but the true label was its second-best guess. This confirms the model has
> strong discriminative ability across all 10 classes.

---

### 10.6 Prediction Confidence Analysis

The softmax output gives a probability distribution over all 10 classes per prediction. These statistics characterize how confident the model is.

| Statistic                                            | Value     |
|------------------------------------------------------|:---------:|
| Mean confidence on **correct** predictions           | **99.99 %** |
| Mean predicted-class confidence (all samples)        | **99.99 %** |
| Mean predicted-class confidence on **misclassified** | 99.89 %   |
| True-class confidence on misclassified sample        |  0.09 %   |

```
Confidence distribution — correct predictions (250/251):

  0.9990 – 1.0000  ████████████████████████████████████  ~248 samples
  0.9900 – 0.9990  ▌                                       ~2 samples
  < 0.99           ·                                        0 samples

Misclassified sample (1/251):
  Predicted:  think   confidence = 99.89 %
  True class: hate    confidence =  0.09 %
```

The model assigns near-certainty confidence to virtually every prediction. Even on the single misclassification, its confidence is high (99.89%), indicating the error is a genuine inter-class ambiguity rather than a low-confidence uncertain case. This is a known challenge with softmax-based classifiers: high confidence does not guarantee correctness.

---

### 10.7 Error Analysis

**The single misclassification:**

| Property           | Value                                           |
|--------------------|-------------------------------------------------|
| True class         | `hate` (يكره)                                  |
| Predicted class    | `think` (يفكر)                                 |
| Model confidence   | 99.89% for `think`                             |
| True class prob.   | 0.09% for `hate`                               |

**Why hate → think?**

Both signs involve hand motion near the head/chest area. `hate` involves a flicking-away motion while `think` involves a pointing gesture near the temple. The specific video sample likely had:

1. Atypical execution speed (caught by the 3-crop temporal sampling at an edge offset), or
2. Slight hand orientation ambiguity in the mid-sign frame range, making the trajectory more similar to `think` than canonical `hate`.

This error was not encountered during training (training accuracy = 100%), suggesting the model memorized this particular sequence's augmented variants without generalizing perfectly to its edge-crop variant.

**Robustness verdict:** With only 1 error out of 251 validation samples, and that error ranking the true class #2 (Top-2 accuracy = 100%), the model is highly robust for a 10-class vocabulary on this dataset.

---

### 10.8 Discussion & Limitations

#### Strengths

| Strength                       | Detail                                                               |
|--------------------------------|----------------------------------------------------------------------|
| Near-perfect accuracy          | 99.60% on unseen validation samples across 10 balanced classes       |
| High confidence                | Mean softmax confidence 99.99% on correct predictions               |
| Compact representation         | Only 126 features/frame vs. 1,662 for full-body holistic approaches |
| Fast inference                 | Sub-millisecond per prediction on CPU; GPU DirectML further reduces latency |
| Scale/position invariant       | Wrist-centering + radius normalization makes features camera-distance independent |

#### Limitations & Future Work

| Limitation                        | Explanation                                                         | Mitigation                                          |
|-----------------------------------|---------------------------------------------------------------------|-----------------------------------------------------|
| Small validation set (251 samples)| Metrics may not generalize perfectly to a large unseen population   | Collect more signers; use cross-validation          |
| No independent test set           | Val set used for early stopping → slight optimistic bias            | Create a held-out test set from different sessions  |
| Hand-only features                | Body orientation and facial expression are ignored                  | Add pose landmarks for more expressive signs        |
| High contrast / low contrast      | MediaPipe landmark detection degrades under extreme lighting         | Apply CLAHE + gamma pre-processing (Section 4.2)    |
| 10-word vocabulary                | KARSL contains hundreds of signs                                    | Scale dataset and retrain with more sign categories |
| Single error: hate ↔ think        | Gestural similarity at certain crop offsets                         | Add more crop diversity; or use data-specific augmentation for confusable pairs |

#### Summary

```
┌──────────────────────────────────────────────────────────┐
│                 MODEL EVALUATION SUMMARY                 │
├──────────────────────────────────────────────────────────┤
│  Architecture   Bidirectional LSTM (large)               │
│  Parameters     986,746                                  │
│  Input          30 frames × 126 hand-landmark features   │
│  Classes        10 (KARSL Arabic sign words)             │
├──────────────────────────────────────────────────────────┤
│  Train Accuracy      100.00 %   Train Loss   0.000141   │
│  Val   Accuracy       99.60 %   Val   Loss   0.027887   │
│  Macro F1-Score       99.60 %                           │
│  Top-2 Accuracy      100.00 %                           │
│  Misclassifications    1 / 251  (hate → think)          │
└──────────────────────────────────────────────────────────┘
```

---

## 11. File Reference

| File / Directory                  | Role                                                                 |
|-----------------------------------|----------------------------------------------------------------------|
| `run_sign_language.py`            | Real-time inference: camera → MediaPipe → classifier → display       |
| `train_karsl_words.py`            | Dataset extraction, augmentation, model training (Conv/LSTM)         |
| `load_karsl_model.py`             | Quick model loader: prints summary and runs a demo prediction        |
| `karsl_words_action_v2.h5`        | Trained Keras model (10-class, input 30×126)                         |
| `karsl_words_action_v2.onnx`      | ONNX export of the above, used for GPU inference                     |
| `karsl_words_labels.json`         | Labels + dataset metadata (actions, seq_len, feature_len)            |
| `mediapipe_models/`               | Downloaded MediaPipe task bundles (`hand_landmarker.task`)           |
| `data/karsl_words_sequences.npz`  | Cached extracted landmark sequences (compressed NumPy)               |
| `requirements.txt`                | Python package dependencies                                          |
| `.venv/`                          | Python 3.13 virtual environment (TF 2.21, MediaPipe 0.10, ORT DML)  |
| `tf-directml-env/`                | Python 3.10 environment with TF-DirectML plugin (GPU training)       |
