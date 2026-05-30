"""
Simple loader/demo for KARSL .h5 model and labels.

Usage:
    python load_karsl_model.py --model karsl_words_action_v2.h5 --labels karsl_words_labels.json --use-random

This prints the model summary and runs a single demo prediction on a zero or random input.
"""

import argparse
import json
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


def main():
    parser = argparse.ArgumentParser(description='Load and demo a KARSL .h5 model')
    parser.add_argument('--model', default='karsl_words_action_v2.h5', help='Path to .h5 model file')
    parser.add_argument('--labels', default='karsl_words_labels.json', help='Path to labels JSON file')
    parser.add_argument('--use-random', action='store_true', help='Use random noise as demo input (default: zeros)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for demo input')
    parser.add_argument('--top-k', type=int, default=3, help='Show top-K predictions')
    args = parser.parse_args()

    # Load labels/metadata
    try:
        with open(args.labels, 'r', encoding='utf-8') as f:
            labels = json.load(f)
    except Exception as e:
        print(f'Error reading labels file {args.labels}: {e}', file=sys.stderr)
        sys.exit(1)

    actions = labels.get('actions') or labels.get('actions', [])
    seq_len = int(labels.get('sequence_length', 30))
    feat_len = int(labels.get('feature_length', 21 * 3 * 2))

    print(f'Loaded labels: {len(actions)} actions')
    if actions:
        print('Actions:', actions)
    print(f'Sequence length: {seq_len}, feature length: {feat_len}\n')

    # Load model
    print(f'Loading model from: {args.model}')
    try:
        model = load_model(args.model, compile=False)
    except Exception as e:
        print(f'Failed to load model: {e}', file=sys.stderr)
        sys.exit(1)

    print('\nModel summary:')
    model.summary()

    # Prepare demo input
    rng = np.random.default_rng(args.seed)
    if args.use_random:
        demo = rng.normal(0.0, 0.02, size=(1, seq_len, feat_len)).astype(np.float32)
    else:
        demo = np.zeros((1, seq_len, feat_len), dtype=np.float32)

    # Run prediction
    print('\nRunning demo prediction...')
    preds = model.predict(demo, verbose=0)
    preds = np.asarray(preds)
    if preds.ndim == 1:
        preds = preds[np.newaxis, ...]

    probs = preds[0]
    if actions and len(actions) != probs.shape[0]:
        print('Warning: number of actions in labels does not match model output size', file=sys.stderr)

    top_k = min(args.top_k, probs.shape[0])
    top_idxs = np.argsort(probs)[::-1][:top_k]
    print('\nTop predictions:')
    for rank, idx in enumerate(top_idxs, start=1):
        label = actions[idx] if idx < len(actions) else f'index_{idx}'
        print(f'  {rank}. {label} (index={idx}) -> {probs[idx]:.4f}')


if __name__ == '__main__':
    main()
