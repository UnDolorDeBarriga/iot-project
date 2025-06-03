#!/usr/bin/env python3
"""
train_one_image.py

Load an existing trainable .tflite, run exactly one training step on
the given 28x28 PNG + label, then overwrite (or write) the updated
.tflite file.

Usage:
  ./train_one_image.py \
    --model /path/to/emnist_litert_train.tflite \
    --image /path/to/28x28.png \
    --label  5 \
    --out   /path/to/emnist_litert_train_updated.tflite
"""

import argparse
import sys
import numpy as np
from PIL import Image
import tensorflow as tf


def load_and_preprocess(path):
    """
    Load a 28x28 grayscale PNG, return a float32 tensor of shape (1,28,28,1).
    """
    img = Image.open(path).convert("L").resize((28, 28))
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr.reshape((1, 28, 28, 1))


def main():
    p = argparse.ArgumentParser(
        description="Run one TFLite training step on a single 28x28 PNG + label."
    )
    p.add_argument(
        "--model", required=True,
        help="Path to existing trainable .tflite (e.g. emnist_litert_train.tflite)."
    )
    p.add_argument(
        "--image", required=True,
        help="Path to a 28x28 grayscale PNG to train on."
    )
    p.add_argument(
        "--label", type=int, required=True,
        help="Integer label (0-25) for that image."
    )
    p.add_argument(
        "--out", required=True,
        help="Where to write the updated .tflite (can be same as --model to overwrite)."
    )
    args = p.parse_args()

    # 1) Load the trainable TFLite into a Python Interpreter
    interpreter = tf.lite.Interpreter(
        model_path=args.model,
        experimental_enable_resource_variables=True,
        experimental_preserve_all_tensors=True
    )
    interpreter.allocate_tensors()

    # 2) Grab the 'train' signature runner
    try:
        train_runner = interpreter.get_signature_runner("train")
    except ValueError:
        print(
            "ERROR: Could not find a 'train' signature in that .tflite.\n"
            "Make sure it was converted with both 'train' and 'infer' signatures.",
            file=sys.stderr
        )
        sys.exit(1)

    # 3) Preprocess the single PNG → shape (1,28,28,1) float32
    x = load_and_preprocess(args.image)
    y = np.array([args.label], dtype=np.int64)  # shape (1,)

    # 4) Run exactly one training step
    out = train_runner(images=x, labels=y)
    loss_val = out["loss"].numpy().item()
    print(f"Trained on '{args.image}' (label={args.label}) → loss={loss_val:.4f}")

    # 5) Write the updated FlatBuffer back out
    buf = interpreter._get_c_tensorbuffer()  # internal buffer → full .tflite
    with open(args.out, "wb") as f:
        f.write(buf)

    print(f"Wrote updated model to: {args.out}")


if __name__ == "__main__":
    main()
