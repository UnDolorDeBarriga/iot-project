#!/bin/bash
#
# Usage: ./run_all_inference.sh /path/to/png-folder
#
# This will run ./infer_emnist on every .png in the given folder.

if [ $# -ne 1 ]; then
  echo "Usage: $0 <folder-containing-pngs>"
  exit 1
fi

IMG_DIR="$1"

if [ ! -d "$IMG_DIR" ]; then
  echo "Error: '$IMG_DIR' is not a directory"
  exit 1
fi

for img in "$IMG_DIR"/*.png; do
  # If there are no .png files, this glob expands literally to "*.png"
  if [ ! -f "$img" ]; then
    echo "No PNGs found in $IMG_DIR"
    exit 0
  fi

  echo "=== Inference on $(basename "$img") ==="
  ./infer_emnist "$img"
  echo
done
