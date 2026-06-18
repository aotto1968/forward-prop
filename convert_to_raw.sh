#!/bin/bash
# Convert an image to MNIST-compatible raw 28x28 grayscale format
# Usage: bash convert_to_raw.sh input.jpg [output.raw]
#
# Requirements: ImageMagick (convert command)
#
# The output is 784 raw bytes, uint8, row-major, 0=white 255=black.
# This can be fed to: mlp-otto-score-ifc-xnor.exe --image output.raw

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 input.jpg [output.raw]"
    echo "Converts any image to MNIST-format raw 28x28 grayscale."
    echo "Requires ImageMagick."
    exit 1
fi

INPUT="$1"
OUTPUT="${2:-${INPUT%.*}.raw}"

if ! command -v convert &>/dev/null; then
    echo "ERROR: ImageMagick 'convert' not found."
    echo "Install: apt install imagemagick   (Debian/Ubuntu)"
    echo "         brew install imagemagick   (macOS)"
    echo "         dnf install ImageMagick    (Fedora)"
    exit 1
fi

echo "Converting $INPUT → $OUTPUT (28×28 grayscale raw)"

# -resize 28x28!  : force exact size (stretch if needed)
# -negate        : invert (MNIST has white background, black ink)
# -depth 8       : 8-bit grayscale
# gray:-         : output raw bytes to stdout
convert "$INPUT" -resize 28x28! -negate -depth 8 gray:- > "$OUTPUT"

BYTES=$(stat -c%s "$OUTPUT" 2>/dev/null || stat -f%z "$OUTPUT" 2>/dev/null)
if [ "$BYTES" = "784" ]; then
    echo "Done. $BYTES bytes written."
    echo "Classify: mlp-otto-score-ifc-xnor.exe --model models/model-xnor.otto --image $OUTPUT"
else
    echo "WARNING: Expected 784 bytes, got $BYTES. File may be corrupted."
fi
