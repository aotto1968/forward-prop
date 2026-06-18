#!/bin/bash
# Convert an image to MNIST-compatible PGM (Portable GrayMap) format.
# PGM is a standard format viewable in any image viewer.
# Usage: bash convert_to_pgm.sh input.jpg [output.pgm]
#
# Requirements: ImageMagick (convert command)
#
# The classifier accepts PGM (P5) directly:
#   ./mlp-otto-score-ifc-xnor.exe --model models/model-xnor.otto --image output.pgm

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 input.jpg [output.pgm]"
    echo "Converts any image to MNIST-format PGM (28x28 grayscale)."
    echo "Requires ImageMagick."
    exit 1
fi

INPUT="$1"
OUTPUT="${2:-${INPUT%.*}.pgm}"

if ! command -v convert &>/dev/null; then
    echo "ERROR: ImageMagick 'convert' not found."
    echo "Install: apt install imagemagick   (Debian/Ubuntu)"
    echo "         brew install imagemagick   (macOS)"
    exit 1
fi

echo "Converting $INPUT → $OUTPUT (28×28 PGM grayscale)"

# -resize 28x28!  : force exact size (stretch if needed)
# -negate        : invert (MNIST has white background, black ink)
# -depth 8       : 8-bit grayscale
# pgm:-          : output PGM to stdout (standard image format)
convert "$INPUT" -resize 28x28! -negate -depth 8 pgm:- > "$OUTPUT"

echo "Done."
echo "Classify: ./mlp-otto-score-ifc-xnor.exe --model models/model-xnor.otto --image $OUTPUT"
