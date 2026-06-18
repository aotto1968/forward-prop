#!/bin/bash
# Download MNIST dataset for Otto Score inference demo
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$DIR/mnist"

echo "Downloading MNIST training images..."
curl -L -o "$DIR/mnist/train-images-idx3-ubyte.gz" \
  "https://github.com/aotto1968/forward-prop/raw/main/www/data/mnist/train-images-idx3-ubyte.gz" 2>/dev/null || \
  wget -O "$DIR/mnist/train-images-idx3-ubyte.gz" \
  "https://github.com/aotto1968/forward-prop/raw/main/www/data/mnist/train-images-idx3-ubyte.gz" 2>/dev/null || {
    echo "Download failed. Trying alternative source..."
    curl -L -o "$DIR/mnist/train-images-idx3-ubyte.gz" \
      "https://raw.githubusercontent.com/aotto1968/forward-prop/main/www/data/mnist/train-images-idx3-ubyte.gz"
}

echo "Downloading MNIST training labels..."
curl -L -o "$DIR/mnist/train-labels-idx1-ubyte.gz" \
  "https://github.com/aotto1968/forward-prop/raw/main/www/data/mnist/train-labels-idx1-ubyte.gz" 2>/dev/null || \
  wget -O "$DIR/mnist/train-labels-idx1-ubyte.gz" \
  "https://github.com/aotto1968/forward-prop/raw/main/www/data/mnist/train-labels-idx1-ubyte.gz" 2>/dev/null || {
    curl -L -o "$DIR/mnist/train-labels-idx1-ubyte.gz" \
      "https://raw.githubusercontent.com/aotto1968/forward-prop/main/www/data/mnist/train-labels-idx1-ubyte.gz"
}

echo "MNIST data downloaded to $DIR/mnist/"
ls -lh "$DIR/mnist/"
