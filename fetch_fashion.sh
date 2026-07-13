#!/bin/bash
# fetch_fashion.sh — Download Fashion-MNIST dataset
# =====================================================
# Downloads Fashion-MNIST (same IDX3 format as MNIST)
# to www/data/mnist-fashion/.
#
# Usage:
#   bash fetch_fashion.sh          # download to www/data/mnist-fashion/
#
# Fashion-MNIST: 28×28 grayscale, 10 classes (clothing items)
# Labels: 0=T-shirt/top, 1=Trouser, 2=Pullover, 3=Dress, 4=Coat,
#         5=Sandal, 6=Shirt, 7=Sneaker, 8=Bag, 9=Ankle boot
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
OUTDIR="$DIR/www/data/mnist-fashion"
mkdir -p "$OUTDIR"

FILES=(
  "train-images-idx3-ubyte.gz"
  "train-labels-idx1-ubyte.gz"
  "t10k-images-idx3-ubyte.gz"
  "t10k-labels-idx1-ubyte.gz"
)

declare -A MINSIZE
MINSIZE["train-images-idx3-ubyte.gz"]=20000000
MINSIZE["train-labels-idx1-ubyte.gz"]=25000
MINSIZE["t10k-images-idx3-ubyte.gz"]=4000000
MINSIZE["t10k-labels-idx1-ubyte.gz"]=4500

PBASE="https://forward-prop.nhi1.de/data/mnist-fashion"

fetch_file() {
    local fname="$1"
    local outpath="$OUTDIR/$fname"
    local minsize="${MINSIZE[$fname]:-0}"

    echo "Downloading $fname ..."

    if curl -L -f -sS -o "$outpath" "$PBASE/$fname" 2>/dev/null; then
        local actual
        actual=$(stat -c%s "$outpath" 2>/dev/null || echo 0)
        if [ "$actual" -ge "$minsize" ]; then
            echo "  OK ($actual bytes)"
            return 0
        fi
        echo "  Too small ($actual bytes, need $minsize)"
    fi

    if wget -q -O "$outpath" "$PBASE/$fname" 2>/dev/null; then
        local actual
        actual=$(stat -c%s "$outpath" 2>/dev/null || echo 0)
        if [ "$actual" -ge "$minsize" ]; then
            echo "  OK ($actual bytes, wget)"
            return 0
        fi
    fi

    echo "  FAILED: Could not download $fname"
    echo "  Try: curl -L -o $outpath $PBASE/$fname"
    return 1
}

for fname in "${FILES[@]}"; do
    fetch_file "$fname" || exit 1
done

echo ""
echo "Fashion-MNIST data downloaded to $OUTDIR/"
ls -lh "$OUTDIR/"
