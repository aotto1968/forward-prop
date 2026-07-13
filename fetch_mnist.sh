#!/bin/bash
# Download MNIST dataset for Otto Score inference demo
# Source: https://forward-prop.nhi1.de/data/mnist/
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
OUTDIR="$DIR/www/data/mnist"
mkdir -p "$OUTDIR"

# List of files to download: local_name source_url
FILES=(
  "train-images-idx3-ubyte.gz"
  "train-labels-idx1-ubyte.gz"
  "t10k-images-idx3-ubyte.gz"
  "t10k-labels-idx1-ubyte.gz"
)

# Minimum expected sizes (bytes) to detect broken downloads
declare -A MINSIZE
MINSIZE["train-images-idx3-ubyte.gz"]=9000000
MINSIZE["train-labels-idx1-ubyte.gz"]=20000
MINSIZE["t10k-images-idx3-ubyte.gz"]=1500000
MINSIZE["t10k-labels-idx1-ubyte.gz"]=4000

PBASE="https://forward-prop.nhi1.de/data/mnist"

fetch_file() {
    local fname="$1"
    local outpath="$OUTDIR/$fname"
    local minsize="${MINSIZE[$fname]:-0}"

    echo "Downloading $fname ..."

    # Try curl first, fall back to wget
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
    return 1
}

for fname in "${FILES[@]}"; do
    fetch_file "$fname" || exit 1
done

echo ""
echo "MNIST data downloaded to $OUTDIR/"
ls -lh "$OUTDIR/"
