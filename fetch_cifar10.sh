#!/bin/bash
# fetch_cifar10.sh — Download CIFAR-10 binary dataset
# ====================================================
# Downloads and extracts CIFAR-10 binary to data/cifar-10-batches-bin/
#
# Usage:
#   bash fetch_cifar10.sh          # download to data/
#   bash fetch_cifar10.sh /path   # download to /path/
#
# CIFAR-10 binary format:
#   Each file: 10000 × (1 byte label + 1024 R + 1024 G + 1024 B)
#   5 train batches + 1 test batch
#
# Mirrors (in order of preference):
#   1. https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz  (official)
#   2. https://pjreddie.com/media/files/cifar-10-binary.tar.gz  (pjreddie)

set -euo pipefail

BASE="${1:-data}"
TARGET="$BASE/cifar-10-batches-bin"

if [ -f "$TARGET/data_batch_1.bin" ]; then
    echo "[OK] CIFAR-10 already downloaded: $TARGET"
    ls -1 "$TARGET"/*.bin 2>/dev/null
    exit 0
fi

mkdir -p "$BASE"

URLS=(
    "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
    "https://pjreddie.com/media/files/cifar-10-binary.tar.gz"
)

TARBALL="/tmp/cifar-10-binary.tar.gz"
DOWNLOADED=0

for URL in "${URLS[@]}"; do
    echo "Trying: $URL"
    if command -v curl &>/dev/null; then
        curl -L -o "$TARBALL" "$URL" --connect-timeout 30 --max-time 600 -s && { DOWNLOADED=1; break; }
    elif command -v wget &>/dev/null; then
        wget -q --timeout=30 "$URL" -O "$TARBALL" && { DOWNLOADED=1; break; }
    fi
done

if [ "$DOWNLOADED" -eq 0 ]; then
    echo "[ERROR] Failed to download CIFAR-10 from any mirror."
    echo "  Manually download from: https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
    echo "  Then extract to: $TARGET/"
    exit 1
fi

echo "Extracting to $BASE/ ..."
tar -xzf "$TARBALL" -C "$BASE/" --strip-components=1 \
    cifar-10-batches-bin/data_batch_1.bin \
    cifar-10-batches-bin/data_batch_2.bin \
    cifar-10-batches-bin/data_batch_3.bin \
    cifar-10-batches-bin/data_batch_4.bin \
    cifar-10-batches-bin/data_batch_5.bin \
    cifar-10-batches-bin/test_batch.bin \
    cifar-10-batches-bin/batches.meta.txt

echo "[OK] CIFAR-10 downloaded to $TARGET"
ls -1 "$TARGET"/*.bin
echo ""
echo "Done. Run 'make all' to build programs."
