# Makefile - Otto Score + Reference Implementations (public demo)
# ==============================================================
# Self-contained build for the public GitHub release.
#
# Programs:
#   mlp-otto-score-ifc.c          - Otto Score inference (bitwise MAJ3 + Bayes)
#   mlp-otto-score-ensemble.c     - Otto Score ensemble trainer (iterative correction)
#   mlp-flt32-trn-w1-adam.c       - Float32 2-layer AdamW trainer (reference)
#   mlp-flt32-ifc.c               - Float32 2-layer inference (reference)
#   mlp-bin32-trn-w1-hebbian.c    - Bitwise Hebbian trainer (reference, ~78%)
#   mlp-bin32-ifc.c               - Bin32 inference (XNOR+popcnt, no float)
#
# Shared headers:
#   ki-common.h       - float32/matmul/AdamW helpers (shared with flt32)
#   ki-otto-common.h  - Otto Score specific (batch correction, precision)
#   lib/maj3.h, lib/w0_random.h
# Targets:
#   make (all)     — builds everything
#   make setup     — download MNIST dataset (required once)
#   make otto      — Otto Score inference only
#   make flt32     — Float32 AdamW trainer + ifc
#   make hebbian   — Hebbian trainer + bin32-ifc
#   make ensemble  — Otto Score ensemble trainer
#   make clean     — remove executables
#   make test      — quick accuracy test with bundled models
#   make push      — push to GitHub (requires configured remote)

CC       = gcc
CFLAGS   = -O3 -march=native -fopenmp -Wall -Wextra -Werror \
           -Wconversion -Wsign-conversion -Wfloat-equal -Wundef -Wshadow -Wunused \
           -Ilib
LDLIBS   = -lm -lz

.PHONY: all otto flt32 hebbian ensemble setup clean test test-image push

all: otto flt32 hebbian ensemble

setup:
	@echo "Downloading MNIST dataset..."
	@bash fetch_mnist.sh

# ── Otto Score (bitwise MAJ3 + Bayes log-Score) ────────────────────

otto: mlp-otto-score-ifc-xnor.exe mlp-otto-score-ifc-xor.exe

mlp-otto-score-ifc-xnor.exe: mlp-otto-score-ifc.c ki-common.h lib/maj3.h
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

mlp-otto-score-ifc-xor.exe: mlp-otto-score-ifc.c ki-common.h lib/maj3.h
	$(CC) $(CFLAGS) -DH0_XOR -o $@ $< $(LDLIBS)

test-image: mlp-otto-score-ifc-xnor.exe
	@echo "=== Single image classification test ==="
	@echo "  tests/digit5.pgm (MNIST train[0], label=5)..."
	@./mlp-otto-score-ifc-xnor.exe --model models/model-xnor.otto --image tests/digit5.pgm 2>&1 | grep 'Predicted'
	@echo "  tests/digit9.pgm (MNIST train[4], label=9)..."
	@./mlp-otto-score-ifc-xnor.exe --model models/model-xnor.otto --image tests/digit9.pgm 2>&1 | grep 'Predicted'
	@echo "  tests/shoe.pgm (Fashion-MNIST ankle boot — should NOT be a digit)"
	@./mlp-otto-score-ifc-xnor.exe --model models/model-xnor.otto --image tests/shoe.pgm 2>&1 | grep 'Predicted'

test: mlp-otto-score-ifc-xnor.exe mlp-otto-score-ifc-xor.exe mlp-flt32-ifc.exe mlp-bin32-ifc-xnor.exe \
      models/flt32-w1-h512/weights.meta models/hebbian-h512/weights.meta
	@echo "=== Otto Score XNOR v1 single (H=512, eval=10000) ==="
	@./mlp-otto-score-ifc-xnor.exe --model models/model-xnor.otto --evalN 10000 2>&1 | grep -E 'Eval:|Time:'
	@echo ""
	@echo "=== Otto Score XOR v1 single (H=512, eval=10000) ==="
	@./mlp-otto-score-ifc-xor.exe --model models/model-xor.otto --evalN 10000 2>&1 | grep -E 'Eval:|Time:'
	@echo ""
	@echo "=== Otto Score XNOR ensemble v6 (H=128x3, eval=10000) ==="
	@./mlp-otto-score-ifc-xnor.exe --model models/model-ensemble-xnor.otto --evalN 10000 2>&1 | grep -E 'Eval:|Time:'
	@echo ""
	@echo "=== Otto Score XOR ensemble v6 (H=128x3, eval=10000) ==="
	@./mlp-otto-score-ifc-xor.exe --model models/model-ensemble-xor.otto --evalN 10000 2>&1 | grep -E 'Eval:|Time:'
	@echo ""
	@echo "=== Float32 AdamW (H=512, same bit-mass, eval=10000) ==="
	@./mlp-flt32-ifc.exe --model models/flt32-w1-h512 --evalN 10000 2>&1 | grep -E 'Eval:|Time:'
	@echo ""
	@echo "=== Bin32 Hebbian (H=512, eval=10000) ==="
	@./mlp-bin32-ifc-xnor.exe --model models/hebbian-h512 --evalN 10000 2>&1 | grep -E 'Eval:|Time:'

# ── Float32 2-Layer AdamW Reference (matmul + LReLU, no bitwise) ──
#   trainer:   mlp-flt32-trn-w1-adam.exe
#   inference: mlp-flt32-ifc.exe
#   model:     models/flt32-w1-h512/ (auto-trained on first `make test`)

flt32: mlp-flt32-trn-w1-adam.exe mlp-flt32-ifc.exe

mlp-flt32-trn-w1-adam.exe: mlp-flt32-trn-w1-adam.c ki-common.h
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

mlp-flt32-ifc.exe: mlp-flt32-ifc.c ki-common.h
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

# ── Hebbian Reference (bitwise XNOR/XOR + MAJ3 + popcount, no convergence) ──
#   Trains W1 via batch-Hebbian on frozen W0.
#   Note: oscillates around 80-82% — does NOT converge like Otto Score or AdamW.
#   Included as negative reference and for comparison.

hebbian: mlp-bin32-trn-w1-hebbian-xnor.exe mlp-bin32-trn-w1-hebbian-xor.exe mlp-bin32-ifc-xnor.exe mlp-bin32-ifc-xor.exe

mlp-bin32-trn-w1-hebbian-xnor.exe: mlp-bin32-trn-w1-hebbian.c ki-common.h lib/w0_random.h lib/maj3.h
	$(CC) $(CFLAGS) -DPACKING=1 -o $@ $< $(LDLIBS)

mlp-bin32-trn-w1-hebbian-xor.exe: mlp-bin32-trn-w1-hebbian.c ki-common.h lib/w0_random.h lib/maj3.h
	$(CC) $(CFLAGS) -DPACKING=1 -DH0_XOR -o $@ $< $(LDLIBS)

# ── Bin32 Hebbian Inference (XNOR + popcnt, no float) ──

mlp-bin32-ifc-xnor.exe: mlp-bin32-ifc.c ki-common.h lib/maj3.h
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

mlp-bin32-ifc-xor.exe: mlp-bin32-ifc.c ki-common.h lib/maj3.h
	$(CC) $(CFLAGS) -DH0_XOR -o $@ $< $(LDLIBS)

# ── Otto Score Ensemble Trainer (iterative correction, Bayes log-score) ──
#   Builds model.otto files compatible with mlp-otto-score-ifc.
#   Example:
#     ./mlp-otto-score-ensemble-xnor.exe --hiddenN 128 --ensembleN 3 --epochsN 10 --out models/otto-h128-e3

ensemble: mlp-otto-score-ensemble-xnor.exe mlp-otto-score-ensemble-xor.exe

mlp-otto-score-ensemble-xnor.exe: mlp-otto-score-ensemble.c ki-otto-common.h lib/maj3.h
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

mlp-otto-score-ensemble-xor.exe: mlp-otto-score-ensemble.c ki-otto-common.h lib/maj3.h
	$(CC) $(CFLAGS) -DH0_XOR -o $@ $< $(LDLIBS)

# Auto-train float32 H=512 model if missing (~9s at 10 epochs, same bit-mass as Otto Score H=512)
models/flt32-w1-h512/weights.meta: mlp-flt32-trn-w1-adam.exe
	@echo "  Training float32 H=512 model (10 epochs)..."
	@mkdir -p models/flt32-w1-h512
	@./mlp-flt32-trn-w1-adam.exe --hiddenN 512 --epochsN 10 --out models/flt32-w1-h512 2>&1 | tail -1

# Auto-train Hebbian H=512 model if missing (~2s at 3 epochs)
models/hebbian-h512/weights.meta: mlp-bin32-trn-w1-hebbian-xnor.exe
	@echo "  Training Hebbian H=512 model (3 epochs)..."
	@mkdir -p models/hebbian-h512
	@./mlp-bin32-trn-w1-hebbian-xnor.exe --hiddenN 512 --epochsN 3 --out models/hebbian-h512 2>&1 | tail -1

clean:
	rm -f *.exe

push:
	@if ! git rev-parse --git-dir >/dev/null 2>&1; then \
		echo "[ERROR] Not a git repository. Run: git init && git add -A && git commit -m 'msg'"; \
		exit 1; \
	fi
	@if ! git remote -v | grep -q origin; then \
		echo "[ERROR] No remote 'origin' configured."; \
		echo "  Setup: git remote add origin https://<TOKEN>@github.com/aotto1968/forward-prop.git"; \
		echo "   or:   git remote add origin git@github.com:aotto1968/forward-prop.git"; \
		exit 1; \
	fi
	@if ! git diff --quiet HEAD 2>/dev/null || ! git diff --cached --quiet HEAD 2>/dev/null; then \
		echo "  Uncommitted changes. Run: git add -A && git commit -m '...'"; \
		exit 1; \
	fi
	@echo "Pushing to GitHub..."
	git push -u origin master
