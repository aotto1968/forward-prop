# Makefile — Otto Score Inference (public demo)
# ===============================================
# Self-contained build for the public GitHub release.
# Builds both XNOR and XOR inference executables.
#
# Targets:
#   make (all)  — builds both
#   make setup  — download MNIST dataset (required once)
#   make xnor   — mlp-otto-score-ifc-xnor.exe (default: ~(in^W0))
#   make xor    — mlp-otto-score-ifc-xor.exe  (in^W0)
#   make clean  — remove executables
#   make test   — quick accuracy test with bundled model
#   make push   — push to GitHub (requires configured remote)

CC       = gcc
CFLAGS   = -O3 -march=native -fopenmp -Wall -Wextra -Werror \
           -Wconversion -Wsign-conversion -Wfloat-equal -Wundef -Wshadow -Wunused \
           -Ilib
LDLIBS   = -lm -lz

SRC      = mlp-otto-score-ifc.c

.PHONY: all xnor xor clean test test-image setup push

all: xnor xor

setup:
	@echo "Downloading MNIST dataset..."
	@bash fetch_mnist.sh

test-image: mlp-otto-score-ifc-xnor.exe
	@echo "=== Single image classification test ==="
	@echo "  tests/digit5.pgm (MNIST train[0], label=5)..."
	@./mlp-otto-score-ifc-xnor.exe --model models/model-xnor.otto --image tests/digit5.pgm 2>&1 | grep 'Predicted'
	@echo "  tests/digit9.pgm (MNIST train[4], label=9)..."
	@./mlp-otto-score-ifc-xnor.exe --model models/model-xnor.otto --image tests/digit9.pgm 2>&1 | grep 'Predicted'
	@echo "  tests/shoe.pgm (Fashion-MNIST ankle boot — should NOT be a digit)"
	@./mlp-otto-score-ifc-xnor.exe --model models/model-xnor.otto --image tests/shoe.pgm 2>&1 | grep 'Predicted'

xnor: mlp-otto-score-ifc-xnor.exe
xor:  mlp-otto-score-ifc-xor.exe

mlp-otto-score-ifc-xnor.exe: $(SRC) ki-common.h lib/maj3.h
	$(CC) $(CFLAGS) -o $@ $(SRC) $(LDLIBS)

mlp-otto-score-ifc-xor.exe: $(SRC) ki-common.h lib/maj3.h
	$(CC) $(CFLAGS) -DH0_XOR -o $@ $(SRC) $(LDLIBS)

test: mlp-otto-score-ifc-xnor.exe mlp-otto-score-ifc-xor.exe
	@echo "=== XNOR inference with XNOR-trained model ==="
	@./mlp-otto-score-ifc-xnor.exe --model models/model-xnor.otto --evalN 2000 2>&1 | grep -E 'Model:|Eval:'
	@echo ""
	@echo "=== XOR inference with XOR-trained model ==="
	@./mlp-otto-score-ifc-xor.exe --model models/model-xor.otto --evalN 2000 2>&1 | grep -E 'Model:|Eval:'

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
