# Makefile — Otto Score + Hebbian + Adam MNIST (Trainer + Inference)
# ================================================================
# Self-contained MNIST demo for the public distribution.
#
# Targets:
#   make all         — Otto + Hebbian + Adam (XNOR + XOR)
#   make otto        — Otto Score trainer only
#   make hebbian     — Hebbian trainer only
#   make adam        — Float32 AdamW trainer only
#   make model-otto  — train MNIST Otto Score model
#   make model-hebbian  — train MNIST Hebbian model
#   make model-adam  — train MNIST AdamW model
#   make clean       — remove executables + models
#   make setup       — download MNIST dataset
# ================================================================

CC       = gcc
CFLAGS   = -O3 -g -march=native -fopenmp -Wall -Wextra -Werror \
           -Wconversion -Wsign-conversion -Wfloat-equal -Wundef -Wshadow -Wunused \
           -I. -I../lib -I..
LDLIBS   = -lm -lz

MODEL_DIR = ../models

.PHONY: all otto hebbian adam vis-errors merge-ensemble clean setup model-otto model-hebbian model-adam

all: otto hebbian adam vis-errors merge-ensemble

# ── Dataset ────────────────────────────────────────────────
setup:
	@echo "Downloading MNIST dataset..."
	@mkdir -p data
	@bash ../fetch_mnist.sh

# ═══════════════════════════════════════════════════════════
# Otto Score Trainer
# ═══════════════════════════════════════════════════════════
OTTO_SRC = mlp-bin32-otto-trn-seq.c

otto: mnist-mlp-bin32-otto-trn-xnor.exe mnist-mlp-bin32-otto-trn-xor.exe

mnist-mlp-bin32-otto-trn-xnor.exe: $(OTTO_SRC) ki-common.h
	$(CC) $(CFLAGS) \
	  -Wno-sign-conversion -Wno-unused-parameter -Wno-unused-function -Wno-unused-variable \
	  -o $@ $< $(LDLIBS)

mnist-mlp-bin32-otto-trn-xor.exe: $(OTTO_SRC) ki-common.h
	$(CC) $(CFLAGS) \
	  -Wno-sign-conversion -Wno-unused-parameter -Wno-unused-function -Wno-unused-variable \
	  -DH0_XOR -o $@ $< $(LDLIBS)

model-otto: mnist-mlp-bin32-otto-trn-xnor.exe
	@mkdir -p $(MODEL_DIR)/mnist-otto-h512-e10
	@./mnist-mlp-bin32-otto-trn-xnor.exe --hiddenN 512 --epochsN 10 \
	  --encoding latest --export $(MODEL_DIR)/mnist-otto-h512-e10 2>&1 | grep '^REPORT'

# ═══════════════════════════════════════════════════════════
# Hebbian Trainer (unified: MNIST + CIFAR via ki-local.h)
# ═══════════════════════════════════════════════════════════
HEBB_SRC = mlp-bin32-hebbian-trn.c

hebbian: mnist-mlp-bin32-hebbian-trn-xnor.exe mnist-mlp-bin32-hebbian-trn-xor.exe

mnist-mlp-bin32-hebbian-trn-xnor.exe: $(HEBB_SRC) ki-common.h
	$(CC) $(CFLAGS) \
	  -Wno-sign-conversion -Wno-unused-parameter -Wno-unused-function -Wno-unused-variable \
	  -o $@ $< $(LDLIBS)

mnist-mlp-bin32-hebbian-trn-xor.exe: $(HEBB_SRC) ki-common.h
	$(CC) $(CFLAGS) \
	  -Wno-sign-conversion -Wno-unused-parameter -Wno-unused-function -Wno-unused-variable \
	  -DH0_XOR -o $@ $< $(LDLIBS)


model-hebbian: mnist-mlp-bin32-hebbian-trn-xnor.exe
	@mkdir -p $(MODEL_DIR)/mnist-hebbian-h512-e10
	@./mnist-mlp-bin32-hebbian-trn-xnor.exe --hiddenN 512 --epochsN 10 \
	  --encoding latest --export $(MODEL_DIR)/mnist-hebbian-h512-e10 2>&1 | grep '^REPORT'

# ═══════════════════════════════════════════════════════════
# Float32 AdamW Trainer (unified: MNIST + CIFAR via ki-local.h)
# ═══════════════════════════════════════════════════════════
ADAM_SRC = mlp-flt32-adam-trn.c

adam: mnist-mlp-flt32-adam-trn.exe

mnist-mlp-flt32-adam-trn.exe: $(ADAM_SRC) ki-common.h ../lib/ki-adamw.h
	$(CC) $(CFLAGS) \
	  -Wno-sign-conversion -Wno-unused-parameter -Wno-unused-function -Wno-unused-variable \
	  -Wno-format-truncation -Wno-pointer-to-int-cast -o $@ $< $(LDLIBS)

model-adam: mnist-mlp-flt32-adam-trn.exe
	@mkdir -p $(MODEL_DIR)/mnist-adam-h512-e10
	@./mnist-mlp-flt32-adam-trn.exe --hiddenN 512 --epochsN 10 \
	  --export $(MODEL_DIR)/mnist-adam-h512-e10 2>&1 | grep '^REPORT'

# ═══════════════════════════════════════════════════════════════
# Fast single tests (use cached model if available, train if missing)
# ═══════════════════════════════════════════════════════════════

.PHONY: test-otto test-adam test-hebbian test

# ── MNIST ──────────────────────────────────────────────────
test-otto: otto
	@test -f $(MODEL_DIR)/mnist-otto-h512-e10/model.otto || $(MAKE) model-otto
	@echo -e "\n=== Otto Score MNIST (H=512, 10 ep, latest) ==="
	@./mnist-mlp-bin32-otto-trn-xnor.exe \
	  --import $(MODEL_DIR)/mnist-otto-h512-e10 --evalN 10000 --encoding latest 2>&1 | grep '^REPORT'

test-adam: adam
	@test -f $(MODEL_DIR)/mnist-adam-h512-e10/weights-0.meta || $(MAKE) model-adam
	@echo -e "\n=== Float32 AdamW MNIST (H=512, 10 ep) ==="
	@./mnist-mlp-flt32-adam-trn.exe \
	  --import $(MODEL_DIR)/mnist-adam-h512-e10 --evalN 10000 2>&1 | grep '^REPORT'

test-hebbian: hebbian
	@test -f $(MODEL_DIR)/mnist-hebbian-h512-e10/weights-0.meta || $(MAKE) model-hebbian
	@echo -e "\n=== Bin32 Hebbian MNIST (H=512, 10 ep, latest) ==="
	@./mnist-mlp-bin32-hebbian-trn-xnor.exe \
	  --import $(MODEL_DIR)/mnist-hebbian-h512-e10 --evalN 10000 --encoding latest 2>&1 | grep '^REPORT'

test: test-otto test-adam test-hebbian

# ═══════════════════════════════════════════════════════════
# Error Visualizer (MNIST + CIFAR via ki-local.h)
# ═══════════════════════════════════════════════════════════
VIS_SRC = mlp-otto-vis-errors.c

vis-errors: mnist-mlp-otto-vis-errors.exe

mnist-mlp-otto-vis-errors.exe: $(VIS_SRC) ki-common.h
	$(CC) $(CFLAGS) \
	  -Wno-sign-conversion -Wno-unused-parameter -Wno-unused-function \
	  -o $@ $< $(LDLIBS)

# ═══════════════════════════════════════════════════════════════
# Merge-Ensemble — score archive merger
# ═══════════════════════════════════════════════════════════════
MERGE_SRC = merge-ensemble.c

merge-ensemble: mnist-merge-ensemble.exe

mnist-merge-ensemble.exe: $(MERGE_SRC)
	$(CC) $(CFLAGS) -Wno-sign-conversion -Wno-unused-parameter -Wno-unused-function -o $@ $< $(LDLIBS)

# ═══════════════════════════════════════════════════════════════
# Bit-Voting Baseline (experiment)
# ═══════════════════════════════════════════════════════════════

BITVOTE_SRC = mlp-bin32-otto-trn-bitvoting.c

bitvote: mnist-mlp-bin32-otto-trn-bitvoting.exe

mnist-mlp-bin32-otto-trn-bitvoting.exe: $(BITVOTE_SRC) $(HDRS)
	$(CC) $(CFLAGS) -Wno-sign-conversion -Wno-unused-parameter -Wno-unused-function -Wno-unused-variable -o $@ $< $(LDLIBS)

# ── Clean ──────────────────────────────────────────────────
clean:
	rm -f *.exe

clean-all: clean
	rm -rf $(MODEL_DIR)/mnist-*

optimize-ensemble:
	@echo "  [SKIP] optimize-ensemble is a Python tool at bin/optimize-ensemble"
