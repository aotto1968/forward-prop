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
#   make model-hebb  — train MNIST Hebbian model
#   make model-adam  — train MNIST AdamW model
#   make clean       — remove executables + models
#   make setup       — download MNIST dataset
# ================================================================

CC       = gcc
CFLAGS   = -O3 -march=native -fopenmp -Wall -Wextra -Werror \
           -Wconversion -Wsign-conversion -Wfloat-equal -Wundef -Wshadow -Wunused \
           -I../../lib
LDLIBS   = -lm -lz

MODEL_DIR = ../models

.PHONY: all otto hebbian adam clean setup model-otto model-hebb model-adam

all: otto hebbian adam

# ── Dataset ────────────────────────────────────────────────
setup:
	@echo "Downloading MNIST dataset..."
	@bash ../fetch_mnist.sh

# ═══════════════════════════════════════════════════════════
# Otto Score Trainer
# ═══════════════════════════════════════════════════════════
OTTO_SRC = mlp-bin32-otto-trn.c

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
	  --encoding exp --export $(MODEL_DIR)/mnist-otto-h512-e10 2>&1 | tail -1

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

model-hebb: mnist-mlp-bin32-hebbian-trn-xnor.exe
	@mkdir -p $(MODEL_DIR)/mnist-hebbian-h512-e10
	@./mnist-mlp-bin32-hebbian-trn-xnor.exe --hiddenN 512 --epochsN 10 \
	  --encoding exp --export $(MODEL_DIR)/mnist-hebbian-h512-e10 2>&1 | tail -1

# ═══════════════════════════════════════════════════════════
# Float32 AdamW Trainer (unified: MNIST + CIFAR via ki-local.h)
# ═══════════════════════════════════════════════════════════
ADAM_SRC = mlp-flt32-adam-trn.c

adam: mnist-mlp-flt32-adam-trn.exe

mnist-mlp-flt32-adam-trn.exe: $(ADAM_SRC) ki-common.h ../lib/ki-adamw.h
	$(CC) $(CFLAGS) \
	  -Wno-sign-conversion -Wno-unused-parameter -Wno-unused-function -Wno-unused-variable \
	  -Wno-format-truncation -Wno-pointer-to-int-cast \
	  -o $@ $< $(LDLIBS)

model-adam: mnist-mlp-flt32-adam-trn.exe
	@mkdir -p $(MODEL_DIR)/mnist-adam-h512-e10
	@./mnist-mlp-flt32-adam-trn.exe --hiddenN 512 --epochsN 10 \
	  --export $(MODEL_DIR)/mnist-adam-h512-e10 2>&1 | grep '^REPORT'

# ── Clean ──────────────────────────────────────────────────
clean:
	rm -f *.exe
	rm -rf $(MODEL_DIR)/mnist-otto-h512-e10 $(MODEL_DIR)/mnist-hebbian-h512-e10
	rm -rf $(MODEL_DIR)/mnist-otto-h512 $(MODEL_DIR)/mnist-hebbian-h512
	rm -rf $(MODEL_DIR)/mnist-adam-h512-e10
