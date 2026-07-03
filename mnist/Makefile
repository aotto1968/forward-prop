# Makefile — Otto Score MNIST (Trainer + Inference)
# ===================================================
# Self-contained MNIST demo for the public distribution.
#
# Targets:
#   make all        — trainer (XNOR + XOR)
#   make model-otto — train MNIST Otto Score model
#   make clean      — remove executables + models
#   make setup      — download MNIST dataset
# ===================================================

CC       = gcc
CFLAGS   = -O3 -march=native -fopenmp -Wall -Wextra -Werror \
           -Wconversion -Wsign-conversion -Wfloat-equal -Wundef -Wshadow -Wunused \
           -I../../lib
LDLIBS   = -lm -lz

MODEL_DIR = ../models

.PHONY: all trainer clean setup model-otto

all: trainer

# ── Dataset ────────────────────────────────────────────────
setup:
	@echo "Downloading MNIST dataset..."
	@bash ../fetch_mnist.sh

# ── Otto Score Trainer ─────────────────────────────────────
TRN_SRC = mlp-bin32-otto-trn.c

trainer: mnist-mlp-bin32-otto-trn-xnor.exe mnist-mlp-bin32-otto-trn-xor.exe

mnist-mlp-bin32-otto-trn-xnor.exe: $(TRN_SRC) ki-common.h
	$(CC) $(CFLAGS) \
	  -Wno-sign-conversion -Wno-unused-parameter -Wno-unused-function \
	  -o $@ $< $(LDLIBS)

mnist-mlp-bin32-otto-trn-xor.exe: $(TRN_SRC) ki-common.h
	$(CC) $(CFLAGS) \
	  -Wno-sign-conversion -Wno-unused-parameter -Wno-unused-function \
	  -DH0_XOR -o $@ $< $(LDLIBS)

# ── Trained Model (cached) ─────────────────────────────────
model-otto: mnist-mlp-bin32-otto-trn-xnor.exe
	@mkdir -p $(MODEL_DIR)/mnist-otto-h512
	@./mnist-mlp-bin32-otto-trn-xnor.exe --hiddenN 512 --epochsN 10 \
	  --encoding exp --out $(MODEL_DIR)/mnist-otto-h512 2>&1 | tail -1

# ── Clean ──────────────────────────────────────────────────
clean:
	rm -f *.exe
	rm -rf $(MODEL_DIR)/mnist-otto-h512
