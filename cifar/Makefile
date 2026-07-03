# Makefile — Otto Score CIFAR-10 (Trainer + Inference)
# ======================================================
# Self-contained CIFAR-10 demo for the public distribution.
#
# Targets:
#   make all        — trainer (XNOR + XOR)
#   make model-otto — train CIFAR Otto Score model
#   make clean      — remove executables
#   make setup      — download CIFAR-10 dataset
# ======================================================

CC       = gcc
CFLAGS   = -O3 -march=native -fopenmp -Wall -Wextra -Werror \
           -Wconversion -Wsign-conversion -Wfloat-equal -Wundef -Wshadow -Wunused \
           -I../lib
LDLIBS   = -lm -lz

MODEL_DIR = ../models

.PHONY: all trainer clean setup model-otto

all: trainer

# ── Dataset ────────────────────────────────────────────────
setup:
	@echo "Downloading CIFAR-10 dataset..."
	@mkdir -p data
	@bash ../fetch_cifar10.sh

# ── Otto Score Trainer ─────────────────────────────────────
TRN_SRC = mlp-bin32-otto-trn.c

trainer: cifar-mlp-bin32-otto-trn-xnor.exe cifar-mlp-bin32-otto-trn-xor.exe

cifar-mlp-bin32-otto-trn-xnor.exe: $(TRN_SRC) ki-common.h
	$(CC) $(CFLAGS) \
	  -Wno-sign-conversion -Wno-unused-parameter -Wno-unused-function -Wno-unused-variable \
	  -o $@ $< $(LDLIBS)

cifar-mlp-bin32-otto-trn-xor.exe: $(TRN_SRC) ki-common.h
	$(CC) $(CFLAGS) \
	  -Wno-sign-conversion -Wno-unused-parameter -Wno-unused-function -Wno-unused-variable \
	  -DH0_XOR -o $@ $< $(LDLIBS)

# ── Trained Model (cached) ─────────────────────────────────
model-otto: cifar-mlp-bin32-otto-trn-xnor.exe
	@mkdir -p $(MODEL_DIR)/cifar-otto-h256-e10
	@echo -n "start..." && ./cifar-mlp-bin32-otto-trn-xnor.exe --hiddenN 256 --epochsN 10 \
	  --encoding latest --out $(MODEL_DIR)/cifar-otto-h256-e10 2>&1 | grep '^REPORT'

# ── Clean ──────────────────────────────────────────────────
clean:
	rm -f *.exe
	rm -rf $(MODEL_DIR)/cifar-otto-h256-e10
