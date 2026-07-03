# Makefile — Otto Score MNIST (Trainer + Inference)
# ===================================================
# Self-contained MNIST demo for the Otto Score public distribution.
#
# Targets:
#   make all        — trainer + inference (XNOR + XOR)
#   make trainer    — mlp-otto-score-trn-xnor.exe + xor
#   make ifc        — mlp-otto-score-ifc-xnor.exe + xor
#   make clean      — remove executables
#   make setup      — download MNIST dataset
# ===================================================

CC       = gcc
CFLAGS   = -O3 -march=native -fopenmp -Wall -Wextra -Werror \
           -Wconversion -Wsign-conversion -Wfloat-equal -Wundef -Wshadow -Wunused \
           -I../../lib
LDLIBS   = -lm -lz

.PHONY: all trainer ifc clean setup

all: trainer

# ── Dataset ────────────────────────────────────────────────
setup:
	@echo "Downloading MNIST dataset..."
	@bash ../fetch_mnist.sh

# ── Otto Score Trainer ─────────────────────────────────────
TRN_SRC = mlp-otto-score-trn.c

trainer: mnist-mlp-otto-score-trn-xnor.exe mnist-mlp-otto-score-trn-xor.exe

mnist-mlp-otto-score-trn-xnor.exe: $(TRN_SRC) ki-common.h
	$(CC) $(CFLAGS) \
	  -Wno-sign-conversion -Wno-unused-parameter -Wno-unused-function \
	  -o $@ $< $(LDLIBS)

mnist-mlp-otto-score-trn-xor.exe: $(TRN_SRC) ki-common.h
	$(CC) $(CFLAGS) \
	  -Wno-sign-conversion -Wno-unused-parameter -Wno-unused-function \
	  -DH0_XOR -o $@ $< $(LDLIBS)

# ── Inference via trainer's --model flag ────────────────────
ifc:
	@echo "  Inference via: mnist-mlp-otto-score-trn-xnor.exe --model <file> --encoding exp"

# ── Clean ──────────────────────────────────────────────────
clean:
	rm -f mnist-mlp-otto-score-trn-xnor.exe mnist-mlp-otto-score-trn-xor.exe
