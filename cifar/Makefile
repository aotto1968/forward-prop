# Makefile — Otto Score CIFAR-10 (Trainer + Inference)
# ======================================================
# Self-contained CIFAR-10 demo for the public distribution.
#
# Targets:
#   make all        — trainer + inference (XNOR + XOR)
#   make trainer    — cifar-mlp-bin32-otto-trn-xnor.exe + xor
#   make ifc        — cifar-mlp-bin32-otto-ifc-xnor.exe + xor
#   make clean      — remove executables
#   make setup      — download CIFAR-10 dataset
# ======================================================

CC       = gcc
CFLAGS   = -O3 -march=native -fopenmp -Wall -Wextra -Werror \
           -Wconversion -Wsign-conversion -Wfloat-equal -Wundef -Wshadow -Wunused \
           -I../lib
LDLIBS   = -lm -lz

.PHONY: all trainer ifc clean setup

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
	  -Wno-sign-conversion -Wno-unused-parameter -Wno-unused-function \
	  -o $@ $< $(LDLIBS)

cifar-mlp-bin32-otto-trn-xor.exe: $(TRN_SRC) ki-common.h
	$(CC) $(CFLAGS) \
	  -Wno-sign-conversion -Wno-unused-parameter -Wno-unused-function \
	  -DH0_XOR -o $@ $< $(LDLIBS)

# ── Inference via trainer's --model flag ────────────────────
ifc:
	@echo "  Inference via: cifar-mlp-bin32-otto-trn-xnor.exe --model <file>"

# ── Clean ──────────────────────────────────────────────────
clean:
	rm -f *.exe
