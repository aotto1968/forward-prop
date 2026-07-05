# Makefile — Otto Score + Hebbian CIFAR-10 (Trainer + Inference)
# ==============================================================
# Self-contained CIFAR-10 demo for the public distribution.
#
# Targets:
#   make all         — Otto + Hebbian trainer (XNOR + XOR)
#   make otto        — Otto Score trainer only
#   make hebbian     — Hebbian trainer only
#   make model-otto  — train CIFAR Otto Score model
#   make model-hebb  — train CIFAR Hebbian model
#   make clean       — remove executables
#   make setup       — download CIFAR-10 dataset
# ==============================================================

CC       = gcc
CFLAGS   = -O3 -march=native -fopenmp -Wall -Wextra -Werror \
           -Wconversion -Wsign-conversion -Wfloat-equal -Wundef -Wshadow -Wunused \
           -I../lib
LDLIBS   = -lm -lz

MODEL_DIR = ../models

.PHONY: all otto hebbian clean setup model-otto model-hebb

all: otto hebbian

# ── Dataset ────────────────────────────────────────────────
setup:
	@echo "Downloading CIFAR-10 dataset..."
	@mkdir -p data
	@bash ../fetch_cifar10.sh

# ═══════════════════════════════════════════════════════════
# Otto Score Trainer
# ═══════════════════════════════════════════════════════════
OTTO_SRC = mlp-bin32-otto-trn.c

otto: cifar-mlp-bin32-otto-trn-xnor.exe cifar-mlp-bin32-otto-trn-xor.exe

cifar-mlp-bin32-otto-trn-xnor.exe: $(OTTO_SRC) ki-common.h
	$(CC) $(CFLAGS) \
	  -Wno-sign-conversion -Wno-unused-parameter -Wno-unused-function -Wno-unused-variable \
	  -o $@ $< $(LDLIBS)

cifar-mlp-bin32-otto-trn-xor.exe: $(OTTO_SRC) ki-common.h
	$(CC) $(CFLAGS) \
	  -Wno-sign-conversion -Wno-unused-parameter -Wno-unused-function -Wno-unused-variable \
	  -DH0_XOR -o $@ $< $(LDLIBS)

model-otto: cifar-mlp-bin32-otto-trn-xnor.exe
	@mkdir -p $(MODEL_DIR)/cifar-otto-h256-e5
	@./cifar-mlp-bin32-otto-trn-xnor.exe --hiddenN 256 --epochsN 5 \
	  --encoding latest --export $(MODEL_DIR)/cifar-otto-h256-e5 2>&1 | tail -1

# ═══════════════════════════════════════════════════════════
# Hebbian Trainer (unified: via symlink to mnist)
# ═══════════════════════════════════════════════════════════
HEBB_SRC = mlp-bin32-hebbian-trn.c

hebbian: cifar-mlp-bin32-hebbian-trn-xnor.exe cifar-mlp-bin32-hebbian-trn-xor.exe

cifar-mlp-bin32-hebbian-trn-xnor.exe: $(HEBB_SRC) ki-common.h
	$(CC) $(CFLAGS) \
	  -Wno-sign-conversion -Wno-unused-parameter -Wno-unused-function -Wno-unused-variable \
	  -o $@ $< $(LDLIBS)

cifar-mlp-bin32-hebbian-trn-xor.exe: $(HEBB_SRC) ki-common.h
	$(CC) $(CFLAGS) \
	  -Wno-sign-conversion -Wno-unused-parameter -Wno-unused-function -Wno-unused-variable \
	  -DH0_XOR -o $@ $< $(LDLIBS)

model-hebb: cifar-mlp-bin32-hebbian-trn-xnor.exe
	@mkdir -p $(MODEL_DIR)/cifar-hebbian-h256-e5
	@./cifar-mlp-bin32-hebbian-trn-xnor.exe --hiddenN 256 --epochsN 5 \
	  --encoding latest --export $(MODEL_DIR)/cifar-hebbian-h256-e5 2>&1 | tail -1

# ── Clean ──────────────────────────────────────────────────
clean:
	rm -f *.exe
	rm -rf $(MODEL_DIR)/cifar-otto-h256-e5 $(MODEL_DIR)/cifar-hebbian-h256-e5
