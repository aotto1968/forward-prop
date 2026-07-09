# Makefile — Otto Score + Hebbian + Adam CIFAR-10 (Trainer + Inference)
# =====================================================================
# Self-contained CIFAR-10 demo for the public distribution.
#
# Targets:
#   make all         — Otto + Hebbian trainer (XNOR + XOR)
#   make otto        — Otto Score trainer only
#   make hebbian     — Hebbian trainer only
#   make adam        — Float32 AdamW trainer
#   make model-otto  — train CIFAR Otto Score model
#   make model-hebb  — train CIFAR Hebbian model
#   make model-adam  — train CIFAR AdamW model
#   make clean       — remove executables
#   make setup       — download CIFAR-10 dataset
# =====================================================================

CC       = gcc
CFLAGS   = -O3 -march=native -fopenmp -Wall -Wextra -Werror \
           -Wconversion -Wsign-conversion -Wfloat-equal -Wundef -Wshadow -Wunused \
           -I../lib
LDLIBS   = -lm -lz

MODEL_DIR = ../models

.PHONY: all otto hebbian adam ensemble vis-errors clean setup model-otto model-hebb model-adam

all: otto hebbian adam ensemble vis-errors

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
	  --encoding latest --export $(MODEL_DIR)/cifar-otto-h256-e5 2>&1 | grep '^REPORT'

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
	  --encoding latest --export $(MODEL_DIR)/cifar-hebbian-h256-e5 2>&1 | grep '^REPORT'

# ═══════════════════════════════════════════════════════════
# Float32 AdamW Trainer (unified: via symlink to mnist)
# ═══════════════════════════════════════════════════════════
ADAM_SRC = mlp-flt32-adam-trn.c

adam: cifar-mlp-flt32-adam-trn.exe

cifar-mlp-flt32-adam-trn.exe: $(ADAM_SRC) ki-common.h ../lib/ki-adamw.h
	$(CC) $(CFLAGS) \
	  -Wno-sign-conversion -Wno-unused-parameter -Wno-unused-function -Wno-unused-variable \
	  -Wno-format-truncation -o $@ $< $(LDLIBS)

model-adam: cifar-mlp-flt32-adam-trn.exe
	@mkdir -p $(MODEL_DIR)/cifar-adam-h256-e5
	@./cifar-mlp-flt32-adam-trn.exe --hiddenN 256 --epochsN 5 \
	  --encoding latest --export $(MODEL_DIR)/cifar-adam-h256-e5 2>&1 | grep '^REPORT'

# ═══════════════════════════════════════════════════════════
# Ensemble Tool (merge — works with --save-scores)
# ═══════════════════════════════════════════════════════════
ENS_SRC_M = merge-ensemble.c

ensemble: cifar-merge-ensemble.exe

cifar-merge-ensemble.exe: $(ENS_SRC_M)
	$(CC) $(CFLAGS) -Wno-sign-conversion -Wno-unused-parameter -o $@ $< $(LDLIBS)

# ═══════════════════════════════════════════════════════════
# Error Visualizer (CIFAR rgb via ki-local.h)
# ═══════════════════════════════════════════════════════════
VIS_SRC = mlp-otto-vis-errors.c

vis-errors: cifar-mlp-otto-vis-errors.exe

cifar-mlp-otto-vis-errors.exe: $(VIS_SRC) ki-common.h
	$(CC) $(CFLAGS) \
	  -Wno-sign-conversion -Wno-unused-parameter -Wno-unused-function \
	  -o $@ $< $(LDLIBS)

# ── Clean ──────────────────────────────────────────────────
clean:
	rm -f *.exe
	rm -rf $(MODEL_DIR)/cifar-otto-h256-e5 $(MODEL_DIR)/cifar-hebbian-h256-e5 $(MODEL_DIR)/cifar-adam-h256-e5
