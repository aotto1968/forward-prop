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
#   make model-hebbian  — train CIFAR Hebbian model
#   make model-adam  — train CIFAR AdamW model
#   make clean       — remove executables
#   make setup       — download CIFAR-10 dataset
# =====================================================================

CC       = gcc
CFLAGS   = -O3 -g -march=native -fopenmp -Wall -Wextra -Werror \
           -Wconversion -Wsign-conversion -Wfloat-equal -Wundef -Wshadow -Wunused \
           -I. -I../lib -I..
LDLIBS   = -lm -lz

MODEL_DIR = ../models

.PHONY: all otto hebbian adam ensemble vis-errors clean setup model-otto model-hebbian model-adam

all: otto hebbian adam ensemble vis-errors

# ── Dataset ────────────────────────────────────────────────
setup:
	@echo "Downloading CIFAR-10 dataset..."
	@mkdir -p data
	@bash ../fetch_cifar10.sh

# ═══════════════════════════════════════════════════════════
# Otto Score Trainer
# ═══════════════════════════════════════════════════════════
OTTO_SRC = mlp-bin32-otto-trn-seq.c

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


model-hebbian: cifar-mlp-bin32-hebbian-trn-xnor.exe
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
	  -Wno-format-truncation -Wno-pointer-to-int-cast -o $@ $< $(LDLIBS)

model-adam: cifar-mlp-flt32-adam-trn.exe
	@mkdir -p $(MODEL_DIR)/cifar-adam-h256-e5
	@./cifar-mlp-flt32-adam-trn.exe --hiddenN 256 --epochsN 5 \
	  --encoding latest --export $(MODEL_DIR)/cifar-adam-h256-e5 2>&1 | grep '^REPORT'

# ═══════════════════════════════════════════════════════════════
# Fast single tests (use cached model if available, train if missing)
# ═══════════════════════════════════════════════════════════════

.PHONY: test-otto test-adam test-hebbian test

# ── CIFAR-10 ───────────────────────────────────────────────
test-otto: otto
	@test -f $(MODEL_DIR)/cifar-otto-h256-e5/model.otto || $(MAKE) model-otto
	@echo -e "\n=== Otto Score CIFAR-10 (H=256, 5 ep, latest) ==="
	@./cifar-mlp-bin32-otto-trn-xnor.exe \
	  --import $(MODEL_DIR)/cifar-otto-h256-e5 --evalN 10000 --encoding latest 2>&1 | grep '^REPORT'

test-adam: adam
	@test -f $(MODEL_DIR)/cifar-adam-h256-e5/weights-0.meta || $(MAKE) model-adam
	@echo -e "\n=== Float32 AdamW CIFAR-10 (H=256, 5 ep) ==="
	@./cifar-mlp-flt32-adam-trn.exe \
	  --import $(MODEL_DIR)/cifar-adam-h256-e5 --evalN 10000 2>&1 | grep '^REPORT'

test-hebbian: hebbian
	@test -f $(MODEL_DIR)/cifar-hebbian-h256-e5/weights-0.meta || $(MAKE) model-hebbian
	@echo -e "\n=== Bin32 Hebbian CIFAR-10 (H=256, 5 ep, latest) ==="
	@./cifar-mlp-bin32-hebbian-trn-xnor.exe \
	  --import $(MODEL_DIR)/cifar-hebbian-h256-e5 --evalN 10000 --encoding latest 2>&1 | grep '^REPORT'

test: test-otto test-adam test-hebbian

# ═══════════════════════════════════════════════════════════
# Error Visualizer (CIFAR rgb via ki-local.h)
# ═══════════════════════════════════════════════════════════
VIS_SRC = mlp-otto-vis-errors.c

vis-errors: cifar-mlp-otto-vis-errors.exe

cifar-mlp-otto-vis-errors.exe: $(VIS_SRC) ki-common.h
	$(CC) $(CFLAGS) \
	  -Wno-sign-conversion -Wno-unused-parameter -Wno-unused-function \
	  -o $@ $< $(LDLIBS)

# ═══════════════════════════════════════════════════════════
# Ensemble Tool (merge — works with --export-merge-scores)
# ═══════════════════════════════════════════════════════════
MERGE_SRC = merge-ensemble.c

ensemble: cifar-merge-ensemble.exe

cifar-merge-ensemble.exe: $(MERGE_SRC)
	$(CC) $(CFLAGS) -Wno-sign-conversion -Wno-unused-parameter -Wno-unused-function -o $@ $< $(LDLIBS)

# ═══════════════════════════════════════════════════════════════
# Bit-Voting Baseline (experiment)
# ═══════════════════════════════════════════════════════════════

BITVOTE_SRC = mlp-bin32-otto-trn-bitvoting.c

bitvote: cifar-mlp-bin32-otto-trn-bitvoting.exe

cifar-mlp-bin32-otto-trn-bitvoting.exe: $(BITVOTE_SRC) $(HDRS)
	$(CC) $(CFLAGS) -Wno-sign-conversion -Wno-unused-parameter -Wno-unused-function -Wno-unused-variable -o $@ $< $(LDLIBS)

# ── Clean ──────────────────────────────────────────────────
clean:
	rm -f *.exe

clean-all: clean
	rm -rf $(MODEL_DIR)/cifar-*

optimize-ensemble:
	@echo "  [SKIP] optimize-ensemble is a Python tool at bin/optimize-ensemble"
