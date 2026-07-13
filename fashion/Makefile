# Makefile — Otto Score + Hebbian + Adam Fashion-MNIST (Trainer + Inference)
# ========================================================================
# Self-contained Fashion-MNIST demo for the public distribution.
#
# Targets:
#   make all         — Otto + Hebbian trainer (XNOR + XOR)
#   make otto        — Otto Score trainer only
#   make hebbian     — Hebbian trainer only
#   make adam        — Float32 AdamW trainer
#   make model-otto  — train Fashion Otto Score model
#   make model-hebbian  — train Fashion Hebbian model
#   make model-adam  — train Fashion AdamW model
#   make clean       — remove executables
#   make setup       — download Fashion-MNIST dataset
# ========================================================================

CC       = gcc
CFLAGS   = -O3 -g -march=native -fopenmp -Wall -Wextra -Werror \
           -Wconversion -Wsign-conversion -Wfloat-equal -Wundef -Wshadow -Wunused \
           -I../lib
LDLIBS   = -lm -lz

MODEL_DIR = ../models

.PHONY: all otto hebbian adam ensemble vis-errors clean setup \
        model-otto model-hebbian model-adam test test-otto test-adam test-hebbian

all: otto hebbian adam ensemble vis-errors

# ── Dataset ────────────────────────────────────────────────
setup:
	@echo "Downloading Fashion-MNIST dataset..."
	@bash ../fetch_fashion.sh

# ═══════════════════════════════════════════════════════════
# Otto Score Trainer
# ═══════════════════════════════════════════════════════════
OTTO_SRC = mlp-bin32-otto-trn.c

otto: fashion-mlp-bin32-otto-trn-xnor.exe fashion-mlp-bin32-otto-trn-xor.exe

fashion-mlp-bin32-otto-trn-xnor.exe: $(OTTO_SRC) ki-common.h
	$(CC) $(CFLAGS) \
	  -Wno-sign-conversion -Wno-unused-parameter -Wno-unused-function -Wno-unused-variable \
	  -o $@ $< $(LDLIBS)

fashion-mlp-bin32-otto-trn-xor.exe: $(OTTO_SRC) ki-common.h
	$(CC) $(CFLAGS) \
	  -Wno-sign-conversion -Wno-unused-parameter -Wno-unused-function -Wno-unused-variable \
	  -DH0_XOR -o $@ $< $(LDLIBS)

model-otto: fashion-mlp-bin32-otto-trn-xnor.exe
	@mkdir -p $(MODEL_DIR)/fashion-otto-h512-e10
	@./fashion-mlp-bin32-otto-trn-xnor.exe --hiddenN 512 --epochsN 10 \
	  --encoding latest --export $(MODEL_DIR)/fashion-otto-h512-e10 2>&1 | grep '^REPORT'

# ═══════════════════════════════════════════════════════════
# Hebbian Trainer (unified: via symlink to mnist)
# ═══════════════════════════════════════════════════════════
HEBB_SRC = mlp-bin32-hebbian-trn.c

hebbian: fashion-mlp-bin32-hebbian-trn-xnor.exe fashion-mlp-bin32-hebbian-trn-xor.exe

fashion-mlp-bin32-hebbian-trn-xnor.exe: $(HEBB_SRC) ki-common.h
	$(CC) $(CFLAGS) \
	  -Wno-sign-conversion -Wno-unused-parameter \
	  -Wno-unused-function -Wno-unused-variable \
	  -o $@ $< $(LDLIBS)

fashion-mlp-bin32-hebbian-trn-xor.exe: $(HEBB_SRC) ki-common.h
	$(CC) $(CFLAGS) \
	  -Wno-sign-conversion -Wno-unused-parameter \
	  -Wno-unused-function -Wno-unused-variable \
	  -DH0_XOR -o $@ $< $(LDLIBS)

model-hebbian: fashion-mlp-bin32-hebbian-trn-xnor.exe
	@mkdir -p $(MODEL_DIR)/fashion-hebbian-h512-e10
	@./fashion-mlp-bin32-hebbian-trn-xnor.exe --hiddenN 512 --epochsN 10 \
	  --encoding latest --export $(MODEL_DIR)/fashion-hebbian-h512-e10 2>&1 | grep '^REPORT'

# ═══════════════════════════════════════════════════════════
# Float32 AdamW Trainer
# ═══════════════════════════════════════════════════════════
ADAM_SRC = mlp-flt32-adam-trn.c

adam: fashion-mlp-flt32-adam-trn.exe

fashion-mlp-flt32-adam-trn.exe: $(ADAM_SRC) ki-common.h ../lib/ki-adamw.h
	$(CC) $(CFLAGS) \
	  -Wno-sign-conversion -Wno-unused-parameter -Wno-unused-function -Wno-unused-variable \
	  -Wno-format-truncation -Wno-pointer-to-int-cast -o $@ $< $(LDLIBS)

model-adam: fashion-mlp-flt32-adam-trn.exe
	@mkdir -p $(MODEL_DIR)/fashion-adam-h512-e10
	@./fashion-mlp-flt32-adam-trn.exe --hiddenN 512 --epochsN 10 \
	  --export $(MODEL_DIR)/fashion-adam-h512-e10 2>&1 | grep '^REPORT'

# ═══════════════════════════════════════════════════════════
# merge-ensemble (public score archive merger)
# ═══════════════════════════════════════════════════════════
MERGE_SRC = merge-ensemble.c

ensemble: fashion-merge-ensemble.exe

fashion-merge-ensemble.exe: $(MERGE_SRC)
	$(CC) $(CFLAGS) -Wno-sign-conversion -Wno-unused-parameter -o $@ $< $(LDLIBS)

# ═══════════════════════════════════════════════════════════
# Visual Error Inspector
# ═══════════════════════════════════════════════════════════
VIS_SRC = mlp-otto-vis-errors.c

vis-errors: fashion-mlp-otto-vis-errors.exe

fashion-mlp-otto-vis-errors.exe: $(VIS_SRC)
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

# ═══════════════════════════════════════════════════════════
# Tests — reuse cached model from model-* if available
# ═══════════════════════════════════════════════════════════
test-otto: fashion-mlp-bin32-otto-trn-xnor.exe
	@test -d $(MODEL_DIR)/fashion-otto-h512-e10 || $(MAKE) model-otto
	@echo -e "\n=== Otto Score Fashion (H=512, 10 ep, latest) ==="
	@./fashion-mlp-bin32-otto-trn-xnor.exe \
	  --import $(MODEL_DIR)/fashion-otto-h512-e10 --evalN 10000 --encoding latest 2>&1 | grep '^REPORT'

test-hebbian: fashion-mlp-bin32-hebbian-trn-xnor.exe
	@test -d $(MODEL_DIR)/fashion-hebbian-h512-e10 || $(MAKE) model-hebbian
	@echo -e "\n=== Bin32 Hebbian Fashion (H=512, 10 ep, latest) ==="
	@./fashion-mlp-bin32-hebbian-trn-xnor.exe \
	  --import $(MODEL_DIR)/fashion-hebbian-h512-e10 --evalN 10000 --encoding latest 2>&1 | grep '^REPORT'

test-adam: fashion-mlp-flt32-adam-trn.exe
	@test -d $(MODEL_DIR)/fashion-adam-h512-e10 || $(MAKE) model-adam
	@echo -e "\n=== Float32 AdamW Fashion (H=512, 10 ep) ==="
	@./fashion-mlp-flt32-adam-trn.exe \
	  --import $(MODEL_DIR)/fashion-adam-h512-e10 --evalN 10000 2>&1 | grep '^REPORT'

test: test-otto test-adam test-hebbian

# ═══════════════════════════════════════════════════════════
# Clean
# ═══════════════════════════════════════════════════════════
clean:
	rm -f *.exe

clean-all: clean
	rm -rf $(MODEL_DIR)/fashion-*
