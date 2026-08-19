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
# GLib — ki-common.h embeds tprint.c (static funcs) which calls glib
# functions. Resolved via pkg-config so it works on any system.
GLIB_CFLAGS = $(shell pkg-config --cflags glib-2.0)
GLIB_LIBS   = $(shell pkg-config --libs glib-2.0)
CFLAGS   = -O3 -g -march=native -fopenmp -Wall -Wextra -Werror \
           -Wconversion -Wsign-conversion -Wfloat-equal -Wundef -Wshadow -Wunused \
           -I. -I../lib -I.. $(GLIB_CFLAGS)
LDLIBS   = -lm -lz $(GLIB_LIBS)

HDRS = $(wildcard *.h) $(wildcard ../lib/*.h)

MODEL_DIR = ../models

.PHONY: all clean setup
all: otto hebbian adam bitvote  

.PHONY: model 
model: model-otto model-hebbian model-adam model-bitvote

.PHONY: model-debug 
model-debug: model-otto-debug model-hebbian-debug model-adam-debug model-bitvote-debug

# ── Dataset ────────────────────────────────────────────────
setup:
	@echo "Downloading Fashion-MNIST dataset..."
	@bash ../fetch_fashion.sh

# ═══════════════════════════════════════════════════════════
# Otto Score Trainer
# ═══════════════════════════════════════════════════════════
.PHONY: otto model-otto model-otto-debug
# PRF trainer (seq-prof.c) — the FINAL architecture (N members × 1 thread,
# distributed simulation). The old sequential seq.c is archived.
# MODE_INT32: bit-exact integer counters/scores — the chip-faithful default
# (float32 counters reach 90.78 vs 91.10 on the 16-member ensemble, 2026-08-09).
OTTO_SRC = mlp-bin32-otto-trn-seq-prof.c
OTTO_MODE = -DMODE_INT32

# Ensemble member file — the 16-member subset selected by the merge beam
# search (91.10% on the m1t=105 corpus, `--beam 5 --filter @`).
# Trained with --member-file + --maj1-thresh 105 (the corpus' calibration)
# → ~90.8-91.1% (vs 86.7% for the single --encoding latest member).
# See: mnist-fashion/README.md, www/papers/mnist-fashion.html
FASHION_MEMBER = member-H196-E10-OT8-M1-105-INT32.out

otto: fashion-mlp-bin32-otto-trn-xnor.exe

fashion-mlp-bin32-otto-trn-xnor.exe: $(OTTO_SRC) $(HDRS)
	$(CC) $(CFLAGS) $(OTTO_MODE) \
	  -Wno-sign-conversion -Wno-unused-parameter -Wno-unused-variable \
	  -Wno-error=unused-but-set-variable -Wno-error=format-truncation \
	  -o $@ $< $(LDLIBS)

fashion-mlp-bin32-otto-trn-xor.exe: $(OTTO_SRC) $(HDRS)
	$(CC) $(CFLAGS) $(OTTO_MODE) \
	  -Wno-sign-conversion -Wno-unused-parameter -Wno-unused-variable \
	  -Wno-error=unused-but-set-variable -Wno-error=format-truncation \
	  -DH0_XOR -o $@ $< $(LDLIBS)

model-otto: fashion-mlp-bin32-otto-trn-xnor.exe $(FASHION_MEMBER)
	@mkdir -p $(MODEL_DIR)/fashion-otto-h196-e10
	@echo -n $(@) ": "
	@./fashion-mlp-bin32-otto-trn-xnor.exe --member-file $(FASHION_MEMBER) --export $(MODEL_DIR)/fashion-otto-h196-e10 --threadN 8 2>&1 | grep '^REPORT'

model-otto-debug: fashion-mlp-bin32-otto-trn-xnor.exe $(FASHION_MEMBER)
	@mkdir -p $(MODEL_DIR)/fashion-otto-h196-e10
	./fashion-mlp-bin32-otto-trn-xnor.exe --member-file $(FASHION_MEMBER) --export $(MODEL_DIR)/fashion-otto-h196-e10

# ═══════════════════════════════════════════════════════════
# Hebbian Trainer (unified: via symlink to mnist)
# ═══════════════════════════════════════════════════════════
.PHONY: hebbian model-hebbian model-hebbian-debug
HEBB_SRC = mlp-bin32-hebbian-trn.c

hebbian: fashion-mlp-bin32-hebbian-trn-xnor.exe fashion-mlp-bin32-hebbian-trn-xor.exe

fashion-mlp-bin32-hebbian-trn-xnor.exe: $(HEBB_SRC) $(HDRS)
	$(CC) $(CFLAGS) \
	  -Wno-sign-conversion -Wno-unused-parameter \
	  -Wno-unused-variable \
	  -o $@ $< $(LDLIBS)

fashion-mlp-bin32-hebbian-trn-xor.exe: $(HEBB_SRC) $(HDRS)
	$(CC) $(CFLAGS) \
	  -Wno-sign-conversion -Wno-unused-parameter \
	  -Wno-unused-variable \
	  -DH0_XOR -o $@ $< $(LDLIBS)

model-hebbian: fashion-mlp-bin32-hebbian-trn-xnor.exe
	@mkdir -p $(MODEL_DIR)/fashion-hebbian-h512-e10
	@echo -n $(@) ": "
	@./fashion-mlp-bin32-hebbian-trn-xnor.exe --hiddenN 512 --epochsN 10 \
	  --encoding latest --export $(MODEL_DIR)/fashion-hebbian-h512-e10 2>&1 | grep '^REPORT'

model-hebbian-debug: fashion-mlp-bin32-hebbian-trn-xnor.exe
	@mkdir -p $(MODEL_DIR)/fashion-hebbian-h512-e10
	./fashion-mlp-bin32-hebbian-trn-xnor.exe --hiddenN 512 --epochsN 10 \
	  --encoding latest --export $(MODEL_DIR)/fashion-hebbian-h512-e10

# ═══════════════════════════════════════════════════════════
# Float32 AdamW Trainer
# ═══════════════════════════════════════════════════════════
.PHONY: adam model-adam model-adam-debug
ADAM_SRC = mlp-flt32-adam-trn.c

adam: fashion-mlp-flt32-adam-trn.exe

fashion-mlp-flt32-adam-trn.exe: $(ADAM_SRC) $(HDRS)
	$(CC) $(CFLAGS) \
	  -Wno-sign-conversion -Wno-unused-parameter -Wno-unused-variable \
	  -Wno-format-truncation -Wno-pointer-to-int-cast -o $@ $< $(LDLIBS)

model-adam: fashion-mlp-flt32-adam-trn.exe
	@mkdir -p $(MODEL_DIR)/fashion-adam-h512-e10
	@echo -n $(@) ": "
	@./fashion-mlp-flt32-adam-trn.exe --hiddenN 512 --epochsN 10 \
	  --export $(MODEL_DIR)/fashion-adam-h512-e10 2>&1 | grep '^REPORT'

model-adam-debug: fashion-mlp-flt32-adam-trn.exe
	@mkdir -p $(MODEL_DIR)/fashion-adam-h512-e10
	@echo -n $(@) ": "
	./fashion-mlp-flt32-adam-trn.exe --hiddenN 512 --epochsN 10 \
	  --export $(MODEL_DIR)/fashion-adam-h512-e10

# ═══════════════════════════════════════════════════════════
# Bit-Voting (identity W0, I=H width; via -DKI_BITVOTING on seq-prof.c)
# ═══════════════════════════════════════════════════════════
.PHONY: bitvote model-bitvote model-bitvote-debug
# Bit-Voting = Otto with identity W0 (gb = input) and I=H width,
# enabled via -DKI_BITVOTING on the shared seq.c trainer.

BITVOTE_MEMBER = member-H196-E10-BV8-INT32.out

bitvote: fashion-mlp-bin32-otto-trn-bitvoting.exe

fashion-mlp-bin32-otto-trn-bitvoting.exe: mlp-bin32-otto-trn-seq-prof.c $(HDRS)
	$(CC) $(CFLAGS) -Wno-sign-conversion -Wno-unused-parameter -Wno-unused-variable -Wno-error=unused-but-set-variable -Wno-error=format-truncation -DKI_BITVOTING -DMODE_INT32 -o $@ $< $(LDLIBS)

model-bitvote: fashion-mlp-bin32-otto-trn-bitvoting.exe $(BITVOTE_MEMBER)
	@mkdir -p $(MODEL_DIR)/fashion-bitvoting-e10
	@echo -n $(@) ": "
	@./fashion-mlp-bin32-otto-trn-bitvoting.exe --member-file $(BITVOTE_MEMBER) --export $(MODEL_DIR)/fashion-bitvoting-e10 2>&1 | grep '^REPORT'

model-bitvote-debug: fashion-mlp-bin32-otto-trn-bitvoting.exe $(BITVOTE_MEMBER)
	@mkdir -p $(MODEL_DIR)/fashion-bitvoting-e10
	@echo -n $(@) ": "
	./fashion-mlp-bin32-otto-trn-bitvoting.exe --member-file $(BITVOTE_MEMBER) --export $(MODEL_DIR)/fashion-bitvoting-e10

# ═══════════════════════════════════════════════════════════
# merge-ensemble (public score archive merger)
# ═══════════════════════════════════════════════════════════
.PHONY: merge-ensemble
MERGE_SRC = merge-ensemble.c

merge-ensemble: fashion-merge-ensemble.exe

fashion-merge-ensemble.exe: $(MERGE_SRC) $(HDRS)
	$(CC) $(CFLAGS) -Wno-sign-conversion -Wno-unused-parameter -o $@ $< $(LDLIBS)

# ═══════════════════════════════════════════════════════════
# Visual Error Inspector
# ═══════════════════════════════════════════════════════════
.PHONY: vis-errors
VIS_SRC = mlp-otto-vis-errors.c

vis-errors: fashion-mlp-otto-vis-errors.exe

fashion-mlp-otto-vis-errors.exe: $(VIS_SRC) $(HDRS)
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

# ═══════════════════════════════════════════════════════════
# Tests — reuse cached model from model-* if available
# ═══════════════════════════════════════════════════════════
.PHONY: test test-otto test-adam test-hebbian test-bitvoting \
        test-debug test-otto-debug test-adam-debug test-hebbian-debug test-bitvoting-debug

test-otto: fashion-mlp-bin32-otto-trn-xnor.exe
	@test -f $(MODEL_DIR)/fashion-otto-h196-e10/model.otto || $(MAKE) model-otto
	@echo -n $(@) ": "
	@./fashion-mlp-bin32-otto-trn-xnor.exe --import $(MODEL_DIR)/fashion-otto-h196-e10 --evalN 10000 --threadN 8 2>&1 | grep '^REPORT'

test-hebbian: fashion-mlp-bin32-hebbian-trn-xnor.exe
	@test -f $(MODEL_DIR)/fashion-hebbian-h512-e10/weights-0.meta || $(MAKE) model-hebbian
	@echo -n $(@) ": "
	@./fashion-mlp-bin32-hebbian-trn-xnor.exe --import $(MODEL_DIR)/fashion-hebbian-h512-e10 --evalN 10000 --encoding latest 2>&1 | grep '^REPORT'

test-adam: fashion-mlp-flt32-adam-trn.exe
	@test -f $(MODEL_DIR)/fashion-adam-h512-e10/weights-0.meta || $(MAKE) model-adam
	@echo -n $(@) ": "
	@./fashion-mlp-flt32-adam-trn.exe --import $(MODEL_DIR)/fashion-adam-h512-e10 --evalN 10000 2>&1 | grep '^REPORT'

test-bitvoting: fashion-mlp-bin32-otto-trn-bitvoting.exe
	@test -f $(MODEL_DIR)/fashion-bitvoting-e10/model.otto || $(MAKE) model-bitvote
	@echo -n $(@) ": "
	@./fashion-mlp-bin32-otto-trn-bitvoting.exe --import $(MODEL_DIR)/fashion-bitvoting-e10 --evalN 10000 --threadN 8 2>&1 | grep '^REPORT'

test-otto-debug: fashion-mlp-bin32-otto-trn-xnor.exe
	@test -f $(MODEL_DIR)/fashion-otto-h196-e10/model.otto || $(MAKE) model-otto
	@echo -n $(@) ": "
	./fashion-mlp-bin32-otto-trn-xnor.exe --import $(MODEL_DIR)/fashion-otto-h196-e10 --evalN 10000 --threadN 8

test-hebbian-debug: fashion-mlp-bin32-hebbian-trn-xnor.exe
	@test -f $(MODEL_DIR)/fashion-hebbian-h512-e10/weights-0.meta || $(MAKE) model-hebbian
	@echo -n $(@) ": "
	./fashion-mlp-bin32-hebbian-trn-xnor.exe --import $(MODEL_DIR)/fashion-hebbian-h512-e10 --evalN 10000 --encoding latest

test-adam-debug: fashion-mlp-flt32-adam-trn.exe
	@test -f $(MODEL_DIR)/fashion-adam-h512-e10/weights-0.meta || $(MAKE) model-adam
	@echo -n $(@) ": "
	./fashion-mlp-flt32-adam-trn.exe --import $(MODEL_DIR)/fashion-adam-h512-e10 --evalN 10000 --threadN 8

test-bitvoting-debug: fashion-mlp-bin32-otto-trn-bitvoting.exe
	@test -f $(MODEL_DIR)/fashion-bitvoting-e10/model.otto || $(MAKE) model-bitvote
	@echo -n $(@) ": "
	./fashion-mlp-bin32-otto-trn-bitvoting.exe --import $(MODEL_DIR)/fashion-bitvoting-e10 --evalN 10000 --threadN 8

test: test-otto test-adam test-hebbian test-bitvoting
test-debug: test-otto-debug test-adam-debug test-hebbian-debug test-bitvoting-debug

# ═══════════════════════════════════════════════════════════
# Clean
# ═══════════════════════════════════════════════════════════
clean:
	rm -f *.exe

clean-all: clean
	rm -rf $(MODEL_DIR)/fashion-*
