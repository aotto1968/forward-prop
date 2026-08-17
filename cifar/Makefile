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
# GLib + tprint — ki-common.h uses TPrint/tprint_* (needs <glib.h> and tprint.c).
# Resolved via pkg-config so it works on any system; tprint.c linked as a source.
TPRINT_SRC  = ../lib/tprint.c
GLIB_CFLAGS = $(shell pkg-config --cflags glib-2.0)
GLIB_LIBS   = $(shell pkg-config --libs glib-2.0)
CFLAGS   = -O3 -g -march=native -fopenmp -Wall -Wextra -Werror \
           -Wconversion -Wsign-conversion -Wfloat-equal -Wundef -Wshadow -Wunused \
           -I. -I../lib -I.. $(GLIB_CFLAGS)
LDLIBS   = -lm -lz $(TPRINT_SRC) $(GLIB_LIBS)

HDRS = $(wildcard *.h) $(wildcard ../lib/*.h)

MODEL_DIR = ../models

.PHONY: all clean setup model

all: otto hebbian adam bitvote
model: model-otto model-hebbian model-adam model-bitvote

# ── Dataset ────────────────────────────────────────────────
setup:
	@echo "Downloading CIFAR-10 dataset..."
	@mkdir -p data
	@bash ../fetch_cifar10.sh

# ═══════════════════════════════════════════════════════════
# Otto Score Trainer
# ═══════════════════════════════════════════════════════════
.PHONY: otto model-otto
# PRF trainer (seq-prof.c) — the FINAL architecture (N members × 1 thread,
# distributed simulation). seq.c is archived (A/B testing only, 2026-08-09).
# MODE_INT32: bit-exact integer counters/scores — the chip-faithful default.
OTTO_SRC = mlp-bin32-otto-trn-seq-prof.c
OTTO_MODE = -DMODE_INT32

otto: cifar-mlp-bin32-otto-trn-xnor.exe

cifar-mlp-bin32-otto-trn-xnor.exe: $(OTTO_SRC) $(HDRS)
	$(CC) $(CFLAGS) $(OTTO_MODE) \
	  -Wno-sign-conversion -Wno-unused-parameter -Wno-unused-variable \
	  -Wno-error=unused-but-set-variable -Wno-error=format-truncation \
	  -o $@ $< $(LDLIBS)

cifar-mlp-bin32-otto-trn-xor.exe: $(OTTO_SRC) $(HDRS)
	$(CC) $(CFLAGS) $(OTTO_MODE) \
	  -Wno-sign-conversion -Wno-unused-parameter -Wno-unused-variable \
	  -Wno-error=unused-but-set-variable -Wno-error=format-truncation \
	  -DH0_XOR -o $@ $< $(LDLIBS)

model-otto: cifar-mlp-bin32-otto-trn-xnor.exe
	@mkdir -p $(MODEL_DIR)/cifar-otto-h256-e5
	@echo -n $(@) ": "
	@./cifar-mlp-bin32-otto-trn-xnor.exe --hiddenN 256 --epochsN 5 --threadN 8 \
	  --encoding latest --export $(MODEL_DIR)/cifar-otto-h256-e5 2>&1 | grep '^REPORT'

# ═══════════════════════════════════════════════════════════
# Hebbian Trainer (unified: via symlink to mnist)
# ═══════════════════════════════════════════════════════════
.PHONY: hebbian model-hebbian
HEBB_SRC = mlp-bin32-hebbian-trn.c

hebbian: cifar-mlp-bin32-hebbian-trn-xnor.exe

cifar-mlp-bin32-hebbian-trn-xnor.exe: $(HEBB_SRC) $(HDRS)
	$(CC) $(CFLAGS) \
	  -Wno-sign-conversion -Wno-unused-parameter -Wno-unused-variable \
	  -o $@ $< $(LDLIBS)

cifar-mlp-bin32-hebbian-trn-xor.exe: $(HEBB_SRC) $(HDRS)
	$(CC) $(CFLAGS) \
	  -Wno-sign-conversion -Wno-unused-parameter -Wno-unused-variable \
	  -DH0_XOR -o $@ $< $(LDLIBS)


model-hebbian: cifar-mlp-bin32-hebbian-trn-xnor.exe
	@mkdir -p $(MODEL_DIR)/cifar-hebbian-h256-e5
	@echo -n $(@) ": "
	@./cifar-mlp-bin32-hebbian-trn-xnor.exe --hiddenN 256 --epochsN 5 \
	  --encoding latest --export $(MODEL_DIR)/cifar-hebbian-h256-e5 2>&1 | grep '^REPORT'

# ═══════════════════════════════════════════════════════════
# Float32 AdamW Trainer (unified: via symlink to mnist)
# ═══════════════════════════════════════════════════════════
.PHONY: adam model-adam
ADAM_SRC = mlp-flt32-adam-trn.c

adam: cifar-mlp-flt32-adam-trn.exe

cifar-mlp-flt32-adam-trn.exe: $(ADAM_SRC) $(HDRS)
	$(CC) $(CFLAGS) \
	  -Wno-sign-conversion -Wno-unused-parameter -Wno-unused-variable \
	  -Wno-format-truncation -Wno-pointer-to-int-cast -o $@ $< $(LDLIBS)

model-adam: cifar-mlp-flt32-adam-trn.exe
	@mkdir -p $(MODEL_DIR)/cifar-adam-h256-e5
	@echo -n $(@) ": "
	@./cifar-mlp-flt32-adam-trn.exe --hiddenN 256 --epochsN 5 \
	  --encoding latest --export $(MODEL_DIR)/cifar-adam-h256-e5 2>&1 | grep '^REPORT'

# ═══════════════════════════════════════════════════════════════
# Bit-Voting (identity W0, I=H width; via -DKI_BITVOTING on seq-prof.c)
# ═══════════════════════════════════════════════════════════════
.PHONY: bitvote model-bitvote
# Bit-Voting = Otto with identity W0 (gb = input) and I=H width,
# enabled via -DKI_BITVOTING on the shared seq-prof.c trainer.

bitvote: cifar-mlp-bin32-otto-trn-bitvoting.exe

cifar-mlp-bin32-otto-trn-bitvoting.exe: mlp-bin32-otto-trn-seq-prof.c $(HDRS)
	$(CC) $(CFLAGS) -Wno-sign-conversion -Wno-unused-parameter -Wno-unused-variable -Wno-error=unused-but-set-variable -Wno-error=format-truncation -DKI_BITVOTING -o $@ $< $(LDLIBS)

model-bitvote: cifar-mlp-bin32-otto-trn-bitvoting.exe
	@mkdir -p $(MODEL_DIR)/cifar-bitvoting-e5
	@echo -n $(@) ": "
	@./cifar-mlp-bin32-otto-trn-bitvoting.exe --epochsN 5 --threadN 8 \
	  --encoding latest --export $(MODEL_DIR)/cifar-bitvoting-e5 2>&1 | grep '^REPORT'

# ═══════════════════════════════════════════════════════════════
# Fast single tests (use cached model if available, train if missing)
# ═══════════════════════════════════════════════════════════════
.PHONY: test-otto test-adam test-hebbian test-bitvoting test

# ── CIFAR-10 ───────────────────────────────────────────────
test-otto: otto
	@test -f $(MODEL_DIR)/cifar-otto-h256-e5/model.otto || $(MAKE) model-otto
	@echo -n $(@) ": "
	@./cifar-mlp-bin32-otto-trn-xnor.exe --threadN 8 \
	  --import $(MODEL_DIR)/cifar-otto-h256-e5 --evalN 10000 --encoding latest 2>&1 | grep '^REPORT'

test-adam: adam
	@test -f $(MODEL_DIR)/cifar-adam-h256-e5/weights-0.meta || $(MAKE) model-adam
	@echo -n $(@) ": "
	@./cifar-mlp-flt32-adam-trn.exe \
	  --import $(MODEL_DIR)/cifar-adam-h256-e5 --evalN 10000 2>&1 | grep '^REPORT'

test-hebbian: hebbian
	@test -f $(MODEL_DIR)/cifar-hebbian-h256-e5/weights-0.meta || $(MAKE) model-hebbian
	@echo -n $(@) ": "
	@./cifar-mlp-bin32-hebbian-trn-xnor.exe \
	  --import $(MODEL_DIR)/cifar-hebbian-h256-e5 --evalN 10000 --encoding latest 2>&1 | grep '^REPORT'

test-bitvote: bitvote
	@test -f $(MODEL_DIR)/cifar-bitvoting-e5/model.otto || $(MAKE) model-bitvote
	@echo -n $(@) ": "
	@./cifar-mlp-bin32-otto-trn-bitvoting.exe --threadN 8 \
	  --import $(MODEL_DIR)/cifar-bitvoting-e5 --evalN 10000 --encoding latest 2>&1 | grep '^REPORT'

test: test-otto test-adam test-hebbian test-bitvote

# ═══════════════════════════════════════════════════════════
# Error Visualizer (CIFAR rgb via ki-local.h)
# ═══════════════════════════════════════════════════════════
.PHONY: vis-errors
VIS_SRC = mlp-otto-vis-errors.c

vis-errors: cifar-mlp-otto-vis-errors.exe

cifar-mlp-otto-vis-errors.exe: $(VIS_SRC) $(HDRS)
	$(CC) $(CFLAGS) \
	  -Wno-sign-conversion -Wno-unused-parameter \
	  -o $@ $< $(LDLIBS)

# ═══════════════════════════════════════════════════════════
# Ensemble Tool (merge — works with --export-merge-scores)
# ═══════════════════════════════════════════════════════════
.PHONY: merge-ensemble
MERGE_SRC = merge-ensemble.c

merge-ensemble: cifar-merge-ensemble.exe

cifar-merge-ensemble.exe: $(MERGE_SRC) $(HDRS)
	$(CC) $(CFLAGS) -Wno-sign-conversion -Wno-unused-parameter -o $@ $< $(LDLIBS)

# ── Clean ──────────────────────────────────────────────────
clean:
	rm -f *.exe

clean-all: clean
	rm -rf $(MODEL_DIR)/cifar-*

optimize-ensemble:
	@echo "  [SKIP] optimize-ensemble is a Python tool at bin/optimize-ensemble"
