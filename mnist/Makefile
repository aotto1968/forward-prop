# Makefile — Otto Score + Hebbian + Adam MNIST (Trainer + Inference)
# ================================================================
# Self-contained MNIST demo for the public distribution.
#
# Targets:
#   make all         — Otto + Hebbian + Adam (XNOR + XOR)
#   make otto        — Otto Score trainer only
#   make hebbian     — Hebbian trainer only
#   make adam        — Float32 AdamW trainer only
#   make model-otto  — train MNIST Otto Score model
#   make model-hebbian  — train MNIST Hebbian model
#   make model-adam  — train MNIST AdamW model
#   make clean       — remove executables + models
#   make setup       — download MNIST dataset
# ================================================================

CC       = ccache gcc-15
CFLAGS   = -O3 -g -march=native -fopenmp -Wall -Wextra -Werror \
           -Wconversion -Wsign-conversion -Wfloat-equal -Wundef -Wshadow -Wunused \
           -I. -I../lib -I..
LDLIBS   = -lm -lz

HDRS = $(wildcard *.h) $(wildcard ../lib/*.h)

# GLib + tprint — tprint.h includes <glib.h> for gboolean/gint types.
# Resolve include/library paths via pkg-config so builds work on any system
# (previously these vars were undefined -> glib.h not found / tprint.c not linked).
GLIB_CFLAGS = $(shell pkg-config --cflags glib-2.0)
GLIB_LIBS   = $(shell pkg-config --libs glib-2.0)

# carquet (Parquet-Writer für --sample-index) — merge-ensemble.c needs it.
# The ORIGINAL lives here (otto-score-ifc/3rdparty) so this tree is autonomous;
# ../3rdparty in the forward-prop dev dirs is a symlink to it.
CARQUET_ROOT   = $(abspath ../3rdparty/carquet-install)
CARQUET_CFLAGS = -I$(CARQUET_ROOT)/include
CARQUET_LIBS   = $(CARQUET_ROOT)/lib64/libcarquet.a \
                 $(CARQUET_ROOT)/lib64/liblz4.a -lzstd -lz -lm -lpthread -fopenmp

MODEL_DIR = ../models

.PHONY: all clean setup model

all: otto hebbian adam bitvote
model: model-otto model-hebbian model-adam model-bitvote

# ── Dataset ────────────────────────────────────────────────
setup:
	@echo "Downloading MNIST dataset..."
	@mkdir -p data
	@bash ../fetch_mnist.sh

# ═══════════════════════════════════════════════════════════
# Otto Score Trainer
# ═══════════════════════════════════════════════════════════
.PHONY: otto model-otto
# PRF trainer (seq-prof.c) — the FINAL architecture (N members × 1 thread,
# distributed simulation). seq.c is archived (A/B testing only, 2026-08-09).
# MODE_INT32: bit-exact integer counters/scores — the chip-faithful default.
OTTO_SRC = mlp-bin32-otto-trn-seq-prof.c
OTTO_MODE = -DMODE_INT32

otto: mnist-mlp-bin32-otto-trn-xnor.exe 

mnist-mlp-bin32-otto-trn-xnor.exe: $(OTTO_SRC) $(HDRS)
	$(CC) $(CFLAGS) $(OTTO_MODE) \
	  -Wno-sign-conversion -Wno-unused-parameter \
	  -o $@ $< $(GLIB_CFLAGS) $(LDLIBS) $(GLIB_LIBS)

mnist-mlp-bin32-otto-trn-xor.exe: $(OTTO_SRC) $(HDRS)
	$(CC) $(CFLAGS) $(OTTO_MODE) \
	  -Wno-sign-conversion -Wno-unused-parameter \
	  -DH0_XOR -o $@ $< $(GLIB_CFLAGS) $(LDLIBS) $(GLIB_LIBS)

model-otto: mnist-mlp-bin32-otto-trn-xnor.exe
	@mkdir -p $(MODEL_DIR)/mnist-otto-h512-e10
	@echo -n $(@) ": "
	@./mnist-mlp-bin32-otto-trn-xnor.exe --threadN 8 --member-file member-H196-E10-OT8-M1-105-INT32.out \
	  --export $(MODEL_DIR)/mnist-otto-h512-e10 2>&1 | grep '^REPORT'

model-otto-dbg: mnist-mlp-bin32-otto-trn-xnor.exe
	@mkdir -p $(MODEL_DIR)/mnist-otto-h512-e10
	./mnist-mlp-bin32-otto-trn-xnor.exe --threadN 8 --member-file member-H196-E10-OT8-M1-105-INT32.out \
	  --export $(MODEL_DIR)/mnist-otto-h512-e10

# ═══════════════════════════════════════════════════════════
# Hebbian Trainer (unified: MNIST + CIFAR via ki-local.h)
# ═══════════════════════════════════════════════════════════
.PHONY: hebbian model-hebbian
HEBB_SRC = mlp-bin32-hebbian-trn.c

hebbian: mnist-mlp-bin32-hebbian-trn-xnor.exe mnist-mlp-bin32-hebbian-trn-xor.exe

mnist-mlp-bin32-hebbian-trn-xnor.exe: $(HEBB_SRC) $(HDRS)
	$(CC) $(CFLAGS) \
	  -Wno-sign-conversion -Wno-unused-parameter -Wno-unused-variable \
	  -o $@ $< $(GLIB_CFLAGS) $(LDLIBS) $(GLIB_LIBS)

mnist-mlp-bin32-hebbian-trn-xor.exe: $(HEBB_SRC) $(HDRS)
	$(CC) $(CFLAGS) \
	  -Wno-sign-conversion -Wno-unused-parameter -Wno-unused-variable \
	  -DH0_XOR -o $@ $< $(GLIB_CFLAGS) $(LDLIBS) $(GLIB_LIBS)


model-hebbian: mnist-mlp-bin32-hebbian-trn-xnor.exe
	@mkdir -p $(MODEL_DIR)/mnist-hebbian-h512-e10
	@echo -n $(@) ": "
	@./mnist-mlp-bin32-hebbian-trn-xnor.exe --hiddenN 512 --epochsN 10 \
	  --encoding latest --export $(MODEL_DIR)/mnist-hebbian-h512-e10 2>&1 | grep '^REPORT'

# ═══════════════════════════════════════════════════════════
# Float32 AdamW Trainer (unified: MNIST + CIFAR via ki-local.h)
# ═══════════════════════════════════════════════════════════
.PHONY: adam model-adam
ADAM_SRC = mlp-flt32-adam-trn.c

adam: mnist-mlp-flt32-adam-trn.exe

mnist-mlp-flt32-adam-trn.exe: $(ADAM_SRC) $(HDRS)
	$(CC) $(CFLAGS) \
	  -Wno-sign-conversion -Wno-unused-parameter -Wno-unused-variable \
	  -Wno-format-truncation -Wno-pointer-to-int-cast -o $@ $< $(GLIB_CFLAGS) $(LDLIBS) $(GLIB_LIBS)

model-adam: mnist-mlp-flt32-adam-trn.exe
	@mkdir -p $(MODEL_DIR)/mnist-adam-h512-e10
	@echo -n $(@) ": "
	@./mnist-mlp-flt32-adam-trn.exe --hiddenN 512 --epochsN 10 \
	  --export $(MODEL_DIR)/mnist-adam-h512-e10 2>&1 | grep '^REPORT'

# ═══════════════════════════════════════════════════════════════
# Bit-Voting (identity W0, I=H width; via -DKI_BITVOTING on seq-prof.c)
# ═══════════════════════════════════════════════════════════════
.PHONY: bitvote model-bitvote
# Bit-Voting = Otto with identity W0 (gb = input) and I=H width,
# enabled via -DKI_BITVOTING on the shared seq-prof.c trainer.
# See: plans/plan-2026-08-02-ki-bitvoting-flag.md

bitvote: mnist-mlp-bin32-otto-trn-bitvoting.exe

mnist-mlp-bin32-otto-trn-bitvoting.exe: mlp-bin32-otto-trn-seq-prof.c $(HDRS)
	$(CC) $(CFLAGS) -Wno-sign-conversion -Wno-unused-parameter -Wno-unused-variable -Wno-error=unused-but-set-variable -Wno-error=format-truncation -DKI_BITVOTING -DMODE_INT32 -o $@ $< $(GLIB_CFLAGS) $(LDLIBS) $(GLIB_LIBS)

model-bitvote: mnist-mlp-bin32-otto-trn-bitvoting.exe
	@mkdir -p $(MODEL_DIR)/mnist-bitvoting-e10
	@echo -n $(@) ": "
	@./mnist-mlp-bin32-otto-trn-bitvoting.exe --threadN 8 --member-file member-H196-E10-BV8-INT32.out \
	  --export $(MODEL_DIR)/mnist-bitvoting-e10 2>&1 | grep '^REPORT'

# ═══════════════════════════════════════════════════════════════
# Fast single tests (use cached model if available, train if missing)
# ═══════════════════════════════════════════════════════════════

.PHONY: test-otto test-adam test-hebbian test-bitvoting test

# ── MNIST ──────────────────────────────────────────────────
test-otto: otto
	@test -f $(MODEL_DIR)/mnist-otto-h512-e10/model.otto || $(MAKE) model-otto
	@echo -n $(@) ": "
	@./mnist-mlp-bin32-otto-trn-xnor.exe \
	  --import $(MODEL_DIR)/mnist-otto-h512-e10 --evalN 10000 --encoding latest --threadN 8 2>&1 | grep '^REPORT'

test-adam: adam
	@test -f $(MODEL_DIR)/mnist-adam-h512-e10/weights-0.meta || $(MAKE) model-adam
	@echo -n $(@) ": "
	@./mnist-mlp-flt32-adam-trn.exe \
	  --import $(MODEL_DIR)/mnist-adam-h512-e10 --evalN 10000 2>&1 | grep '^REPORT'

test-hebbian: hebbian
	@test -f $(MODEL_DIR)/mnist-hebbian-h512-e10/weights-0.meta || $(MAKE) model-hebbian
	@echo -n $(@) ": "
	@./mnist-mlp-bin32-hebbian-trn-xnor.exe \
	  --import $(MODEL_DIR)/mnist-hebbian-h512-e10 --evalN 10000 --encoding latest 2>&1 | grep '^REPORT'

test-bitvoting: bitvote
	@test -f $(MODEL_DIR)/mnist-bitvoting-e10/model.otto || $(MAKE) model-bitvote
	@echo -n $(@) ": "
	@./mnist-mlp-bin32-otto-trn-bitvoting.exe \
	  --import $(MODEL_DIR)/mnist-bitvoting-e10 --evalN 10000 --encoding latest --threadN 8 2>&1 | grep '^REPORT'

test: test-otto test-adam test-hebbian test-bitvoting

# ═══════════════════════════════════════════════════════════
# Error Visualizer (MNIST + CIFAR via ki-local.h)
# ═══════════════════════════════════════════════════════════
.PHONY: vis-errors
VIS_SRC = mlp-otto-vis-errors.c

vis-errors: mnist-mlp-otto-vis-errors.exe

mnist-mlp-otto-vis-errors.exe: $(VIS_SRC) $(HDRS)
	$(CC) $(CFLAGS) \
	  -Wno-sign-conversion -Wno-unused-parameter \
	  -o $@ $< $(GLIB_CFLAGS) $(LDLIBS) $(GLIB_LIBS)

# ═══════════════════════════════════════════════════════════════
# Merge-Ensemble — score archive merger
# ═══════════════════════════════════════════════════════════════
.PHONY: merge-ensemble
MERGE_SRC = merge-ensemble.c

merge-ensemble: mnist-merge-ensemble.exe

mnist-merge-ensemble.exe: $(MERGE_SRC) $(HDRS)
	$(CC) $(CFLAGS) -Wno-sign-conversion -Wno-unused-parameter -o $@ $< $(GLIB_CFLAGS) $(LDLIBS) $(GLIB_LIBS) $(CARQUET_LIBS) $(CARQUET_CFLAGS)

# ── Clean ──────────────────────────────────────────────────
clean:
	rm -f *.exe

clean-all: clean
	rm -rf $(MODEL_DIR)/mnist-*

optimize-ensemble:
	@echo "  [SKIP] optimize-ensemble is a Python tool at bin/optimize-ensemble"
