# Makefile — Reference Implementations (MNIST + CIFAR-10)
# ========================================================
# Builds float32 AdamW reference binaries for both datasets.
# Hebbian references are now unified in ../mnist/ and ../cifar/.
#
# Targets:
#   make all               — MNIST + CIFAR Adam
#   make model-adam-mnist  — train MNIST AdamW model
#   make model-adam-cifar  — train CIFAR AdamW model
#   make clean             — remove executables
# ========================================================

CC       = gcc
CFLAGS   = -O3 -march=native -fopenmp -I. -I../lib
CFLAGS_CIFAR = -O3 -march=native -fopenmp -Icifar-include -I. -I../lib
LDLIBS   = -lm -lz

.PHONY: all mnist cifar clean \
        model-adam-mnist model-adam-cifar

all: mnist cifar

# ═══════════════════════════════════════════════════════════════
# MNIST Adam
# ═══════════════════════════════════════════════════════════════
MNIST_ADAM_TRN = mnist-mlp-flt32-adam-trn.exe
MNIST_ADAM_IFC = mnist-mlp-flt32-adam-ifc.exe

mnist: $(MNIST_ADAM_TRN) $(MNIST_ADAM_IFC)

$(MNIST_ADAM_TRN): mnist-mlp-flt32-adam-trn.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)
$(MNIST_ADAM_IFC): mnist-mlp-flt32-adam-ifc.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

# ═══════════════════════════════════════════════════════════════
# CIFAR Adam
# ═══════════════════════════════════════════════════════════════
CIFAR_ADAM_TRN = cifar-mlp-flt32-w1-adam-trn.exe
CIFAR_ADAM_IFC = cifar-mlp-flt32-adam-ifc.exe

cifar: $(CIFAR_ADAM_TRN) $(CIFAR_ADAM_IFC)

$(CIFAR_ADAM_TRN): cifar-mlp-flt32-w1-adam-trn.c
	$(CC) $(CFLAGS_CIFAR) -o $@ $< $(LDLIBS)
$(CIFAR_ADAM_IFC): cifar-mlp-flt32-adam-ifc.c
	$(CC) $(CFLAGS_CIFAR) -o $@ $< $(LDLIBS)

# ═══════════════════════════════════════════════════════════════
# Trained Models (cached)
# ═══════════════════════════════════════════════════════════════
MODEL_DIR = ../models

model-adam-mnist: $(MNIST_ADAM_TRN)
	@mkdir -p $(MODEL_DIR)/mnist-adam-h512-e10
	@echo -n "start..." && ./$(MNIST_ADAM_TRN) --hiddenN 512 --epochsN 10 --out $(MODEL_DIR)/mnist-adam-h512-e10 2>&1 | grep '^REPORT'

model-adam-cifar: $(CIFAR_ADAM_TRN)
	@mkdir -p $(MODEL_DIR)/cifar-adam-h256-e5
	@echo -n "start..." && ./$(CIFAR_ADAM_TRN) --hiddenN 256 --epochsN 5 --evalN 10000 --out $(MODEL_DIR)/cifar-adam-h256-e5 2>&1 | grep '^REPORT'

# ═══════════════════════════════════════════════════════════════
# Clean
# ═══════════════════════════════════════════════════════════════
clean:
	rm -f *.exe
