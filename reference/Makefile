# Makefile — Reference Implementations (MNIST + CIFAR-10)
# ========================================================
# Builds all reference binaries for the public distribution.
#
# Targets:
#   make all           — MNIST + CIFAR references
#   make mnist         — MNIST refs only (hebbian + adam)
#   make cifar         — CIFAR refs only (hebbian + adam)
#   make model-adam-mnist   — train MNIST AdamW model
#   make model-hebb-mnist   — train MNIST Hebbian model
#   make model-adam-cifar   — train CIFAR AdamW model
#   make model-hebb-cifar   — train CIFAR Hebbian model
#   make clean         — remove executables + models
# ========================================================

CC       = gcc
CFLAGS   = -O3 -march=native -fopenmp -I. -I../lib
CFLAGS_CIFAR = -O3 -march=native -fopenmp -Icifar-include -I. -I../lib
LDLIBS   = -lm -lz

.PHONY: all mnist cifar clean \
        model-adam-mnist model-hebb-mnist \
        model-adam-cifar model-hebb-cifar

all: mnist cifar

# ═══════════════════════════════════════════════════════════════
# MNIST References
# ═══════════════════════════════════════════════════════════════

MNIST_HEBB_TRN_XNOR = mnist-mlp-bin32-hebbian-trn-xnor.exe
MNIST_HEBB_TRN_XOR  = mnist-mlp-bin32-hebbian-trn-xor.exe
MNIST_HEBB_IFC_XNOR = mnist-mlp-bin32-hebbian-ifc-xnor.exe
MNIST_HEBB_IFC_XOR  = mnist-mlp-bin32-hebbian-ifc-xor.exe
MNIST_ADAM_TRN      = mnist-mlp-flt32-adam-trn.exe
MNIST_ADAM_IFC      = mnist-mlp-flt32-adam-ifc.exe

mnist: \
  $(MNIST_HEBB_TRN_XNOR) $(MNIST_HEBB_TRN_XOR) \
  $(MNIST_HEBB_IFC_XNOR) $(MNIST_HEBB_IFC_XOR) \
  $(MNIST_ADAM_TRN) $(MNIST_ADAM_IFC)

$(MNIST_HEBB_TRN_XNOR): mnist-mlp-bin32-hebbian-trn.c
	$(CC) $(CFLAGS) -DPACKING=1 -o $@ $< $(LDLIBS)
$(MNIST_HEBB_TRN_XOR): mnist-mlp-bin32-hebbian-trn.c
	$(CC) $(CFLAGS) -DPACKING=1 -DH0_XOR -o $@ $< $(LDLIBS)
$(MNIST_HEBB_IFC_XNOR): mnist-mlp-bin32-hebbian-ifc.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)
$(MNIST_HEBB_IFC_XOR): mnist-mlp-bin32-hebbian-ifc.c
	$(CC) $(CFLAGS) -DH0_XOR -o $@ $< $(LDLIBS)
$(MNIST_ADAM_TRN): mnist-mlp-flt32-adam-trn.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)
$(MNIST_ADAM_IFC): mnist-mlp-flt32-adam-ifc.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

# ═══════════════════════════════════════════════════════════════
# CIFAR References
# ═══════════════════════════════════════════════════════════════

CIFAR_HEBB_TRN_XNOR = cifar-mlp-bin32-w1-hebbian-trn-xnor.exe
CIFAR_HEBB_TRN_XOR  = cifar-mlp-bin32-w1-hebbian-trn-xor.exe
CIFAR_HEBB_IFC_XNOR = cifar-mlp-bin32-hebbian-ifc-xnor.exe
CIFAR_HEBB_IFC_XOR  = cifar-mlp-bin32-hebbian-ifc-xor.exe
CIFAR_ADAM_TRN      = cifar-mlp-flt32-w1-adam-trn.exe
CIFAR_ADAM_IFC      = cifar-mlp-flt32-adam-ifc.exe

cifar: \
  $(CIFAR_HEBB_TRN_XNOR) $(CIFAR_HEBB_TRN_XOR) \
  $(CIFAR_HEBB_IFC_XNOR) $(CIFAR_HEBB_IFC_XOR) \
  $(CIFAR_ADAM_TRN) $(CIFAR_ADAM_IFC)

$(CIFAR_HEBB_TRN_XNOR): cifar-mlp-bin32-w1-hebbian-trn.c
	$(CC) $(CFLAGS_CIFAR) -DPACKING=1 -o $@ $< $(LDLIBS)
$(CIFAR_HEBB_TRN_XOR): cifar-mlp-bin32-w1-hebbian-trn.c
	$(CC) $(CFLAGS_CIFAR) -DPACKING=1 -DH0_XOR -o $@ $< $(LDLIBS)
$(CIFAR_HEBB_IFC_XNOR): cifar-mlp-bin32-hebbian-ifc.c
	$(CC) $(CFLAGS_CIFAR) -o $@ $< $(LDLIBS)
$(CIFAR_HEBB_IFC_XOR): cifar-mlp-bin32-hebbian-ifc.c
	$(CC) $(CFLAGS_CIFAR) -DH0_XOR -o $@ $< $(LDLIBS)
$(CIFAR_ADAM_TRN): cifar-mlp-flt32-w1-adam-trn.c
	$(CC) $(CFLAGS_CIFAR) -o $@ $< $(LDLIBS)
$(CIFAR_ADAM_IFC): cifar-mlp-flt32-adam-ifc.c
	$(CC) $(CFLAGS_CIFAR) -o $@ $< $(LDLIBS)

# ═══════════════════════════════════════════════════════════════
# Trained Models (cached: only rebuild when trainer binary changes)
# ═══════════════════════════════════════════════════════════════
MODEL_DIR = ../models

model-adam-mnist: $(MNIST_ADAM_TRN)
	@mkdir -p $(MODEL_DIR)/mnist-adam-h512
	@./$(MNIST_ADAM_TRN) --hiddenN 512 --epochsN 10 --out $(MODEL_DIR)/mnist-adam-h512 2>&1 | tail -1

model-hebb-mnist: $(MNIST_HEBB_TRN_XNOR) $(MNIST_HEBB_IFC_XNOR)
	@mkdir -p $(MODEL_DIR)/mnist-hebbian-h512
	@./$(MNIST_HEBB_TRN_XNOR) --hiddenN 512 --epochsN 10 --out $(MODEL_DIR)/mnist-hebbian-h512 2>&1 | tail -1

model-adam-cifar: $(CIFAR_ADAM_TRN)
	@mkdir -p $(MODEL_DIR)/cifar-adam-h256
	@./$(CIFAR_ADAM_TRN) --hiddenN 256 --epochsN 3 --evalN 10000 --out $(MODEL_DIR)/cifar-adam-h256 2>&1 | tail -1

model-hebb-cifar: $(CIFAR_HEBB_TRN_XNOR)
	@mkdir -p $(MODEL_DIR)/cifar-hebbian-h256
	@./$(CIFAR_HEBB_TRN_XNOR) --hiddenN 256 --epochsN 3 --evalN 10000 --out $(MODEL_DIR)/cifar-hebbian-h256 2>&1 | tail -1

# ═══════════════════════════════════════════════════════════════
# Clean
# ═══════════════════════════════════════════════════════════════
clean:
	rm -f *.exe
