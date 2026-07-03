# Makefile - Otto Score + Reference Implementations (public demo)
# ==============================================================
CC       = gcc
CFLAGS   = -O3 -march=native -fopenmp -Wall -Wextra -Werror \
           -Wconversion -Wsign-conversion -Wfloat-equal -Wundef -Wshadow -Wunused \
           -I. -Ilib -I..
REF_CFLAGS = -O3 -march=native -fopenmp -Ireference -Ilib
LDLIBS   = -lm -lz

.PHONY: all otto mnist cifar flt32 hebbian setup clean test test-image push

all: otto flt32 hebbian

otto: mnist cifar
mnist: ; $(MAKE) -C mnist all
cifar: ; $(MAKE) -C cifar all
setup-mnist: ; @bash fetch_mnist.sh
setup-cifar: ; @bash fetch_cifar10.sh
setup: setup-mnist setup-cifar

# ── Models (cached: only rebuild when trainer binary changes) ──
OTTO_MODEL  = models/mnist-otto-h512/model.otto
ADAM_MODEL  = models/mnist-adam-h512/weights.meta
HEBB_MODEL  = models/mnist-hebbian-h512/weights.meta
CIFAR_MODEL = models/cifar-otto-h256/model.otto
CIFAR_ADAM  = models/cifar-adam-h256/weights.meta
CIFAR_HEBB  = models/cifar-hebbian-h256/weights.meta

$(OTTO_MODEL): mnist/mnist-mlp-bin32-otto-trn-xnor.exe
	@mkdir -p models/mnist-otto-h512
	@cd mnist && ./mnist-mlp-bin32-otto-trn-xnor.exe --hiddenN 512 --epochsN 10 --encoding exp --out ../models/mnist-otto-h512 2>&1 | tail -1

$(ADAM_MODEL): reference/mnist-mlp-flt32-adam-trn.exe reference/mnist-mlp-flt32-adam-ifc.c.exe
	@mkdir -p models/mnist-adam-h512
	@./reference/mnist-mlp-flt32-adam-trn.exe --hiddenN 512 --epochsN 10 --out models/mnist-adam-h512 2>&1 | tail -1

$(HEBB_MODEL): reference/mnist-mlp-bin32-hebbian-trn-xnor.exe reference/mnist-mlp-bin32-hebbian-ifc-xnor.exe
	@mkdir -p models/mnist-hebbian-h512
	@./reference/mnist-mlp-bin32-hebbian-trn-xnor.exe --hiddenN 512 --epochsN 10 --out models/mnist-hebbian-h512 2>&1 | tail -1

$(CIFAR_MODEL): cifar/cifar-mlp-bin32-otto-trn-xnor.exe
	@mkdir -p models/cifar-otto-h256
	@cd cifar && ./cifar-mlp-bin32-otto-trn-xnor.exe --hiddenN 256 --epochsN 5 --out ../models/cifar-otto-h256 2>&1 | tail -1

$(CIFAR_ADAM): reference/cifar-mlp-flt32-w1-adam-trn.exe
	@mkdir -p models/cifar-adam-h256
	@./reference/cifar-mlp-flt32-w1-adam-trn.exe --hiddenN 256 --epochsN 3 --evalN 10000 --out models/cifar-adam-h256 2>&1 | tail -1

$(CIFAR_HEBB): reference/cifar-mlp-bin32-w1-hebbian-trn-xnor.exe
	@mkdir -p models/cifar-hebbian-h256
	@./reference/cifar-mlp-bin32-w1-hebbian-trn-xnor.exe --hiddenN 256 --epochsN 3 --evalN 10000 --out models/cifar-hebbian-h256 2>&1 | tail -1

# ── Test (evaluates cached models, fast on repeat) ─────────────
test: cifar mnist $(OTTO_MODEL) $(ADAM_MODEL) $(HEBB_MODEL) $(CIFAR_MODEL) $(CIFAR_ADAM) $(CIFAR_HEBB)
	@echo "=== Otto Score MNIST (H=512, 10 ep, exp8) ==="
	@echo -n "start..." && ./mnist/mnist-mlp-bin32-otto-trn-xnor.exe --model $(OTTO_MODEL) --evalN 10000 --encoding exp 2>&1 | grep '^REPORT'
	@echo "=== Float32 AdamW MNIST (H=512, 10 ep) ==="
	@echo -n "start..." && ./reference/mnist-mlp-flt32-adam-ifc.c.exe --model models/mnist-adam-h512 --evalN 10000 2>&1 | grep '^REPORT'
	@echo "=== Bin32 Hebbian MNIST (H=512, 10 ep) ==="
	@echo -n "start..." && ./reference/mnist-mlp-bin32-hebbian-ifc-xnor.exe --model models/mnist-hebbian-h512 --evalN 10000 2>&1 | grep '^REPORT'
	@echo "=== Otto Score CIFAR-10 (H=256, 7 ep, latest) ==="
	@echo -n "start..." && cd cifar && ./cifar-mlp-bin32-otto-trn-xnor.exe --hiddenN 256 --epochsN 7 --encoding latest --evalN 10000 2>&1 | grep '^REPORT'
	@echo "=== Float32 AdamW CIFAR-10 (H=256, 3 ep) ==="
	@echo -n "start..." && ./reference/cifar-mlp-flt32-w1-adam-trn.exe --hiddenN 256 --epochsN 3 --evalN 10000 2>&1 | grep '^REPORT'
	@echo "=== Bin32 Hebbian CIFAR-10 (H=256, 3 ep) ==="
	@echo -n "start..." && ./reference/cifar-mlp-bin32-w1-hebbian-trn-xnor.exe --hiddenN 256 --epochsN 3 --evalN 10000 2>&1 | grep 'Best eval'

# ── Single Image Test ─────────────────────────────────────────
test-image: mnist $(OTTO_MODEL)
	@./mnist/mlp-bin32-otto-ifc-xnor.exe --model $(OTTO_MODEL) --image tests/digit5.pgm 2>&1 | grep 'Predicted'
	@./mnist/mlp-bin32-otto-ifc-xnor.exe --model $(OTTO_MODEL) --image tests/digit9.pgm 2>&1 | grep 'Predicted'
	@./mnist/mlp-bin32-otto-ifc-xnor.exe --model $(OTTO_MODEL) --image tests/shoe.pgm 2>&1 | grep 'Predicted'

# ── Hebbian Reference (bitwise, no convergence) ───────────────
hebbian: \
  reference/mnist-mlp-bin32-hebbian-trn-xnor.exe \
  reference/mnist-mlp-bin32-hebbian-trn-xor.exe \
  reference/mnist-mlp-bin32-hebbian-ifc-xnor.exe \
  reference/mnist-mlp-bin32-hebbian-ifc-xor.exe \
  reference/cifar-mlp-bin32-w1-hebbian-trn-xnor.exe \
  reference/cifar-mlp-bin32-w1-hebbian-trn-xor.exe \
  reference/cifar-mlp-bin32-hebbian-ifc-xnor.exe \
  reference/cifar-mlp-bin32-hebbian-ifc-xor.exe
reference/mnist-mlp-bin32-hebbian-trn-xnor.exe: reference/mnist-mlp-bin32-hebbian-trn.c
	$(CC) $(REF_CFLAGS) -DPACKING=1 -o $@ $< $(LDLIBS)
reference/mnist-mlp-bin32-hebbian-trn-xor.exe: reference/mnist-mlp-bin32-hebbian-trn.c
	$(CC) $(REF_CFLAGS) -DPACKING=1 -DH0_XOR -o $@ $< $(LDLIBS)
reference/mnist-mlp-bin32-hebbian-ifc-xnor.exe: reference/mnist-mlp-bin32-hebbian-ifc.c
	$(CC) $(REF_CFLAGS) -o $@ $< $(LDLIBS)
reference/mnist-mlp-bin32-hebbian-ifc-xor.exe: reference/mnist-mlp-bin32-hebbian-ifc.c
	$(CC) $(REF_CFLAGS) -DH0_XOR -o $@ $< $(LDLIBS)
reference/cifar-mlp-bin32-w1-hebbian-trn-xnor.exe: reference/cifar-mlp-bin32-w1-hebbian-trn.c
	$(CC) -Ireference/cifar-include $(REF_CFLAGS) -DPACKING=1 -o $@ $< $(LDLIBS)
reference/cifar-mlp-bin32-w1-hebbian-trn-xor.exe: reference/cifar-mlp-bin32-w1-hebbian-trn.c
	$(CC) -Ireference/cifar-include $(REF_CFLAGS) -DPACKING=1 -DH0_XOR -o $@ $< $(LDLIBS)
reference/cifar-mlp-bin32-hebbian-ifc-xnor.exe: reference/cifar-mlp-bin32-hebbian-ifc.c
	$(CC) -Ireference/cifar-include $(REF_CFLAGS) -o $@ $< $(LDLIBS)
reference/cifar-mlp-bin32-hebbian-ifc-xor.exe: reference/cifar-mlp-bin32-hebbian-ifc.c
	$(CC) -Ireference/cifar-include $(REF_CFLAGS) -DH0_XOR -o $@ $< $(LDLIBS)

# ── Float32 AdamW Reference ───────────────────────────────────
flt32: \
  reference/mnist-mlp-flt32-adam-trn.exe \
  reference/mnist-mlp-flt32-adam-ifc.c.exe \
  reference/cifar-mlp-flt32-w1-adam-trn.exe
reference/mnist-mlp-flt32-adam-trn.exe: reference/mnist-mlp-flt32-adam-trn.c
	$(CC) $(REF_CFLAGS) -o $@ $< $(LDLIBS)
reference/mnist-mlp-flt32-adam-ifc.c.exe: reference/mnist-mlp-flt32-adam-ifc.c
	$(CC) $(REF_CFLAGS) -o $@ $< $(LDLIBS)
reference/cifar-mlp-flt32-w1-adam-trn.exe: reference/cifar-mlp-flt32-w1-adam-trn.c
	$(CC) -Ireference/cifar-include $(REF_CFLAGS) -o $@ $< $(LDLIBS)

# ── Clean ─────────────────────────────────────────────────────
clean:
	$(MAKE) -C mnist clean; $(MAKE) -C cifar clean
	rm -f **/*.exe
	rm -rf models/mnist-* models/cifar-*

# ── Push ──────────────────────────────────────────────────────
push:
	@git push -u origin master
