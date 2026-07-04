# Makefile — Otto Score + Reference Implementations (public demo)
# ==============================================================
# All build work is delegated to subdirectory Makefiles.
# This Makefile only orchestrates top-level targets.
# ==============================================================

.PHONY: all otto adam hebbian setup clean \
        test test-mnist test-cifar \
        test-mnist-otto test-mnist-adam test-mnist-hebbian \
        test-cifar-otto test-cifar-adam test-cifar-hebbian \
        setup-mnist setup-cifar

# ═══════════════════════════════════════════════════════════════
# Build
# ═══════════════════════════════════════════════════════════════
all:
	$(MAKE) -s -C mnist all
	$(MAKE) -s -C cifar all
	$(MAKE) -s -C reference all

otto:     mnist cifar
mnist:    ; $(MAKE) -s -C mnist all
cifar:    ; $(MAKE) -s -C cifar all
adam:     ; $(MAKE) -s -C reference mnist cifar
hebbian:  ; $(MAKE) -s -C reference mnist cifar

# ═══════════════════════════════════════════════════════════════
# Dataset setup
# ═══════════════════════════════════════════════════════════════
setup-mnist: ; @bash fetch_mnist.sh
setup-cifar: ; @bash fetch_cifar10.sh
setup: setup-mnist setup-cifar

# ═══════════════════════════════════════════════════════════════
# Trained model cache (train if missing, then fast eval)
# ═══════════════════════════════════════════════════════════════
model-otto-mnist:   ; $(MAKE) -s -C mnist     model-otto
model-otto-cifar:   ; $(MAKE) -s -C cifar     model-otto
model-adam-mnist:   ; $(MAKE) -s -C reference model-adam-mnist
model-hebb-mnist:   ; $(MAKE) -s -C mnist     model-hebb
model-adam-cifar:   ; $(MAKE) -s -C reference model-adam-cifar
model-hebb-cifar:   ; $(MAKE) -s -C cifar     model-hebb

models: model-otto-mnist model-otto-cifar model-adam-mnist model-hebb-mnist model-adam-cifar model-hebb-cifar

.PHONY: models model-otto-mnist model-otto-cifar model-adam-mnist model-hebb-mnist model-adam-cifar model-hebb-cifar

# ═══════════════════════════════════════════════════════════════
# Test — baut nur fehlende Modelle, dann eval via IFC
# ═══════════════════════════════════════════════════════════════
test: all
	$(MAKE) -s test-mnist
	$(MAKE) -s test-cifar

# ═══════════════════════════════════════════════════════════════
# Fast single tests (use cached model if available, train if missing)
# ═══════════════════════════════════════════════════════════════

# ── MNIST ──────────────────────────────────────────────────
test-mnist-otto: mnist
	@test -f models/mnist-otto-h512-e10/model.otto || $(MAKE) model-otto-mnist
	@echo -e "\n=== Otto Score MNIST (H=512, 10 ep, exp8) ==="
	@echo -n "start..." && ./mnist/mnist-mlp-bin32-otto-trn-xnor.exe \
	  --model models/mnist-otto-h512-e10/model.otto --evalN 10000 --encoding exp 2>&1 | grep '^REPORT'

test-mnist-adam: reference
	@test -f models/mnist-adam-h512-e10/weights.meta || $(MAKE) model-adam-mnist
	@echo -e "\n=== Float32 AdamW MNIST (H=512, 10 ep) ==="
	@echo -n "start..." && ./reference/mnist-mlp-flt32-adam-ifc.exe \
	  --model models/mnist-adam-h512-e10 --evalN 10000 2>&1 | grep '^REPORT'

test-mnist-hebbian: mnist
	@test -f models/mnist-hebbian-h512-e10/weights-0.meta || $(MAKE) model-hebb-mnist
	@echo -e "\n=== Bin32 Hebbian MNIST (H=512, 10 ep) ==="
	@echo -n "start..." && ./mnist/mnist-mlp-bin32-hebbian-trn-xnor.exe \
	  --model models/mnist-hebbian-h512-e10 --evalN 10000 --encoding exp 2>&1 | grep '^REPORT'

test-mnist: test-mnist-otto test-mnist-adam test-mnist-hebbian

# ── CIFAR-10 ───────────────────────────────────────────────
test-cifar-otto: cifar
	@test -f models/cifar-otto-h256-e5/model.otto || $(MAKE) model-otto-cifar
	@echo -e "\n=== Otto Score CIFAR-10 (H=256, 5 ep) ==="
	@./cifar/cifar-mlp-bin32-otto-trn-xnor.exe \
	  --model models/cifar-otto-h256-e5/model.otto --evalN 10000 --encoding latest 2>&1 | grep '^REPORT'

test-cifar-adam: reference
	@test -f models/cifar-adam-h256-e5/weights.meta || $(MAKE) model-adam-cifar
	@echo -e "\n=== Float32 AdamW CIFAR-10 (H=256, 5 ep) ==="
	@./reference/cifar-mlp-flt32-adam-ifc.exe \
	  --model models/cifar-adam-h256-e5 --evalN 10000 2>&1 | grep '^REPORT'

test-cifar-hebbian: cifar
	@test -f models/cifar-hebbian-h256-e5/weights-0.meta || $(MAKE) model-hebb-cifar
	@echo -e "\n=== Bin32 Hebbian CIFAR-10 (H=256, 5 ep, latest) ==="
	@./cifar/cifar-mlp-bin32-hebbian-trn-xnor.exe \
	  --model models/cifar-hebbian-h256-e5 --evalN 10000 --encoding latest 2>&1 | grep '^REPORT'

test-cifar: test-cifar-otto test-cifar-adam test-cifar-hebbian

# ═══════════════════════════════════════════════════════════════
# Clean
# ═══════════════════════════════════════════════════════════════
clean:
	$(MAKE) -C mnist clean
	$(MAKE) -C cifar clean
	$(MAKE) -C reference clean

clean-all: clean
	rm -rf models/mnist-* models/cifar-*

# ── Push ──────────────────────────────────────────────────────
push:
	@git push -u origin master
