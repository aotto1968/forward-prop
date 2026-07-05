# Makefile — Otto Score + Adam + Hebbian (public demo)
# =====================================================
# All build work is delegated to subdirectory Makefiles.
# This Makefile only orchestrates top-level targets.
#
# Programs:
#   mnist/  — Otto Score, Hebbian, Adam for MNIST
#   cifar/  — Otto Score, Hebbian, Adam for CIFAR (symlinks to mnist/)
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

mnist:    ; $(MAKE) -s -C mnist all
cifar:    ; $(MAKE) -s -C cifar all
otto:     mnist cifar
hebbian:  mnist cifar
adam:     mnist cifar

# ═══════════════════════════════════════════════════════════════
# Dataset setup
# ═══════════════════════════════════════════════════════════════
setup-mnist: ; @bash fetch_mnist.sh
setup-cifar: ; @bash fetch_cifar10.sh
setup: setup-mnist setup-cifar

# ═══════════════════════════════════════════════════════════════
# Trained model cache (train if missing, then fast eval)
# ═══════════════════════════════════════════════════════════════
model-otto-mnist:   ; $(MAKE) -s -C mnist model-otto
model-otto-cifar:   ; $(MAKE) -s -C cifar model-otto
model-adam-mnist:
	$(MAKE) -s -C mnist adam
	@mkdir -p models/mnist-adam-h512-e10
	@./mnist/mnist-mlp-flt32-adam-trn.exe --hiddenN 512 --epochsN 10 \
	  --export models/mnist-adam-h512-e10 2>&1 | grep '^REPORT'
model-adam-cifar:
	$(MAKE) -s -C cifar model-adam
model-hebb-mnist:   ; $(MAKE) -s -C mnist model-hebb
model-hebb-cifar:   ; $(MAKE) -s -C cifar model-hebb

models: model-otto-mnist model-otto-cifar model-adam-mnist model-hebb-mnist \
        model-adam-cifar model-hebb-cifar

.PHONY: models model-otto-mnist model-otto-cifar model-adam-mnist \
        model-hebb-mnist model-adam-cifar model-hebb-cifar

# ═══════════════════════════════════════════════════════════════
# Test — baut nur fehlende Modelle, dann eval via --import
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
	@./mnist/mnist-mlp-bin32-otto-trn-xnor.exe \
	  --import models/mnist-otto-h512-e10 --evalN 10000 --encoding exp 2>&1 | grep '^REPORT'

test-mnist-adam:
	@test -f models/mnist-adam-h512-e10/weights-0.meta || $(MAKE) model-adam-mnist
	@echo -e "\n=== Float32 AdamW MNIST (H=512, 10 ep) ==="
	@./mnist/mnist-mlp-flt32-adam-trn.exe \
	  --import models/mnist-adam-h512-e10 --evalN 10000 2>&1 | grep '^REPORT'

test-mnist-hebbian: mnist
	@test -f models/mnist-hebbian-h512-e10/weights-0.meta || $(MAKE) model-hebb-mnist
	@echo -e "\n=== Bin32 Hebbian MNIST (H=512, 10 ep) ==="
	@./mnist/mnist-mlp-bin32-hebbian-trn-xnor.exe \
	  --import models/mnist-hebbian-h512-e10 --evalN 10000 --encoding exp 2>&1 | grep '^REPORT'

test-mnist: test-mnist-otto test-mnist-adam test-mnist-hebbian

# ── CIFAR-10 ───────────────────────────────────────────────
test-cifar-otto: cifar
	@test -f models/cifar-otto-h256-e5/model.otto || $(MAKE) model-otto-cifar
	@echo -e "\n=== Otto Score CIFAR-10 (H=256, 5 ep) ==="
	@./cifar/cifar-mlp-bin32-otto-trn-xnor.exe \
	  --import models/cifar-otto-h256-e5 --evalN 10000 --encoding latest 2>&1 | grep '^REPORT'

test-cifar-adam:
	@test -f models/cifar-adam-h256-e5/weights-0.meta || $(MAKE) model-adam-cifar
	@echo -e "\n=== Float32 AdamW CIFAR-10 (H=256, 5 ep) ==="
	@./cifar/cifar-mlp-flt32-adam-trn.exe \
	  --import models/cifar-adam-h256-e5 --evalN 10000 2>&1 | grep '^REPORT'

test-cifar-hebbian: cifar
	@test -f models/cifar-hebbian-h256-e5/weights-0.meta || $(MAKE) model-hebb-cifar
	@echo -e "\n=== Bin32 Hebbian CIFAR-10 (H=256, 5 ep, latest) ==="
	@./cifar/cifar-mlp-bin32-hebbian-trn-xnor.exe \
	  --import models/cifar-hebbian-h256-e5 --evalN 10000 --encoding latest 2>&1 | grep '^REPORT'

test-cifar: test-cifar-otto test-cifar-adam test-cifar-hebbian

# ═══════════════════════════════════════════════════════════════
# Clean
# ═══════════════════════════════════════════════════════════════
clean:
	$(MAKE) -C mnist clean
	$(MAKE) -C cifar clean

clean-all: clean
	rm -rf models/mnist-* models/cifar-*

# ── Push ──────────────────────────────────────────────────────
push:
	@git push -u origin master
