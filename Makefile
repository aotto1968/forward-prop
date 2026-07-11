# Makefile — Otto Score + Adam + Hebbian (public demo)
# =====================================================
# All build work is delegated to subdirectory Makefiles.
# This Makefile only orchestrates top-level targets.
#
# Programs:
#   mnist/  — Otto Score, Hebbian, Adam for MNIST
#   cifar/  — Otto Score, Hebbian, Adam for CIFAR (symlinks to mnist/)
# ==============================================================

.PHONY: all otto adam hebbian vis-errors setup clean \
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
vis-errors:
	$(MAKE) -s -C mnist vis-errors
	$(MAKE) -s -C cifar vis-errors

# ═══════════════════════════════════════════════════════════════
# Dataset setup
# ═══════════════════════════════════════════════════════════════
setup-mnist: ; @bash fetch_mnist.sh
setup-cifar: ; @bash fetch_cifar10.sh
setup: setup-mnist setup-cifar

# ═══════════════════════════════════════════════════════════════
# Trained model cache (train if missing, then fast eval)
# ═══════════════════════════════════════════════════════════════
model-mnist-otto:   ; $(MAKE) -s -C mnist model-otto
model-cifar-otto:   ; $(MAKE) -s -C cifar model-otto
model-mnist-adam:   ; $(MAKE) -s -C mnist model-adam
model-cifar-adam:   ; $(MAKE) -s -C cifar model-adam
model-mnist-hebbian:   ; $(MAKE) -s -C mnist model-hebbian
model-cifar-hebbian:   ; $(MAKE) -s -C cifar model-hebbian

models-mnist: model-mnist-otto model-mnist-adam model-mnist-hebbian
models-cifar: model-cifar-otto model-cifar-adam model-cifar-hebbian
models: models-mnist models-cifar 

.PHONY: models models-mnist models-cifar \
	model-mnist-otto model-mnist-hebbian model-mnist-adam \
        model-cifar-otto model-cifar-hebbian model-cifar-adam

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
test-mnist-otto:    ; $(MAKE) -s -C mnist test-otto
test-mnist-adam:    ; $(MAKE) -s -C mnist test-adam
test-mnist-hebbian: ; $(MAKE) -s -C mnist test-hebbian
test-mnist:         ; $(MAKE) -s -C mnist test

# ── CIFAR-10 ───────────────────────────────────────────────
test-cifar-otto:    ; $(MAKE) -s -C cifar test-otto
test-cifar-adam:    ; $(MAKE) -s -C cifar test-adam
test-cifar-hebbian: ; $(MAKE) -s -C cifar test-hebbian
test-cifar:         ; $(MAKE) -s -C cifar test

.PHONY: test-mnist test-mnist-otto test-mnist-adam test-mnist-hebbian \
	test-cifar test-cifar-otto test-cifar-adam test-cifar-hebbian

# ═══════════════════════════════════════════════════════════════
# Clean
# ═══════════════════════════════════════════════════════════════
clean:
	$(MAKE) -C mnist clean
	$(MAKE) -C cifar clean

clean-all:
	$(MAKE) -C mnist clean-all
	$(MAKE) -C cifar clean-all

# ── Push ──────────────────────────────────────────────────────
push:
	@git push -u origin master
