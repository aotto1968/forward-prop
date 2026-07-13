# Makefile — Otto Score + Adam + Hebbian (public demo)
# =====================================================
# All build work is delegated to subdirectory Makefiles.
# This Makefile only orchestrates top-level targets.
#
# Programs:
#   mnist/   — Otto Score, Hebbian, Adam for MNIST
#   cifar/   — Otto Score, Hebbian, Adam for CIFAR (symlinks to mnist/)
#   fashion/ — Otto Score, Hebbian, Adam for Fashion-MNIST (symlinks to mnist/)
# ==============================================================

.PHONY: all otto adam hebbian vis-errors setup clean \
        test test-mnist test-cifar test-fashion \
        test-mnist-otto test-mnist-adam test-mnist-hebbian \
        test-cifar-otto test-cifar-adam test-cifar-hebbian \
        test-fashion-otto test-fashion-adam test-fashion-hebbian \
        setup-mnist setup-cifar setup-fashion

# ═══════════════════════════════════════════════════════════════
# Build
# ═══════════════════════════════════════════════════════════════
all:
	$(MAKE) -s -C mnist all
	$(MAKE) -s -C cifar all
	$(MAKE) -s -C fashion all

mnist:    ; $(MAKE) -s -C mnist all
cifar:    ; $(MAKE) -s -C cifar all
fashion:  ; $(MAKE) -s -C fashion all
otto:     mnist cifar fashion
hebbian:  mnist cifar fashion
adam:     mnist cifar fashion
vis-errors:
	$(MAKE) -s -C mnist vis-errors
	$(MAKE) -s -C cifar vis-errors
	$(MAKE) -s -C fashion vis-errors

# ═══════════════════════════════════════════════════════════════
# Dataset setup
# ═══════════════════════════════════════════════════════════════
setup-mnist:   ; @bash fetch_mnist.sh
setup-cifar:   ; @bash fetch_cifar10.sh
setup-fashion: ; @bash fetch_fashion.sh
setup: setup-mnist setup-cifar setup-fashion

# ═══════════════════════════════════════════════════════════════
# Trained model cache (train if missing, then fast eval)
# ═══════════════════════════════════════════════════════════════
model-mnist-otto:     ; $(MAKE) -s -C mnist model-otto
model-cifar-otto:     ; $(MAKE) -s -C cifar model-otto
model-fashion-otto:   ; $(MAKE) -s -C fashion model-otto
model-mnist-adam:     ; $(MAKE) -s -C mnist model-adam
model-cifar-adam:     ; $(MAKE) -s -C cifar model-adam
model-fashion-adam:   ; $(MAKE) -s -C fashion model-adam
model-mnist-hebbian:   ; $(MAKE) -s -C mnist model-hebbian
model-cifar-hebbian:   ; $(MAKE) -s -C cifar model-hebbian
model-fashion-hebbian: ; $(MAKE) -s -C fashion model-hebbian

model-mnist: model-mnist-otto model-mnist-adam model-mnist-hebbian
model-cifar: model-cifar-otto model-cifar-adam model-cifar-hebbian
model-fashion: model-fashion-otto model-fashion-adam model-fashion-hebbian
models: model-mnist model-cifar model-fashion

.PHONY: models model-mnist model-cifar model-fashion \
	model-mnist-otto model-mnist-hebbian model-mnist-adam \
        model-cifar-otto model-cifar-hebbian model-cifar-adam \
        model-fashion-otto model-fashion-hebbian model-fashion-adam

# ═══════════════════════════════════════════════════════════════
# Test — build missing models only, then eval via --import
# ═══════════════════════════════════════════════════════════════
test: all
	$(MAKE) -s test-mnist
	$(MAKE) -s test-cifar
	$(MAKE) -s test-fashion

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

# ── Fashion-MNIST ──────────────────────────────────────────
test-fashion-otto:    ; $(MAKE) -s -C fashion test-otto
test-fashion-adam:    ; $(MAKE) -s -C fashion test-adam
test-fashion-hebbian: ; $(MAKE) -s -C fashion test-hebbian
test-fashion:         ; $(MAKE) -s -C fashion test

.PHONY: test-mnist test-mnist-otto test-mnist-adam test-mnist-hebbian \
	test-cifar test-cifar-otto test-cifar-adam test-cifar-hebbian \
	test-fashion test-fashion-otto test-fashion-adam test-fashion-hebbian

# ═══════════════════════════════════════════════════════════════
# Clean
# ═══════════════════════════════════════════════════════════════
clean:
	$(MAKE) -C mnist clean
	$(MAKE) -C cifar clean
	$(MAKE) -C fashion clean

clean-all:
	$(MAKE) -C mnist clean-all
	$(MAKE) -C cifar clean-all
	$(MAKE) -C fashion clean-all

# ── Push ──────────────────────────────────────────────────────
push:
	@git push -u origin master
