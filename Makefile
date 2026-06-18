# Makefile — Otto Score Inference (public demo)
# ===============================================
# Self-contained build for the public GitHub release.
# Builds both XNOR and XOR inference executables.
#
# Targets:
#   make (all)  — builds both
#   make setup  — download MNIST dataset (required once)
#   make xnor   — mlp-otto-score-ifc-xnor.exe (default: ~(in^W0))
#   make xor    — mlp-otto-score-ifc-xor.exe  (in^W0)
#   make clean  — remove executables
#   make test   — quick accuracy test with bundled model

CC       = gcc
CFLAGS   = -O3 -march=native -fopenmp -Wall -Wextra -Werror \
           -Wconversion -Wsign-conversion -Wfloat-equal -Wundef -Wshadow -Wunused \
           -Ilib
LDLIBS   = -lm -lz

SRC      = mlp-otto-score-ifc.c

.PHONY: all xnor xor clean test setup

all: xnor xor

setup:
	@echo "Downloading MNIST dataset..."
	@bash fetch_mnist.sh

xnor: mlp-otto-score-ifc-xnor.exe
xor:  mlp-otto-score-ifc-xor.exe

mlp-otto-score-ifc-xnor.exe: $(SRC) ki-common.h lib/maj3.h
	$(CC) $(CFLAGS) -o $@ $(SRC) $(LDLIBS)

mlp-otto-score-ifc-xor.exe: $(SRC) ki-common.h lib/maj3.h
	$(CC) $(CFLAGS) -DH0_XOR -o $@ $(SRC) $(LDLIBS)

test: mlp-otto-score-ifc-xnor.exe
	@echo "Testing XNOR inference with bundled model..."
	@./mlp-otto-score-ifc-xnor.exe --out models/ --evalN 2000
	@echo ""
	@echo "Testing XOR inference with bundled model..."
	@./mlp-otto-score-ifc-xor.exe --out models/ --evalN 2000

clean:
	rm -f *.exe
