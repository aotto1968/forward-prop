#!/usr/bin/env bash
# ======================================================================
# cifar-1/run-ensemble.sh — Run training with automatic non-conflicting seed
# ======================================================================
#
# Usage:
#   bash run-ensemble.sh <training-command> [flags...]
#
# Description:
#   1. Creates scores/ directory if missing
#   2. Scans scores/*.ens for existing seeds (from SD{seed} in filenames)
#   3. Generates a random seed NOT in the used set
#   4. Appends --seed <new_seed> --export-merge-scores scores/ to your command
#   5. Runs the command and reports the chosen seed
#
# Examples:
#   bash run-ensemble.sh ./cifar-mlp-bin32-otto-trn-xnor.exe \
#       --hiddenN 128 --epochsN 3 --encoding latest
#
#   bash run-ensemble.sh ./cifar-mlp-bin32-otto-trn-xnor.exe \
#       --qq --hiddenN 256 --encoding exp
#
# The scores/ directory can later be merged with:
#   ./cifar-merge-ensemble.exe scores/
# ======================================================================

SCORES_DIR="scores"
SCRIPT_NAME="$(basename "$0")"

# ── Help function ─────────────────────────────────────────────────────
show_help() {
    cat <<EOF
Usage: bash ${SCRIPT_NAME} [options] <training-command> [flags...]

Runs a training command with an automatically chosen non-conflicting --seed
and --export-merge-scores scores/ appended (unless --export-merge-scores is already given).

Options:
  -h, --help       Show this help text
  --repeat N       Repeat the command N times (default: 1)
                   Each repetition gets a fresh random seed.
                   The command is called once per repetition.

Examples:
  bash ${SCRIPT_NAME} ./cifar-mlp-bin32-otto-trn-xnor.exe \\
      --hiddenN 128 --epochsN 3 --encoding latest

  bash ${SCRIPT_NAME} --repeat 5 ./cifar-mlp-bin32-otto-trn-xnor.exe \\
      --hiddenN 512 --epochsN 3 --splitVN 2 --target-err 0.4

  bash ${SCRIPT_NAME} ./cifar-mlp-bin32-otto-trn-xnor.exe \\
      --hiddenN 1024 --epochsN 7 --encoding latest --export-merge-scores scores-1024/

The scores/ directory can later be merged with:
  ./cifar-merge-ensemble.exe scores/
EOF
    exit 0
}

# ── Parse script options ─────────────────────────────────────────────
REPEAT=1

while true; do
    case "${1:-}" in
        -h|--help)
            show_help
            ;;
        --repeat)
            shift
            if [ $# -eq 0 ]; then
                echo "[${SCRIPT_NAME}] ERROR: --repeat requires a number"
                exit 1
            fi
            if ! [[ "${1}" =~ ^[0-9]+$ ]] || [ "${1}" -lt 1 ]; then
                echo "[${SCRIPT_NAME}] ERROR: --repeat needs a positive integer, got '${1}'"
                exit 1
            fi
            REPEAT=$1
            shift
            ;;
        --)
            shift
            break
            ;;
        -*)
            # Unknown flag — could be the training command's flag
            # Stop parsing (training command starts here)
            break
            ;;
        *)
            # First non-flag argument = start of training command
            break
            ;;
    esac
done

# ── Check arguments ──────────────────────────────────────────────────
if [ $# -eq 0 ]; then
    echo "[${SCRIPT_NAME}] ERROR: Missing training command"
    echo "Usage: bash ${SCRIPT_NAME} [-h] [--repeat N] <training-command> [flags...]"
    exit 1
fi

TRAINING_CMD=("$@")

# ── Check if user already specified --export-merge-scores in command ──
HAS_SAVE_SCORES=0
for arg in "${TRAINING_CMD[@]}"; do
    if [ "${HAS_SAVE_SCORES}" = "1" ]; then
        SCORES_DIR="${arg}"
        HAS_SAVE_SCORES=2
    elif [ "${arg}" = "--export-merge-scores" ]; then
        HAS_SAVE_SCORES=1
    fi
done

# ── Run function — one repetition ─────────────────────────────────────
run_one() {
    local scores_dir="$1"
    shift
    local cmd=("$@")

    # Scan for existing seeds
    local used_file
    used_file=$(mktemp /tmp/used_seeds_XXXXXX)
    local total_used=0

    if ls "${scores_dir}"/*.ens &>/dev/null 2>&1; then
        for f in "${scores_dir}"/*.ens; do
            local base
            base="$(basename "$f" .ens)"
            if [[ "$base" =~ _SD([0-9]+)$ ]]; then
                echo "${BASH_REMATCH[1]}" >> "${used_file}"
                total_used=$((total_used + 1))
            fi
        done
    fi

    # Pick unused random seed
    local new_seed=""
    local attempt
    for ((attempt=0; attempt<1000; attempt++)); do
        local candidate
        candidate=$(od -An -N4 -tu4 /dev/urandom 2>/dev/null | tr -d ' ')
        candidate=$(( candidate & 0x3FFFFFFF ))
        if grep -qFw "${candidate}" "${used_file}" 2>/dev/null; then
            continue
        fi
        new_seed="${candidate}"
        break
    done

    rm -f "${used_file}"

    if [ -z "$new_seed" ]; then
        echo "[${SCRIPT_NAME}] ERROR: Could not find unused seed (${total_used} used)"
        return 1
    fi

    echo "[${SCRIPT_NAME}] Using seed=${new_seed}  (${total_used} existing seeds in ${scores_dir}/)"

    if [ "${HAS_SAVE_SCORES}" -ge 1 ]; then
        # User specified --export-merge-scores, just add --seed
        "${cmd[@]}" --seed "${new_seed}"
    else
        # Default: append --export-merge-scores scores/
        "${cmd[@]}" --seed "${new_seed}" --export-merge-scores "${scores_dir}"
    fi
    local rc=$?

    echo ""
    echo "[${SCRIPT_NAME}] Seed=${new_seed}  Archive: ${scores_dir}/  Exit=${rc}"
    return ${rc}
}

# ── Create scores/ directory ──────────────────────────────────────────
mkdir -p "${SCORES_DIR}"

# ── Run (once or --repeat N) ──────────────────────────────────────────
RUN_EXIT=0
for ((r=1; r<=REPEAT; r++)); do
    if [ "${REPEAT}" -gt 1 ]; then
        echo ""
        echo "═══════════════════════════════════════════════════════════"
        echo "  Repetition ${r} / ${REPEAT}"
        echo "═══════════════════════════════════════════════════════════"
    fi
    run_one "${SCORES_DIR}" "${TRAINING_CMD[@]}"
    rc=$?
    if [ "${rc}" -ne 0 ]; then
        RUN_EXIT="${rc}"
        if [ "${REPEAT}" -gt 1 ]; then
            echo "[${SCRIPT_NAME}] WARNING: Repetition ${r} failed (exit=${rc}), continuing..."
        else
            break
        fi
    fi
done

exit "${RUN_EXIT}"
