# source-otto-score-completion.bash
# =================================
# Bash auto-completion for Otto Score trainer binaries.
#
# Usage:
#   source otto-score-ifc/bin/source-otto-score-completion.bash
#
# This registers completions for:
#   - Otto Score trainers (*-mlp-bin32-otto-trn-*.exe)
#   - Hebbian trainers
#   - AdamW trainers
#   - vis-errors
#   - merge-ensemble
#   - run-research.sh / run-ensemble.sh
#
# Path resolution:
#   The script auto-detects the Otto Score project root (parent of ../../
#   relative to this file).  Override with:  export OTTO_HOME=/path/to/project
#
# Requirements: bash-completion package (for `complete -F`)

if [[ -z "$OTTO_HOME" ]]; then
    # Auto-detect from script location:
    #   otto-score-ifc/bin/source-otto-score-completion.bash → project root
    _otto_self="$(realpath "${BASH_SOURCE[0]}")"
    OTTO_HOME="$(realpath "$(dirname "$_otto_self")/../..")"
fi

# ── Add trainer directories to PATH (idempotent) ──────────────
_otto_add_to_path() {
    local d
    for d in "$@"; do
        if [[ -d "$d" && ":$PATH:" != *":$d:"* ]]; then
            PATH="$d:$PATH"
        fi
    done
}
_otto_add_to_path \
    "$OTTO_HOME/mnist-1" \
    "$OTTO_HOME/mnist-fashion" \
    "$OTTO_HOME/cifar-1" \
    "$OTTO_HOME/otto-score-ifc/mnist" \
    "$OTTO_HOME/otto-score-ifc/cifar" \
    "$OTTO_HOME/bin"

# ═══════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════
# Auto-Completion: Otto Score Trainer (otto-trn-xnor and friends)
# ═══════════════════════════════════════════════════════════════════

_otto_trn_complete() {
    local cur="${COMP_WORDS[COMP_CWORD]}"
    local prev="${COMP_WORDS[COMP_CWORD-1]}"

    # ── Resolve command → binary path ─────────────────────
    local cmd="${COMP_WORDS[0]}"
    # run-research.sh / run-ensemble.sh: binary is first positional arg
    if [[ "$cmd" == *run-research* || "$cmd" == *run-ensemble* ]]; then
        for (( _i=1; _i<COMP_CWORD; _i++ )); do
            local _w="${COMP_WORDS[_i]}"
            if [[ "$_w" == "--repeat" ]]; then
                (( _i++ ))
                continue
            fi
            if [[ "$_w" != -* && "$_w" != "-d" && "$_w" != "--debug" && "$_w" != "--no-log" ]]; then
                cmd="$_w"
                break
            fi
        done
    fi
    # Resolve aliases → real binary name
    case "$cmd" in
        xo)  cmd="cifar-mlp-bin32-otto-trn-xnor.exe"  ;;
        xof) cmd="fashion-mlp-bin32-otto-trn-xnor.exe" ;;
        xom) cmd="mnist-mlp-bin32-otto-trn-xnor.exe"   ;;
        ve)  cmd="mnist-mlp-otto-vis-errors.exe"       ;;
    esac
    local binpath
    binpath=$(type -P "$cmd" 2>/dev/null)
    [[ -z "$binpath" && -x "./$cmd" ]] && binpath="./$cmd"
    [[ -z "$binpath" ]] && { COMPREPLY=(); return 0; }

    # ── Cache: flag list from binary --completion ────────
    if [[ "${_OTTO_FLAGS_CACHE_BIN:-}" != "$binpath" ]]; then
        _OTTO_FLAGS_CACHE=$("$binpath" --completion 2>/dev/null)
        _OTTO_FLAGS_CACHE_BIN="$binpath"
    fi

    # ── Complete flag name ──────────────────────────────
    if [[ "$cur" == --* ]]; then
        # shellcheck disable=SC2207
        COMPREPLY=( $(compgen -W "$_OTTO_FLAGS_CACHE" -- "$cur") )
        return 0
    fi

    # ── Complete argument after specific flag ────────────
    local hint
    hint=$("$binpath" --completion "$prev" 2>/dev/null)
    local hint_type="${hint%% *}"
    local hint_vals="${hint#* }"
    [[ "$hint_type" == "$hint_vals" ]] && hint_vals=""

    case "$hint_type" in
        file)
            # shellcheck disable=SC2207
            COMPREPLY=( $(compgen -f -- "$cur") )
            ;;
        dir)
            # shellcheck disable=SC2207
            COMPREPLY=( $(compgen -d -- "$cur") )
            ;;
        num|float)
            COMPREPLY=()  # user types the number
            ;;
        token)
            if [[ -n "$hint_vals" ]]; then
                # shellcheck disable=SC2207
                COMPREPLY=( $(compgen -W "$hint_vals" -- "$cur") )
            else
                COMPREPLY=()  # free-form token
            fi
            ;;
        *)
            # Fallback: show all flags (e.g. first arg after command)
            # shellcheck disable=SC2207
            COMPREPLY=( $(compgen -W "$_OTTO_FLAGS_CACHE" -- "$cur") )
            ;;
    esac
}

# Register completion for all trainer binaries
for _otto_bin in \
    mnist-mlp-bin32-otto-trn-xnor.exe \
    mnist-mlp-bin32-otto-trn-xor.exe \
    cifar-mlp-bin32-otto-trn-xnor.exe \
    cifar-mlp-bin32-otto-trn-xor.exe \
    cifar-mlp-bin32-otto-trn-vn4.exe \
    fashion-mlp-bin32-otto-trn-xnor.exe \
    fashion-mlp-bin32-otto-trn-xor.exe \
    xo; do
    complete -F _otto_trn_complete "$_otto_bin" 2>/dev/null || true
done
unset _otto_bin

# ═══════════════════════════════════════════════════════════════════
# Auto-Completion: vis-errors
# ═══════════════════════════════════════════════════════════════════

_otto_vis_complete() {
    local cur="${COMP_WORDS[COMP_CWORD]}"
    local prev="${COMP_WORDS[COMP_CWORD-1]}"
    local opts="--predictions --export --max --numH --help"

    if [[ "$cur" == --* ]]; then
        # shellcheck disable=SC2207
        COMPREPLY=( $(compgen -W "$opts" -- "$cur") )
        return 0
    fi

    case "$prev" in
        --predictions|--export|--max|--numH)
            # shellcheck disable=SC2207
            COMPREPLY=( $(compgen -f -- "$cur") )
            return 0
            ;;
    esac

    # shellcheck disable=SC2207
    COMPREPLY=( $(compgen -W "$opts" -- "$cur") )
}
complete -F _otto_vis_complete mnist-mlp-otto-vis-errors.exe ve 2>/dev/null || true

# ═══════════════════════════════════════════════════════════════════
# Auto-Completion: run-research.sh / run-ensemble.sh
# ═══════════════════════════════════════════════════════════════════

_otto_run_script_complete() {
    local cur="${COMP_WORDS[COMP_CWORD]}"
    local prev="${COMP_WORDS[COMP_CWORD-1]}"

    # run-ensemble.sh has its own flags BEFORE the training command
    if [[ "${COMP_WORDS[0]}" == *run-ensemble* ]]; then
        if [[ "$cur" == --* ]]; then
            local in_trainer=0
            for (( i=1; i<COMP_CWORD; i++ )); do
                local w="${COMP_WORDS[i]}"
                if [[ "$w" != -* || "$w" == "--repeat" ]]; then
                    if [[ "$w" == "--repeat" ]]; then
                        i=$((i+1))
                        continue
                    fi
                    if [[ "$w" != --* ]]; then
                        in_trainer=1
                        break
                    fi
                fi
            done
            if [[ $in_trainer -eq 1 ]]; then
                _otto_trn_complete
                return 0
            fi
            # shellcheck disable=SC2207
            COMPREPLY=( $(compgen -W "--help --repeat" -- "$cur") )
            return 0
        fi
        if [[ "$prev" == "--repeat" ]]; then
            COMPREPLY=()
            return 0
        fi
    fi

    local bin_found=0
    for (( i=1; i<COMP_CWORD; i++ )); do
        local w="${COMP_WORDS[i]}"
        if [[ "${COMP_WORDS[0]}" == *run-ensemble* ]]; then
            if [[ "$w" == "--repeat" ]]; then
                i=$((i+1))
                continue
            fi
            if [[ "$w" != -* ]]; then
                bin_found=1
                break
            fi
        else
            if [[ "$w" != -* && "$w" != "-d" && "$w" != "--debug" ]]; then
                bin_found=1
                break
            fi
        fi
    done

    if [[ $bin_found -eq 1 ]]; then
        _otto_trn_complete
        return 0
    fi

    # Complete binary/script name — suggest .exe and .py files
    local dirs=( "$OTTO_HOME/mnist-1" "$OTTO_HOME/mnist-fashion" "$OTTO_HOME/cifar-1" "$OTTO_HOME/otto-score-ifc/mnist" "$OTTO_HOME/otto-score-ifc/cifar" "$OTTO_HOME/old/python-first-try" )
    local suggestions=""
    for d in "${dirs[@]}"; do
        if [[ -d "$d" ]]; then
            suggestions+=" $(compgen -f -X '!*.exe' -- "$d/" 2>/dev/null)"
            suggestions+=" $(compgen -f -X '!*.py' -- "$d/" 2>/dev/null)"
        fi
    done
    suggestions+=" $(compgen -f -X '!*.exe' -- "$cur" 2>/dev/null)"
    suggestions+=" $(compgen -f -X '!*.py' -- "$cur" 2>/dev/null)"
    # shellcheck disable=SC2207
    COMPREPLY=( $(compgen -W "$suggestions" -- "$cur") )
}

complete -F _otto_run_script_complete run-research.sh run-ensemble.sh 2>/dev/null || true

# ═══════════════════════════════════════════════════════════════════
# Auto-Completion: merge-ensemble
# ═══════════════════════════════════════════════════════════════════

_otto_merge_complete() {
    local cur="${COMP_WORDS[COMP_CWORD]}"
    local prev="${COMP_WORDS[COMP_CWORD-1]}"
    local opts="--num --max --save --sort --filter --help"

    if [[ "$cur" == --* ]]; then
        # shellcheck disable=SC2207
        COMPREPLY=( $(compgen -W "$opts" -- "$cur") )
        return 0
    fi

    case "$prev" in
        --num|--max)
            COMPREPLY=()
            return 0
            ;;
        --save)
            # shellcheck disable=SC2207
            COMPREPLY=( $(compgen -f -- "$cur") )
            return 0
            ;;
        --sort)
            # shellcheck disable=SC2207
            COMPREPLY=( $(compgen -W "seed ctime" -- "$cur") )
            return 0
            ;;
        --filter)
            COMPREPLY=()
            return 0
            ;;
    esac

    # shellcheck disable=SC2207
    COMPREPLY=( $(compgen -d -- "$cur") )
}
for _otto_merge_bin in \
    mnist-merge-ensemble.exe \
    cifar-merge-ensemble.exe; do
    complete -F _otto_merge_complete "$_otto_merge_bin" 2>/dev/null || true
done
unset _otto_merge_bin

# Cleanup helper functions from namespace
unset -f _otto_add_to_path
