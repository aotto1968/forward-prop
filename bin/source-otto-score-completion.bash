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
#   - bash aliases (any alias whose expansion points at one of the above,
#     e.g. alias cot="run-research.sh cifar-mlp-bin32-otto-trn-xnor.exe")
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
# _otto_add_to_path prepends each dir, so the LAST argument has the
# HIGHEST priority. Only the LOCAL work dirs (cifar-1/mnist-1/
# mnist-fashion) are added — the otto-score-ifc source dirs are NOT:
# their stale binaries shadowed the freshly built ones and caused
# bugs (e.g. old --completion lists). The local binary always wins.
_otto_add_to_path() {
    local d
    for d in "$@"; do
        if [[ -d "$d" && ":$PATH:" != *":$d:"* ]]; then
            PATH="$d:$PATH"
        fi
    done
}
_otto_add_to_path \
    "$OTTO_HOME/bin" \
    "$OTTO_HOME/mnist-fashion" \
    "$OTTO_HOME/mnist-1" \
    "$OTTO_HOME/cifar-1"

# Actively remove any otto-score-ifc source dirs from PATH: their stale
# binaries shadow the freshly built local ones. Also works when this file
# is re-sourced in a long-running shell (add-only would leave them behind).
_otto_strip_path() {
    local p out=""
    local -a _parts
    IFS=: read -r -a _parts <<< "$PATH"
    for p in "${_parts[@]}"; do
        [[ "$p" == *"/otto-score-ifc/"* ]] && continue
        [[ -n "$p" ]] && out="${out:+$out:}$p"
    done
    PATH="$out"
}
_otto_strip_path

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
    local _otto_mtime
    _otto_mtime=$(stat -c '%Y' "$binpath" 2>/dev/null || echo 0)
    if [[ "${_OTTO_FLAGS_CACHE_BIN:-}" != "$binpath" || "${_OTTO_FLAGS_CACHE_MTIME:-0}" != "$_otto_mtime" ]]; then
        _OTTO_FLAGS_CACHE=$("$binpath" --completion 2>/dev/null)
        _OTTO_FLAGS_CACHE_BIN="$binpath"
        _OTTO_FLAGS_CACHE_MTIME="$_otto_mtime"
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
    cifar-mlp-bin32-otto-trn-bitvoting.exe \
    cifar-mlp-bin32-otto-trn-xnor-4.exe \
    cifar-mlp-bin32-otto-trn-xnor-8.exe \
    cifar-mlp-bin32-otto-trn-xnor-12.exe \
    cifar-mlp-bin32-otto-trn-xnor-16.exe \
    cifar-mlp-bin32-otto-trn-xnor-32.exe \
    cifar-mlp-bin32-otto-trn-xnor-8-float.exe \
    cifar-mlp-bin32-otto-trn-xnor-8.dbg \
    cifar-mlp-bin32-otto-trn-xnor-8-float.dbg \
    cifar-mlp-otto-xform-view.exe \
    cifar-mlp-otto-xform-samples.exe \
    fashion-mlp-bin32-otto-trn-xnor.exe \
    fashion-mlp-bin32-otto-trn-xor.exe \
    xo; do
    complete -F _otto_trn_complete "$_otto_bin" 2>/dev/null || true
done
# Also register for any cifar-mlp-bin32-otto-trn-xnor-*.exe pattern
# (future bit-width variants, debug builds)
for _otto_bin in cifar-mlp-bin32-otto-trn-xnor-*.exe cifar-mlp-bin32-otto-trn-xnor-*.dbg; do
    [[ -x "$_otto_bin" ]] && complete -F _otto_trn_complete "$_otto_bin" 2>/dev/null || true
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

    # ── Resolve command → binary path ─────────────────────
    local cmd="${COMP_WORDS[0]}"
    local binpath
    binpath=$(type -P "$cmd" 2>/dev/null)
    [[ -z "$binpath" && -x "./$cmd" ]] && binpath="./$cmd"
    [[ -z "$binpath" ]] && { COMPREPLY=(); return 0; }

    # ── Cache: flag list from binary --completion (same
    #    protocol as the trainer — see merge-ensemble.c) ──
    local _otto_mtime
    _otto_mtime=$(stat -c '%Y' "$binpath" 2>/dev/null || echo 0)
    if [[ "${_OTTO_MERGE_FLAGS_CACHE_BIN:-}" != "$binpath" ||
          "${_OTTO_MERGE_FLAGS_CACHE_MTIME:-0}" != "$_otto_mtime" ]]; then
        _OTTO_MERGE_FLAGS_CACHE=$("$binpath" --completion 2>/dev/null)
        _OTTO_MERGE_FLAGS_CACHE_BIN="$binpath"
        _OTTO_MERGE_FLAGS_CACHE_MTIME="$_otto_mtime"
    fi

    # ── Complete flag name ──────────────────────────────
    if [[ "$cur" == --* ]]; then
        # shellcheck disable=SC2207
        COMPREPLY=( $(compgen -W "$_OTTO_MERGE_FLAGS_CACHE" -- "$cur") )
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
            # merge-ensemble takes DIR as its FIRST positional arg,
            # so suggest a directory until one is present; afterwards
            # fall back to the flag list.
            local has_dir=0
            local _w
            for _w in "${COMP_WORDS[@]:1:COMP_CWORD-1}"; do
                [[ "$_w" != -* ]] && { has_dir=1; break; }
            done
            if [[ $has_dir -eq 0 ]]; then
                # shellcheck disable=SC2207
                COMPREPLY=( $(compgen -d -- "$cur") )
            else
                # shellcheck disable=SC2207
                COMPREPLY=( $(compgen -W "$_OTTO_MERGE_FLAGS_CACHE" -- "$cur") )
            fi
            ;;
    esac
}
for _otto_merge_bin in \
    mnist-merge-ensemble.exe \
    cifar-merge-ensemble.exe; do
    complete -F _otto_merge_complete "$_otto_merge_bin" 2>/dev/null || true
done
unset _otto_merge_bin

# ═══════════════════════════════════════════════════════════════════
# Auto-Completion: bash aliases (complete -D fallback)
# ═══════════════════════════════════════════════════════════════════
# PROBLEM: bash does NOT expand aliases before running programmable
# completion. When the user types `cot --hi<TAB>` with
#   alias cot="run-research.sh cifar-mlp-bin32-otto-trn-xnor.exe"
# COMP_WORDS[0] stays "cot", so none of the completions above fire and
# readline falls back to plain filename completion.
#
# FIX: install a DEFAULT completion (complete -D). Whenever the first
# word of the command line is an alias, we expand it via `alias`,
# rewrite COMP_WORDS/COMP_CWORD as if the alias had been typed out,
# and delegate to the matching completion function. Non-alias commands
# are handed back to the PREVIOUS default handler (bash 5.3 ships
# `_comp_complete_load` for on-demand loading), so existing behavior
# is preserved.
#
# The user does NOT need to configure anything — the alias definition
# itself is the single source of truth.

_otto_alias_words() {
    # Usage: _otto_alias_words NAME
    # Sets _OTTO_ALIAS_WORDS to the alias expansion (quote-aware word
    # split, mirrors bash's own alias expansion). Returns 0 when NAME
    # is an alias, 1 otherwise.
    local _out _body _quote
    _out=$(alias "$1" 2>/dev/null) || return 1
    # alias output: alias cot='run-research.sh cifar-mlp-bin32-otto-trn-xnor.exe'
    _body="${_out#*=}"
    _quote="${_body:0:1}"
    if [[ "$_quote" == "'" || "$_quote" == '"' ]]; then
        _body="${_body:1:${#_body}-2}"
    fi
    _OTTO_ALIAS_WORDS=()
    eval "_OTTO_ALIAS_WORDS=($_body)"
    return 0
}

_otto_default_complete() {
    local cmd="${COMP_WORDS[0]}"
    if ! _otto_alias_words "$cmd"; then
        # Not an alias → hand over to the previous -D handler (if any).
        # -o default (set below) keeps readline's filename fallback.
        if [[ -n "$_OTTO_OLD_DEFAULT_COMPLETE" &&
              "$_OTTO_OLD_DEFAULT_COMPLETE" != "_otto_default_complete" &&
              "$(type -t "$_OTTO_OLD_DEFAULT_COMPLETE")" == "function" ]]; then
            "$_OTTO_OLD_DEFAULT_COMPLETE"
        else
            COMPREPLY=()
        fi
        return 0
    fi

    # Rewrite COMP_WORDS as if the alias were expanded:
    #   old: cot --hi            new: run-research.sh cifar-...exe --hi
    local -a new_words=("${_OTTO_ALIAS_WORDS[@]}")
    local n_exp="${#_OTTO_ALIAS_WORDS[@]}"
    local i
    for (( i = 1; i < ${#COMP_WORDS[@]}; i++ )); do
        new_words+=("${COMP_WORDS[i]}")
    done
    COMP_WORDS=("${new_words[@]}")
    COMP_CWORD=$(( n_exp - 1 + COMP_CWORD ))

    # Route to the matching completion function (single level, no recursion).
    case "${COMP_WORDS[0]}" in
        *run-research*|*run-ensemble*) _otto_run_script_complete ;;
        *merge-ensemble*)             _otto_merge_complete ;;
        *vis-errors*)                 _otto_vis_complete ;;
        *)                            _otto_trn_complete ;;
    esac
}

# Preserve the pre-existing default handler (bash 5.3: _comp_complete_load)
# so non-alias commands keep on-demand completion loading.
_OTTO_OLD_DEFAULT_COMPLETE=""
if [[ "$(complete -p -D 2>/dev/null)" =~ -F[[:space:]]+([^[:space:]]+) ]]; then
    _OTTO_OLD_DEFAULT_COMPLETE="${BASH_REMATCH[1]}"
fi
if [[ "$_OTTO_OLD_DEFAULT_COMPLETE" != "_otto_default_complete" ]]; then
    # Only one -D spec can exist; ours replaces it, the old one is called
    # as fallback from _otto_default_complete above.
    complete -D -o default -F _otto_default_complete
fi

# Cleanup helper functions from namespace
unset -f _otto_add_to_path
