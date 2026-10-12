#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Drive the MadGraph scattering sweep on the local Windows box.
#
#   bash scripts/run_all_windows.sh build     [proc ...]
#   bash scripts/run_all_windows.sh integrand [proc ...]
#   bash scripts/run_all_windows.sh train     [proc ...]
#   bash scripts/run_all_windows.sh heldout   [proc ...]
#   bash scripts/run_all_windows.sh all       [proc ...]
#
# `proc` is one of gg_ttbar, gg_tth, uu_ww_4l (default: all three).
#
# Environment overrides:
#   NFI_PY       python interpreter   (default: conda env `workspace`)
#   NFI_MG5      path to mg5_aMC      (default: the .bat launcher)
#   NFI_PROFILE  speed profile        (default: smoke)
#
# See RUNBOOK.md for what each stage does and the Windows-specific traps
# (working directory, py.py, static linking) that this script handles.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_WIN="C:/Users/laiji/test/conditional-nf-integration"
PY="${NFI_PY:-/c/Users/laiji/.conda/envs/workspace/python.exe}"
MG5="${NFI_MG5:-C:/Users/laiji/test/bin/mg5_aMC.bat}"
PROFILE="${NFI_PROFILE:-smoke}"

# process name -> matrix-element dimension (sets the checkpoint filename)
DIM_GG_TTBAR=2
DIM_GG_TTH=5
DIM_UU_WW_4L=8

ALL_PROCESSES=(gg_ttbar gg_tth uu_ww_4l)

die() { echo "error: $*" >&2; exit 1; }

dimension_of() {
    case "$1" in
        gg_ttbar)  echo "$DIM_GG_TTBAR" ;;
        gg_tth)    echo "$DIM_GG_TTH" ;;
        uu_ww_4l)  echo "$DIM_UU_WW_4L" ;;
        *) die "unknown process '$1'" ;;
    esac
}

standalone_of() { echo "$REPO/generated/$1_standalone"; }
standalone_win_of() { echo "$REPO_WIN/generated/$1_standalone"; }

# MadGraph drivers must run with the CWD inside the standalone tree, because
# lha_read.f looks for ./Cards/ident_card.dat and calls a Fortran `stop`
# (exit status 0, no message) when it cannot find it.  Every other path is
# therefore passed absolute.
run_in_standalone() {
    local proc="$1"; shift
    local dir; dir="$(standalone_of "$proc")"
    [ -d "$dir" ] || die "$dir does not exist; run the 'build' stage first"
    ( cd "$dir" && PYTHONPATH="$REPO_WIN" "$PY" -u "$@" )
}

stage_build() {
    local proc="$1"
    echo "=== build $proc ==="
    local dir; dir="$(standalone_of "$proc")"
    if [ -d "$dir" ]; then
        echo "    $dir already exists - skipping (delete it to rebuild)"
        return 0
    fi
    ( cd "$REPO" && PYTHONPATH="$REPO_WIN" "$PY" -u \
        experiments/setup_madgraph_process.py \
        --config "configs/processes/$proc.json" --mg5 "$MG5" )
}

stage_integrand() {
    local proc="$1"
    echo "=== integrand check $proc ==="
    run_in_standalone "$proc" \
        "$REPO_WIN/experiments/validate_madgraph_process_integrand.py" \
        --config "$REPO_WIN/configs/processes/$proc.json" --points 500
}

stage_train() {
    local proc="$1"
    echo "=== train $proc (profile=$PROFILE) ==="
    run_in_standalone "$proc" \
        "$REPO_WIN/experiments/run_madgraph_process_conditional_copula.py" \
        --config "$REPO_WIN/configs/processes/$proc.json" \
        --speed-profile "$PROFILE" \
        --no-compile-kernels \
        --output "$REPO_WIN/results/${proc}_copula_${PROFILE}"
}

stage_heldout() {
    local proc="$1"
    local dim; dim="$(dimension_of "$proc")"
    local checkpoint="$REPO/results/${proc}_copula_${PROFILE}/madgraph_${proc}_${dim}_final_model.pt"
    if [ ! -f "$checkpoint" ]; then
        die "no checkpoint at $checkpoint - run the 'train' stage first"
    fi
    echo "=== held-out validation $proc ==="
    run_in_standalone "$proc" \
        "$REPO_WIN/experiments/validate_madgraph_process_integral.py" \
        --config "$REPO_WIN/configs/processes/$proc.json" \
        --checkpoint "$REPO_WIN/results/${proc}_copula_${PROFILE}/madgraph_${proc}_${dim}_final_model.pt" \
        --output "$REPO_WIN/results/${proc}_validation_${PROFILE}"
}

main() {
    local stage="${1:-}"; shift || true
    [ -n "$stage" ] || die "usage: $0 {build|integrand|train|heldout|all} [proc ...]"

    local processes=("$@")
    [ ${#processes[@]} -gt 0 ] || processes=("${ALL_PROCESSES[@]}")

    for proc in "${processes[@]}"; do
        dimension_of "$proc" >/dev/null   # validate the name early
    done

    for proc in "${processes[@]}"; do
        case "$stage" in
            build)     stage_build     "$proc" ;;
            integrand) stage_integrand "$proc" ;;
            train)     stage_train     "$proc" ;;
            heldout)   stage_heldout   "$proc" ;;
            all)
                stage_build "$proc"
                stage_integrand "$proc"
                stage_train "$proc"
                stage_heldout "$proc"
                ;;
            *) die "unknown stage '$stage'" ;;
        esac
    done
    echo "=== done: $stage ==="
}

main "$@"
