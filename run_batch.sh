#!/bin/bash
set -e

# Capture any thread allocation supplied by the batch environment BEFORE the
# venv activation script installs its single-thread login-node guards.
# HTCondor does not set OMP_NUM_THREADS itself, so fall back to 2 to match
# `request_cpus = 2` in sweeps.sub.  To run this wrapper locally for the
# one-step smoke test required by CLAUDE.md, pass OMP_NUM_THREADS=1 explicitly.
: "${BATCH_THREADS:=${OMP_NUM_THREADS:-2}}"

# Load python environment
source /eos/user/j/jlai/mango/venv/bin/activate
export PYTHONPATH=/eos/user/j/jlai/mango/conditional-nf-integration:$PYTHONPATH

# venv/bin/activate sets OMP_NUM_THREADS=1 as a shared-login-node guard.  A
# batch slot owns its own CPUs, so override the guards with the captured
# allocation.  Without this, ${OMP_NUM_THREADS:-2} would see the guard's 1 and
# request_cpus would be silently wasted.
export OMP_NUM_THREADS=$BATCH_THREADS
export MKL_NUM_THREADS=$BATCH_THREADS
export OPENBLAS_NUM_THREADS=$BATCH_THREADS
export VECLIB_MAXIMUM_THREADS=$BATCH_THREADS
export NUMEXPR_NUM_THREADS=$BATCH_THREADS
export TORCH_NUM_THREADS=$BATCH_THREADS

echo "=== Host: $(hostname) | Date:$(date) ==="
echo "CPUs visible: $(nproc) | OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "Executing: $@"

exec python -u "$@"
