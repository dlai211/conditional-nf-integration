#!/bin/bash
echo "pwd            = $(pwd)"
echo "hostname       = $(hostname)"
echo "--- is the repo visible? ---"
for p in /eos/user/j/jlai/mango/conditional-nf-integration \
         /eos/user/j/jlai/mango/conditional-nf-integration/experiments \
         /eos/user/j/jlai/mango/venv; do
  if [ -d "$p" ]; then echo "  YES  $p"; else echo "  NO   $p"; fi
done
echo "--- sandbox contents ---"
ls -la | head -15
echo "--- _CONDOR_SCRATCH_DIR = ${_CONDOR_SCRATCH_DIR:-<unset>}"
