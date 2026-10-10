# Source this file to set up the toolchain used by RUNBOOK.md:
#
#     source /eos/user/j/jlai/mango/conditional-nf-integration/env.sh
#
# It only exports variables -- it installs nothing and changes nothing on disk.
#
# NOTE: the paths below are specific to this machine (/eos/user/j/jlai/mango).
# On any other machine, edit the three paths at the top.

export NFI_ROOT=/eos/user/j/jlai/mango
export NFI_REPO="$NFI_ROOT/conditional-nf-integration"

# Python 3.11.5 venv with numpy/scipy/pandas/matplotlib/torch/vegas/pytest.
# The system python3 here is 3.9, which is too old for MG5 3.6.7.
export NFI_PY="$NFI_ROOT/venv/bin/python"

# MadGraph5_aMC@NLO 3.6.7, installed at "$NFI_ROOT/MG5_aMC_v3_6_7", invoked
# through a wrapper that runs it with $NFI_PY instead of the system python3.
export NFI_MG5="$NFI_ROOT/bin/mg5_aMC"

# The drivers import `nf` and the top-level `madgraph_*` modules, but Python
# puts the *script's* directory on sys.path, not the repository root.
export PYTHONPATH="$NFI_REPO"

# Directory of the generated MadGraph standalone for gg -> t t~.
# MadGraph's initialisation resolves Cards/ relative to the CWD, so every
# MadGraph driver must be run from inside this directory.  See RUNBOOK.md §8.
export NFI_STANDALONE="$NFI_REPO/generated/gg_ttbar_standalone"

echo "NFI_REPO       = $NFI_REPO"
echo "NFI_PY         = $NFI_PY"
echo "NFI_MG5        = $NFI_MG5"
echo "NFI_STANDALONE = $NFI_STANDALONE"
if [ ! -d "$NFI_STANDALONE" ]; then
    echo
    echo "NOTE: $NFI_STANDALONE does not exist yet."
    echo "      Run the generation step in RUNBOOK.md section 5 first."
fi
