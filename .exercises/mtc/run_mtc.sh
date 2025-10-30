#!/usr/bin/env bash

set -euxo pipefail

# This script runs the MTC example model with sharrow, mirroring the GitHub Actions workflow.

cd $(dirname "$0")

for repo in "driftlesslabs/activitysim" "ActivitySim/activitysim-prototype-mtc"; do
    dir=$(basename "$repo")
    if [ ! -d "$dir" ] || [ -z "$(ls -A "$dir" 2>/dev/null)" ]; then
        gh repo clone "$repo" -- --depth 1
    else
        git -C "$dir" pull --ff-only || git -C "$dir" pull
    fi
done

uv venv
source .venv/bin/activate
uv pip install -e ../.. # install sharrow in editable mode
uv pip install ./activitysim
uv pip install pytest nbmake

cd activitysim-prototype-mtc
python -m pytest ./test
