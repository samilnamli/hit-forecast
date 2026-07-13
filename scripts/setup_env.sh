#!/usr/bin/env bash
# Set up the HiT-Forecast environment on the L40S host.
set -euo pipefail

PYTHON=${PYTHON:-python3}
VENV=${VENV:-.venv}

$PYTHON -m venv "$VENV"
# shellcheck disable=SC1090
source "$VENV/bin/activate"
pip install --upgrade pip wheel

# Core (torch with CUDA is expected to be preinstalled in most GPU images;
# otherwise install the matching CUDA wheel from https://pytorch.org).
pip install -e ".[gifteval,dev]"

echo
echo "Core install done. To install the foundation-model experts, run:"
echo "  pip install -e '.[experts]'"
echo "Some experts (TimesFM 2.5, TiRex, Chronos-2) may need model-specific extras;"
echo "see their model cards. Verify GPU: python -c 'import torch; print(torch.cuda.is_available())'"
