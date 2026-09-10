#!/usr/bin/env bash
set -euo pipefail

mode="${1:-gpu-cu118}"
prefix="${2:-/workspace/pkgs/MimicNeoAI_immunogenicity_${mode}}"
python_bin="${PYTHON_BIN:-python3.10}"
torch_version="${TORCH_VERSION:-2.0.1}"

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_dir="${prefix}/.venv"

case "${mode}" in
  cpu)
    torch_spec="torch==${torch_version}+cpu"
    torch_index="https://download.pytorch.org/whl/cpu"
    ;;
  gpu-cu118)
    torch_spec="torch==${torch_version}+cu118"
    torch_index="https://download.pytorch.org/whl/cu118"
    ;;
  *)
    echo "Usage: $0 [cpu|gpu-cu118] [install_prefix]" >&2
    exit 2
    ;;
esac

mkdir -p "${prefix}"
"${python_bin}" -m venv "${venv_dir}"
"${venv_dir}/bin/python" -m pip install --upgrade pip setuptools wheel
"${venv_dir}/bin/python" -m pip install --index-url "${torch_index}" "${torch_spec}"
"${venv_dir}/bin/python" -m pip install \
  "scikit-learn>=1.3" \
  "pandas>=2.3.3" \
  "numpy<2" \
  "pyyaml>=6.0.3" \
  "tqdm>=4.67.1"
"${venv_dir}/bin/python" -m pip install --no-deps -e "${repo_dir}"

"${venv_dir}/bin/python" - <<'PY'
import importlib.util
import sys

print("python", sys.executable)
for name in ("torch", "sklearn", "pandas", "numpy", "yaml", "tqdm", "mimicneoai"):
    spec = importlib.util.find_spec(name)
    print(name, "OK" if spec else "MISSING")

import torch

print("torch_version", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
PY

cat <<EOF

Set one of these in a pipeline YAML:

others:
  immunogenicity_python_bin: "${venv_dir}/bin/python"
  immunogenicity_device: "auto"

For CPU-only execution with this environment, set:

others:
  immunogenicity_device: "cpu"
EOF
