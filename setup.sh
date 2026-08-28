#!/usr/bin/env bash
# One-time env setup on the mcli box. Installs deps via the Databricks pip proxy
# (mcli security rule: never public PyPI). Models + dataset download from HF at runtime.
set -euo pipefail

mkdir -p ~/.config/pip
cat > ~/.config/pip/pip.conf <<'EOF'
[global]
index-url = https://pypi-proxy.dev.databricks.com/simple/
trusted-host = pypi-proxy.dev.databricks.com
EOF

pip install -r "$(dirname "$0")/requirements.txt"
echo "[setup] deps installed via Databricks proxy."
