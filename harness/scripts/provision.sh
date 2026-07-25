#!/usr/bin/env bash
# One-time provisioning for the Phase-0 harness.
# Installs swebench (with the cbor2 wheel pin that avoids a Rust build) and checks that
# Docker + the Claude CLI are available. Idempotent.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="$ROOT/.venv"
PY="$VENV/bin/python"

echo "== ensuring pip is current =="
"$PY" -m pip install --upgrade pip -q

echo "== installing swebench (+cbor2 wheel pin, +pandas/pytest) =="
# cbor2>=6 needs a Rust toolchain to build; 5.6.5 ships a prebuilt wheel. Pin it first
# so the resolver reuses it when pulling swebench's transitive `modal` dep.
"$PY" -m pip install "cbor2==5.6.5" --only-binary cbor2 -q
"$PY" -m pip install swebench "cbor2==5.6.5" pandas pytest -q

echo "== verifying imports =="
"$PY" - <<'PYEOF'
import swebench, datasets, pyarrow, pandas
print("swebench", swebench.__version__, "| datasets", datasets.__version__)
from swebench.harness.utils import load_swebench_dataset
ds = load_swebench_dataset("data/interactive-swe", "test", instance_ids=["astropy__astropy-12907"])
print("local dataset loads via swebench OK:", ds[0]["instance_id"])
PYEOF

echo "== checking Docker (needed for the eval pass, not for agent runs) =="
if docker info >/dev/null 2>&1; then
  echo "  docker daemon: UP"
else
  echo "  docker daemon: NOT running -> start Docker Desktop before 'harness eval'"
fi

echo "== checking Claude CLI =="
if command -v claude >/dev/null 2>&1; then
  echo "  claude: $(claude --version 2>/dev/null | head -1)"
else
  echo "  claude NOT on PATH -> set HARNESS_CLAUDE_BIN to its absolute path"
fi

echo "provisioning complete."
