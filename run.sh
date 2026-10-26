#!/usr/bin/env bash
set -euo pipefail

# Resolve resources relative to the application directory.
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# Use the active environment or a local .venv.
if [[ -z "${VIRTUAL_ENV:-}" && -f .venv/bin/activate ]]; then
    source .venv/bin/activate
fi

python app.py
