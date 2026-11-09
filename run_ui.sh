#!/usr/bin/env bash
# Launcher script for TabsGeneration Interactive Web UI
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

if [ -f "$DIR/prototype/venv/bin/python" ]; then
    PYTHON_EXEC="$DIR/prototype/venv/bin/python"
else
    PYTHON_EXEC="python3"
fi

echo "[TabsGeneration] Launching Interactive Audio & Tab Inspector..."
exec "$PYTHON_EXEC" "$DIR/prototype/ui_server.py" "$@"

