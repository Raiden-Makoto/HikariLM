#!/bin/sh
# Start HikariLM on a Mac with Apple Silicon. The first run also installs the hikarilm
# command (hikarilm start, stop, restart, status, log, update); this takes the same words.
cd "$(dirname "$0")" || exit 1
if command -v python3 >/dev/null 2>&1; then
  exec python3 start.py "$@"
elif command -v python >/dev/null 2>&1; then
  exec python start.py "$@"
fi
echo "Python is not installed. Get it from https://www.python.org/downloads/" >&2
exit 1
