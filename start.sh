#!/bin/sh
# Start HikariLM on macOS or Linux. Extra options pass through, e.g. ./start.sh --port 9000
cd "$(dirname "$0")" || exit 1
if command -v python3 >/dev/null 2>&1; then
  exec python3 start.py "$@"
elif command -v python >/dev/null 2>&1; then
  exec python start.py "$@"
fi
echo "Python is not installed. Get it from https://www.python.org/downloads/" >&2
exit 1
