#!/usr/bin/env bash
# Rebuilds web/vendor/codemirror.js from entry.js.
#
# Only needed to upgrade CodeMirror or change the languages. Running HikariLM
# never needs Node: the finished file is committed. If there is no Node here,
# a temporary copy is downloaded and thrown away afterwards.
set -euo pipefail
cd "$(dirname "$0")"

TEMP=""
cleanup() { [ -n "$TEMP" ] && rm -rf "$TEMP"; return 0; }
trap cleanup EXIT

if ! command -v node >/dev/null 2>&1; then
  version="v22.11.0"
  case "$(uname -s)" in Linux) os=linux ;; Darwin) os=darwin ;; *) echo "Install Node 18 or newer, then run this again." >&2; exit 1 ;; esac
  case "$(uname -m)" in x86_64|amd64) arch=x64 ;; aarch64|arm64) arch=arm64 ;; *) echo "Install Node 18 or newer, then run this again." >&2; exit 1 ;; esac
  TEMP="$(mktemp -d)"
  echo "No Node found. Downloading a temporary copy ($version)..."
  curl -fsSL "https://nodejs.org/dist/$version/node-$version-$os-$arch.tar.gz" | tar -xz -C "$TEMP"
  export PATH="$TEMP/node-$version-$os-$arch/bin:$PATH"
fi

npm install --no-audit --no-fund
npm run build
cp node_modules/@codemirror/view/LICENSE ../../web/vendor/CODEMIRROR-LICENSE
ls -l ../../web/vendor/codemirror.js
