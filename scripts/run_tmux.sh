#!/usr/bin/env bash
# Convenience wrapper: start (or attach to) a long-running HiT-Forecast job in tmux.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec bash "$ROOT/scripts/run_l40s.sh" --tmux "$@"
