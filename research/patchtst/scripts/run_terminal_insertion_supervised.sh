#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MODEL=supervised
exec bash "$ROOT/scripts/run_terminal_insertion_byol.sh"
