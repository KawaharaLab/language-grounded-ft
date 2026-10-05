#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
export UV_LINK_MODE=copy
export UV_CACHE_DIR="$repo_root/.cache/uv"
export UV_PYTHON_BIN_DIR="$repo_root/.runtime/bin"
component="${1:?Usage: tools/setup_env.sh simulation|learning}"
case "$component" in
  learning) python_version=3.10.0 ;;
  simulation) python_version=3.11.13 ;;
  *) echo 'Expected simulation or learning' >&2; exit 2 ;;
esac
command -v uv >/dev/null || { echo 'Install uv first: https://docs.astral.sh/uv/getting-started/installation/' >&2; exit 1; }
uv python install --install-dir "$repo_root/.runtime/python" "$python_version"
python_binary="$repo_root/.runtime/python/cpython-${python_version}-linux-x86_64-gnu/bin/python${python_version%.*}"
if [[ ! -x "$repo_root/.venvs/$component/bin/python" ]]; then
  uv venv --python "$python_binary" "$repo_root/.venvs/$component"
fi
uv pip sync --python "$repo_root/.venvs/$component/bin/python" "$repo_root/env/$component-requirements.lock"
