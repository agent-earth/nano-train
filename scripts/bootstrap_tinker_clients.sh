#!/usr/bin/env bash
set -euo pipefail

package_index="${PYPI_INDEX_URL:-https://pypi.org/simple/}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

find_playground_root() {
  local current="$repo_root"
  while [[ "$current" != "/" ]]; do
    if [[ -d "$current/twinkle/.git" && -d "$current/tinker-cookbook/.git" ]]; then
      printf '%s\n' "$current"
      return 0
    fi
    current="$(dirname "$current")"
  done
  return 1
}

playground_root="${PLAYGROUND_ROOT:-$(find_playground_root)}"
twinkle_source="${TWINKLE_SOURCE:-$playground_root/twinkle}"
cookbook_source="${TINKER_COOKBOOK_SOURCE:-$playground_root/tinker-cookbook}"
bootstrap_python="${NANO_TRAIN_BOOTSTRAP_PYTHON:-$playground_root/ultimate-distill-workspace/.venv/bin/python}"

if [[ ! -x "$bootstrap_python" ]]; then
  echo "bootstrap Python not found: $bootstrap_python" >&2
  exit 1
fi
for source_dir in "$twinkle_source" "$cookbook_source"; do
  if [[ ! -f "$source_dir/pyproject.toml" ]]; then
    echo "source checkout missing pyproject.toml: $source_dir" >&2
    exit 1
  fi
done

twinkle_env="$twinkle_source/.venv-client"
cookbook_env="$cookbook_source/.venv"

if [[ ! -x "$twinkle_env/bin/python" ]]; then
  "$bootstrap_python" -m virtualenv "$twinkle_env"
fi
if [[ ! -x "$cookbook_env/bin/python" ]]; then
  "$bootstrap_python" -m virtualenv "$cookbook_env"
fi

"$twinkle_env/bin/python" -m pip install \
  -i "$package_index" \
  -e "$twinkle_source[client]"
"$cookbook_env/bin/python" -m pip install \
  -i "$package_index" \
  -e "$cookbook_source"

PYTHONPATH="$repo_root" "$twinkle_env/bin/python" -m nano_train.cli \
  tinker-compat \
  --config "$repo_root/configs/tinker/twinkle_qwen35_4b_client_v1.json"
PYTHONPATH="$repo_root" "$cookbook_env/bin/python" -m nano_train.cli \
  tinker-compat \
  --config "$repo_root/configs/tinker/native_qwen35_4b_client_v1.json"
