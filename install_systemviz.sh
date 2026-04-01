#!/usr/bin/env bash
set -euo pipefail

mode="copy"
dest="${HOME}/.local/bin/systemviz"
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_file="${repo_dir}/systemviz.py"

usage() {
  cat <<'EOF'
Usage: ./install_systemviz.sh [--global] [--symlink]

Options:
  --global   Install to /usr/local/bin/systemviz instead of ~/.local/bin/systemviz
  --symlink  Create a symlink to the repo file instead of copying it
EOF
}

while (($#)); do
  case "$1" in
    --global)
      dest="/usr/local/bin/systemviz"
      ;;
    --symlink)
      mode="symlink"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown option: %s\n' "$1" >&2
      usage >&2
      exit 1
      ;;
  esac
  shift
done

if [[ ! -f "${source_file}" ]]; then
  printf 'Missing source file: %s\n' "${source_file}" >&2
  exit 1
fi

dest_dir="$(dirname "${dest}")"

if [[ "${dest}" == "${HOME}"/* ]]; then
  mkdir -p "${dest_dir}"
else
  if [[ ! -d "${dest_dir}" ]]; then
    printf 'Destination directory does not exist: %s\n' "${dest_dir}" >&2
    exit 1
  fi
  if [[ ! -w "${dest_dir}" ]]; then
    printf 'Destination is not writable: %s\n' "${dest_dir}" >&2
    printf 'Re-run with sudo for --global, or omit --global for a user-local install.\n' >&2
    exit 1
  fi
fi

if [[ "${mode}" == "symlink" ]]; then
  ln -sfn "${source_file}" "${dest}"
else
  install -m 0755 "${source_file}" "${dest}"
fi

printf 'Installed systemviz to %s\n' "${dest}"
printf 'Run: systemviz\n'
