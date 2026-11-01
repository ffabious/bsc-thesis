#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
document="${1:-$repo_root/thesis/thesis.tex}"
output_dir="${2:-$repo_root/build/latex}"

# GUI applications may not inherit the login shell's Homebrew PATH.
export PATH="/opt/homebrew/bin:/usr/local/bin:/Library/TeX/texbin:$PATH"

for tool in latexmk pdflatex biber; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    printf 'Missing %s. On macOS install the toolchain with: brew install texlive biber\n' "$tool" >&2
    exit 127
  fi
done

# TeX writes a separate auxiliary file for each included chapter.
mkdir -p "$output_dir/chapters"

exec latexmk \
  -pdf \
  -cd \
  -synctex=1 \
  -interaction=nonstopmode \
  -file-line-error \
  -halt-on-error \
  "-outdir=$output_dir" \
  "$document"
