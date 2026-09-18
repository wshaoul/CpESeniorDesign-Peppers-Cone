#!/bin/zsh
set -e

cd "${0:A:h}"

if [[ ! -d macenv && -d .venv ]]; then
  mv .venv macenv
fi

if [[ ! -d macenv ]]; then
  python3 -m venv macenv
fi

# Use the interpreter directly: a migrated venv's activation scripts may still
# reference its old path. Qt skips plugins carrying macOS's hidden file flag.
macenv/bin/python -m pip install --upgrade pip
macenv/bin/python -m pip install -r requirements.txt
chflags -R nohidden macenv
exec macenv/bin/python app.py
