#!/usr/bin/env bash
# Create the project venv and install its dependencies.
# fasttext is built from the bundled fastText-0.9.2 source, because
# PyPI has no fasttext wheel for Python 3.14 (needs g++ and make).
set -euo pipefail
cd "$(dirname "$0")"

python3 -m venv venv
venv/bin/pip install --upgrade pip
venv/bin/pip install nltk numpy
venv/bin/pip install ./fastText-0.9.2

echo
echo "Setup done. Try it:"
echo "  ./domainide.py scale node --dry-run   # preview, no network calls"
echo "  ./domainide.py scale node -a          # real availability check"
echo
echo "Note: the first real run downloads the 7 GB fastText model"
echo "(cc.en.300.bin) automatically — it is not part of the repo."
