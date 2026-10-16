#!/usr/bin/env bash
# Create the project venv and install its dependencies.
# fasttext is built from the bundled fastText-0.9.2 source, because
# PyPI has no fasttext wheel for Python 3.14 (needs g++ and make).
set -euo pipefail
cd "$(dirname "$0")"

# fastText's C++ extension needs a compiler and make.
for tool in g++ make; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "error: $tool is required to build fasttext" >&2
        echo "hint: install build tools first, e.g. sudo apt install build-essential" >&2
        exit 1
    fi
done

python3 -m venv venv
venv/bin/pip install --upgrade pip
# pybind11/setuptools/wheel are build prerequisites: fastText's setup.py
# imports pybind11 while building, and pip's build isolation hides both
# pybind11 and pip itself from it ("RuntimeError: pybind11 install failed.").
# Pre-installing them and building with --no-build-isolation avoids that.
venv/bin/pip install nltk numpy pybind11 setuptools wheel
venv/bin/pip install --no-build-isolation ./fastText-0.9.2

echo
echo "Setup done. Try it:"
echo "  ./domainide.py scale node --dry-run   # preview, no network calls"
echo "  ./domainide.py scale node -a          # real availability check"
echo
echo "Note: the first real run downloads the 7 GB fastText model"
echo "(cc.en.300.bin) automatically — it is not part of the repo."
