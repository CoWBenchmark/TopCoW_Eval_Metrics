#!/bin/bash

# exit on error
set -e

# clean up previous install
rm -rf env_py310/

# use py3.10
python3.10 -m venv env_py310

source env_py310/bin/activate

python -m pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"
