#!/bin/sh

python3 -m venv venv
. venv/bin/activate

pip install -U pip setuptools wheel pip-tools

pip-sync requirements-dev.txt
