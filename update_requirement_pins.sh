#!/bin/sh

pip-compile requirements-dev.in --allow-unsafe -U -o requirements-dev.txt
pip-compile requirements.in --allow-unsafe -U -o requirements.txt
