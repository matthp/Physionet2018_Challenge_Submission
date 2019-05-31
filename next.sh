#! /bin/bash
#
# file: next.sh

set -e
set -o pipefail

RECORD=$1

python3 run_my_classifier.py "$RECORD"
