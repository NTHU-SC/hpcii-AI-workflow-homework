#!/bin/bash
# Build step (not timed). Runs in this directory's copy.
set -e
cd "$(dirname "$0")"
gcc -O1 -fopenmp -o heat heat.c
