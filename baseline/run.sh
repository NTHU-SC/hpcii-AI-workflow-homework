#!/bin/bash
# usage: run.sh INPUT OUTPUT
# Runs the program once on INPUT and writes the final field to OUTPUT.
here="$(cd "$(dirname "$0")" && pwd)"
export OMP_NUM_THREADS=1
exec "$here/heat" "$1" "$2"
