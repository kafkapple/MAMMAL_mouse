#!/bin/bash
# Merge 4-part production keyframe fitting results into single directory
#
# Usage: bash scripts/merge_production_parts.sh
#
# Input:  results/fitting/production_keyframes_part{1,2,3,4}/
# Output: results/fitting/production_900_merged/{obj,params}/

set -e
cd "$(dirname "$0")/.."

SRC_BASE="results/fitting"
DST="$SRC_BASE/production_900_merged"

# Verify all parts complete
TOTAL=0
for p in 1 2 3 4; do
    N=$(ls "$SRC_BASE/production_keyframes_part${p}/obj/" 2>/dev/null | wc -l)
    echo "Part $p: $N/225 OBJs"
    TOTAL=$((TOTAL + N))
    if [ "$N" -lt 225 ]; then
        echo "ERROR: Part $p incomplete ($N/225). Wait for completion."
        exit 1
    fi
done
echo "Total: $TOTAL/900"

# Create merged directory
mkdir -p "$DST/obj" "$DST/params"

# Symlink (not copy) to save disk
for p in 1 2 3 4; do
    SRC="$SRC_BASE/production_keyframes_part${p}"
    for f in "$SRC/obj/"*.obj; do
        ln -sf "$(realpath "$f")" "$DST/obj/$(basename "$f")"
    done
    for f in "$SRC/params/"*.pkl; do
        ln -sf "$(realpath "$f")" "$DST/params/$(basename "$f")"
    done
done

echo ""
echo "Merged: $DST"
echo "  OBJs:   $(ls "$DST/obj/" | wc -l)"
echo "  Params: $(ls "$DST/params/" | wc -l)"
