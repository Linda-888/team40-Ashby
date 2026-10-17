#!/bin/bash
# Copy the lean results into the repo and push them to GitHub (run on Delta, from code/).
#   bash sync_results.sh
# Lean = deck figures + key numbers, scorecard, every model's result JSON and summary plots, and the
# patient list (public TCGA barcodes). Per-patient heatmaps, embeddings, tiles and slides stay on Delta.
# Pushing needs the write deploy key ~/.ssh/tf_deploy_rw (see README); pulls use the read-only key.
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.sh"
cd "$HERE"
export GIT_SSH_COMMAND="ssh -i $HOME/.ssh/tf_deploy_rw -o IdentitiesOnly=yes"

git pull -q --ff-only
rm -rf results
mkdir -p results
cp -r "$TF_ROOT/results/figures" results/
cp "$TF_ROOT"/results/scorecard.md "$TF_ROOT"/results/scorecard.csv results/ 2>/dev/null || true
for d in "$TF_ROOT"/results/*/; do
    m=$(basename "$d")
    [ "$m" = figures ] && continue
    mkdir -p "results/$m"
    cp "$d"*.json "$d"*.png "results/$m/" 2>/dev/null || true
done
cp "$TF_ROOT/data/patients.csv" results/

git add -A results
if git diff --cached --quiet; then
    echo "results unchanged, nothing to push"
    exit 0
fi
git commit -q -m "Results synced from Delta ($(date '+%F %H:%M'))" -m "Code at $(git log -1 --format=%h)." \
    ${COMMIT_TRAILER:+-m "$COMMIT_TRAILER"}
git push -q origin HEAD:main
echo "pushed $(git log -1 --format=%h): $(git show --stat --format= HEAD | tail -1)"
