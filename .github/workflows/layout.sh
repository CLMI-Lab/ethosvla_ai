#!/usr/bin/env bash
# Repairs the release layout and quarantines the pre-correction results export
# so CI reports an honest state instead of failing on plumbing.
#
# Run from your own checkout of CLMI-Lab/ethosvla_ai:
#   bash fix_layout.sh
# then review `git status`, commit, and push.
#
# Reversible: every step is a rename or an untrack. Nothing is deleted, and no
# metric value is altered anywhere.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

echo "==> Moving files out of .github/workflows/ to their correct paths"
mkdir -p scripts

# The corrected README currently lives in .github/workflows/ and the
# uncorrected v1 is at the root. This swaps them.
[ -f .github/workflows/README.md ] && \
  git mv -f .github/workflows/README.md README.md

[ -f .github/workflows/validate_artifacts.py ] && \
  git mv .github/workflows/validate_artifacts.py scripts/validate_artifacts.py

[ -f .github/workflows/requirements.txt ] && \
  git mv .github/workflows/requirements.txt requirements.txt

# GitHub only recognises a license at the repo root, docs/, or .github/.
[ -f .github/workflows/LICENSE ] && git mv .github/workflows/LICENSE LICENSE

echo "==> Removing the Django starter template"
[ -f .github/workflows/django.yml ] && git rm -q .github/workflows/django.yml

echo "==> Untracking the committed archive (kept on disk)"
[ -f EthosVLA.zip ] && git rm -q --cached EthosVLA.zip

echo "==> Quarantining the pre-correction results export"
# This file reports PVS 0.0 / HAR 97.2 / TSR 98.7 pooled over two benchmarks.
# The paper reports PVS 1.1 / HAR 96.4 / TSR 97.2 over three (n = 475).
# It is renamed, not edited: no number in it is changed, and nothing is
# invented to replace it. Rename it back once the corrected export exists.
if [ -f Results/ethosvla_final_results_30baselines.json ]; then
  git mv Results/ethosvla_final_results_30baselines.json \
         Results/DEPRECATED_pre-correction_results.json
fi

echo "==> Files still to copy in from the chat"
for f in .gitignore .github/workflows/dashboard.yml .github/workflows/ci.yml \
         scripts/validate_artifacts.py README.md; do
  [ -f "$f" ] || echo "    MISSING: $f"
done

echo
git status --short

cat <<'EOF'

Next:
  git add -A
  git commit -m "Fix release layout; quarantine pre-correction results export"
  git push

Still open, and not fixed by this script:
  1. Re-run the evaluation with episode-level PVS, bootstrap pooling and
     DROID; export to Results/ethosvla_final_results_30baselines.json and
     regenerate figures/fig4_ablation.* from it. Until that lands, no
     released artifact backs the paper's headline numbers, and CI says so.
  2. figures/fig4_ablation.png/.pdf still render the superseded row
     (97.2 / 98.7 / 0.0 / 3.3 / 3.4 / 55.6) and deltas that contradict
     Table 15. This is Figure 6 in the manuscript.
  3. Publish LICENSE-BAAD (text from https://www.licenses.ai/) or keep the
     README's "not yet published" wording.
  4. Results/scannet_ethosvla_results.csv is a header with no rows.
EOF
