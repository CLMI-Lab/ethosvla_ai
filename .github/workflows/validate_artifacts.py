#!/usr/bin/env python3
"""
Artifact consistency checks for the EthosVLA release.

Run locally or in CI:

    python scripts/validate_artifacts.py

Checks performed
----------------
1.  Every .json in the repo parses.
2.  Every .csv in the repo parses and is non-empty.
3.  Every .ipynb is valid notebook JSON.
4.  Headline metrics in released result files agree with the values the
    paper reports. This is the check that matters: the paper states that
    an episode-level PVS makes a 0.0% figure impossible, so any released
    artifact still carrying PVS 0.0 for the full stack is a superseded
    pre-correction file and must be regenerated before release.
5.  Files the README points at actually exist.

Exit code is non-zero if any check fails.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

def _find_root() -> Path:
    """Locate the repository root independently of where this script lives.

    The script has been committed at more than one path, so deriving the root
    from __file__ alone is unreliable: run from .github/workflows/ it would
    resolve to .github/ and report every artifact as missing.
    """
    for start in (Path(__file__).resolve().parent, Path.cwd().resolve()):
        for candidate in (start, *start.parents):
            if (candidate / ".git").exists():
                return candidate
            # Fallback marker for archive extracts with no .git directory.
            if (candidate / "Results").is_dir() and (candidate / "README.md").is_file():
                return candidate
    raise SystemExit(
        "Could not locate the repository root. Run this from inside a "
        "checkout of the repository."
    )


ROOT = _find_root()

# Vendored upstream data is not ours to validate.
SKIP_DIRS = {".git", "cache", "IS-Bench", "__MACOSX", ".ipynb_checkpoints"}

# Corrected headline values, pooled continuous-manipulation split (n = 475).
PAPER_HEADLINE = {
    "HAR": 96.4,
    "TSR": 97.2,
    "PVS": 1.1,
    "BDI": 4.1,
    "COR": 5.9,
    "EXQ": 55.5,
}
TOLERANCE = 0.05

failures: list[str] = []
warnings: list[str] = []


def walk(pattern: str):
    for path in sorted(ROOT.rglob(pattern)):
        if any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts):
            continue
        yield path


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def check_json() -> None:
    for path in walk("*.json"):
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            failures.append(f"malformed JSON: {rel(path)} ({exc})")


def check_csv() -> None:
    for path in walk("*.csv"):
        try:
            rows = list(csv.reader(path.open(encoding="utf-8", newline="")))
        except Exception as exc:
            failures.append(f"malformed CSV: {rel(path)} ({exc})")
            continue
        if not rows:
            failures.append(f"empty CSV: {rel(path)}")
        elif len(rows) == 1:
            warnings.append(f"CSV has a header but no data rows: {rel(path)}")


def check_notebooks() -> None:
    for path in walk("*.ipynb"):
        try:
            nb = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            failures.append(f"malformed notebook: {rel(path)} ({exc})")
            continue
        if "cells" not in nb:
            failures.append(f"notebook has no 'cells' key: {rel(path)}")


def check_headline_metrics() -> None:
    """Guard against republishing pre-correction numbers."""
    target = ROOT / "Results" / "ethosvla_final_results_30baselines.json"
    if not target.exists():
        failures.append(f"missing results file: {rel(target)}")
        return

    data = json.loads(target.read_text(encoding="utf-8"))
    combined = data.get("combined_avg")
    if combined is None:
        failures.append(f"{rel(target)}: no 'combined_avg' key")
        return

    for metric, expected in PAPER_HEADLINE.items():
        actual = combined.get(metric)
        if actual is None:
            failures.append(f"{rel(target)}: combined_avg missing {metric}")
            continue
        if abs(float(actual) - expected) > TOLERANCE:
            failures.append(
                f"{rel(target)}: combined_avg {metric} = {actual}, "
                f"paper reports {expected}"
            )

    if float(combined.get("PVS", -1)) == 0.0:
        failures.append(
            f"{rel(target)}: PVS is 0.0. The paper states this figure is "
            "impossible under an episode-level PVS definition given nonzero "
            "COR. This file predates that correction and must be regenerated."
        )

    # The pooled estimate must be an episode-level bootstrap over three
    # benchmarks, not an arithmetic mean over two.
    per_bench = [k for k in data if k.endswith("_avg") and k != "combined_avg"]
    if len(per_bench) == 2:
        a, b = (data[k] for k in per_bench)
        for metric in PAPER_HEADLINE:
            if metric not in a or metric not in b:
                continue
            mean = (float(a[metric]) + float(b[metric])) / 2
            if abs(mean - float(combined.get(metric, 1e9))) < 1e-9:
                warnings.append(
                    f"{rel(target)}: combined_avg {metric} equals the "
                    f"arithmetic mean of {' and '.join(per_bench)}. The paper "
                    "specifies episode-level bootstrap weighting over three "
                    "benchmarks (RLBench + ScanNet + DROID, n = 475)."
                )
                break

    if not any("droid" in k.lower() for k in data):
        failures.append(
            f"{rel(target)}: no DROID entries, but the pooled n = 475 "
            "estimate requires DROID's 100 episodes."
        )

    baselines = data.get("baselines")
    if isinstance(baselines, (list, dict)) and len(baselines) != 30:
        warnings.append(
            f"{rel(target)}: {len(baselines)} baselines present, "
            "paper compares against 30."
        )


def check_readme_paths() -> None:
    readme = ROOT / "README.md"
    if not readme.exists():
        failures.append("missing README.md")
        return
    text = readme.read_text(encoding="utf-8")
    for claimed in ("LICENSE", "LICENSE-BAAD"):
        if f"]({claimed})" in text and not (ROOT / claimed).exists():
            failures.append(
                f"README links to {claimed}, which does not exist in the repo."
            )


def main() -> int:
    check_json()
    check_csv()
    check_notebooks()
    check_headline_metrics()
    check_readme_paths()

    for warning in warnings:
        print(f"WARNING  {warning}")
    for failure in failures:
        print(f"FAIL     {failure}")

    if failures:
        print(f"\n{len(failures)} check(s) failed, {len(warnings)} warning(s).")
        return 1

    print(f"\nAll checks passed ({len(warnings)} warning(s)).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
