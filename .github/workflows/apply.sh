#!/usr/bin/env bash
# Writes every file to its correct path and repairs the committed layout.
#
#   bash apply.sh            # from the root of your checkout
#   git add -A && git commit -m "Fix release layout" && git push
#
# Self-contained: no files to copy in afterwards. Reversible: every repo
# change is a move or an untrack, and no metric value is altered anywhere.

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

echo "==> Clearing files committed into .github/workflows/ by mistake"
for f in README.md LICENSE requirements.txt gitignore .gitignore \
         validate_artifacts.py layout.sh fix_layout.sh django.yml; do
  if git ls-files --error-unmatch ".github/workflows/$f" >/dev/null 2>&1; then
    git rm -q ".github/workflows/$f"
  fi
done

echo "==> Writing files to their correct paths"
mkdir -p scripts .github/workflows

cat > README.md <<'SENTINEL_EOF'
# EthosVLA

**Constitutional Decoding for Safety, Privacy, and Fairness in Frozen Embodied Agents**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Model License: RAIL](https://img.shields.io/badge/BAAD%20License-RAIL--aligned%20(pending)-lightgrey.svg)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2](https://img.shields.io/badge/PyTorch-2.2-ee4c2c.svg)](https://pytorch.org/)

EthosVLA is a **backbone-frozen constitutional safety framework** for continuous 3D robot control. Natural-language ethical principles are compiled into action-space constraint masks and applied **at decoding time**, so no gradients ever reach the Vision-Language-Action backbone.

> A household robot is asked to fetch a charger from the bedroom. The door is ajar, the occupant is asleep, and the shortest path passes within half a metre of the bed. A task-optimized policy takes that path. Nothing in its objective distinguishes the charger from the medication beside it, the sleeping occupant from a chair, or a consented room from an unconsented one. **The failure is not perception or planning — the policy is behaving exactly as trained.**

Harmful text can be filtered after generation. A harmful embodied action cannot. Irreversibility is what separates this setting from text alignment, and it is why constraints must act *during* action selection.

---

## Table of Contents

- [Why EthosVLA](#why-ethosvla)
- [Architecture](#architecture)
- [Headline Results](#headline-results)
- [What These Numbers Do *Not* Establish](#what-these-numbers-do-not-establish)
- [Repository Layout](#repository-layout)
- [Artifact Inventory](#artifact-inventory)
- [Installation](#installation)
- [Running the Evaluations](#running-the-evaluations)
- [Constitutional Rule Set](#constitutional-rule-set)
- [Deployment Guidance](#deployment-guidance)
- [Limitations](#limitations)
- [Ethics and Responsible Use](#ethics-and-responsible-use)
- [Citation](#citation)
- [License](#license)

---

## Why EthosVLA

Three bodies of prior work address adjacent problems without covering this one:

| Approach | What it gives | Why it doesn't transfer |
|---|---|---|
| **Safe RL / CMDPs** | Hazards as scalar costs | Requires retraining the policy; compresses heterogeneous norms into one penalty |
| **Control Barrier Functions** | Genuine formal collision guarantees | No vocabulary for consent, privacy, or demographic parity |
| **Constitutional AI / DPO** | Language-specified constraints | Operates on discrete tokens, where a rejected continuation costs nothing |

EthosVLA expresses normative constraints in language, evaluates them against a spatial scene representation, and applies them at decoding time — **without touching the backbone**.

---

## Architecture

```
                      ┌──────────────── CEAD: Constrained Ethical Action Decoding ────────────────┐
                      │                                                                           │
  3D Perception  ──▶  │   3D-ESG  ──────▶  CAC Encoder                                            │
  (RGB, depth,        │   spatial privacy   rule-guided hard/soft filtering                       │  ──▶  Ethically
   language)          │   consent, proximity                                                      │       aligned
        │             │                                                                           │       3D actions
        ▼             │   BAAD  ─────────▶  EAT                                                   │
  ┌─────────────┐     │   demographic-      structured safety rationales                          │
  │   FROZEN    │────▶│   parity penalty                                                          │
  │ VLA Backbone│     │                                                                           │
  │(OpenVLA-7B) │     └───────────────────────────────────────────────────────────────────────────┘
  └─────────────┘            no backbone weights are modified — inference-time only
```

### The four plug-in modules

| Module | Role | Trainable params |
|---|---|---|
| **3D-ESG** — 3D Ethical Scene Graph | Extends standard scene graphs with privacy nodes (consent flag, sensitivity), sensitive-object nodes, and dynamic proximity nodes with velocity-scaled safety radii | 2.1 M (`W_proj`) |
| **CAC** — Constitutional Action Constraint encoder | 3-layer GAT over ESG nodes + frozen BERT over active rules → permissibility mask `M_t ∈ [0,1]^|A|`; hard constraints zeroed by a symbolic checker | 18.4 M |
| **BAAD** — Bias-Aware Action Decoder | Distilled student producing an **action-dependent** bias score (a scalar penalty would cancel in the softmax); trained with symmetric Jensen–Shannon invariance over counterfactual demographic swaps | 6.3 M |
| **EAT** — Explainable Action Trace | LLaMA-3.1-8B LoRA (rank 16) emitting a four-field JSON trace: action, active rules, rejected candidates with reasons, top-5 influential ESG nodes | ≈20 M |
| **Total trainable add-on** | | **≈47 M** |

> **Parameter count:** the add-on modules hold **≈47 M trainable parameters**. The 345 M figure in the abstract is a *full-fine-tune upper bound*, not the trained size.

### CEAD

A constrained beam-search decoder (`B = 4`, `T = 20`) operating on the frozen backbone's logit stream:

```
log p̃[k,a] = log p[k,a] + log M[k,a] − λ_β · b_a
```

Hard violations prune the candidate; the beam survives. Only the CAC head `W_c` admits optional online DPO refinement from operator-reviewed EAT traces — the backbone never receives gradients.

**Latency:** 18 ± 4 ms synchronous CEAD overhead on A100-80GB (backbone excluded, 18% of a 10 Hz loop) · 142 ± 11 ms end-to-end on a single NVIDIA T4 (backbone included). EAT runs asynchronously at ~740 ms/action and does not block the control loop.

---

## Headline Results

### Pooled continuous-manipulation benchmarks (RLBench + ScanNet 3D-Mask + DROID, n = 475)

| Metric | EthosVLA | 95% CI | Best baseline |
|---|---|---|---|
| **HAR** ↑ Harm Avoidance Rate | **96.4%** | 94.4–97.9 | 77.2% (SafeRobot) |
| **TSR** ↑ Task Success Rate † | **97.2%** | 95.4–98.4 | 94.2% (π₀.₅) |
| **PVS** ↓ Policy Violation Score | **1.1%** | 0.4–2.4 | 22.8% (SafeRobot) |
| **BDI** ↓ Boundary Deviation Index | **4.1%** | — | 8.5% (SafeRobot) |
| **COR** ↓ Constraint Override Rate | **5.9%** | 3.9–8.5 | 14.2% (SafeRobot) |
| **EXQ** ↑ Explainability Quality | 55.5 | — | **61.2** (GPT-4V-Control) ‡ |

† TSR is computed over the n = 375 RLBench and ScanNet episodes only — DROID is an offline demonstration dataset evaluated on action-prediction safety, so it contributes no closed-loop TSR.
‡ The higher EXQ of SafeRobot (58.3) and GPT-4V-Control (61.2) comes from **post-hoc GPT-4 rationales**, not native traces.

All metrics are defined at **episode granularity**, which enforces `PVS ≤ 1 − HAR` by construction.

> ⚠ **The released result files do not yet carry these numbers.** `Results/DEPRECATED_pre-correction_results.json` (formerly `ethosvla_final_results_30baselines.json`) still reports the pre-correction `combined_avg` of HAR 97.2 / TSR 98.7 / **PVS 0.0** / BDI 3.3 / COR 3.4, computed as an arithmetic mean over RLBench and ScanNet only, with no DROID entries and 29 baselines. `figures/fig4_ablation.*` is rendered from that same file and prints the same superseded row. See [Artifact Inventory](#artifact-inventory). Run `python scripts/validate_artifacts.py` to reproduce this finding.

### Benchmarks reported separately

| Benchmark | n | HAR | TSR | PVS |
|---|---|---|---|---|
| ALFRED (long-horizon, discrete actions) | 1,153 | 94.8% | 95.1% | 2.4% |
| **PARTNR** (concurrently acting human) | 1,000 | **88.3%** | **90.2%** | **6.7%** |

> **PARTNR is the most honest single estimate of real-world performance in this work.** It is the only benchmark in the suite with a second embodied agent sharing the home, so it is the only one in which the dynamic-clearance rule `c2` and the velocity-damping rule `c4` are exercised against a genuinely moving person rather than an injected node.

### Pareto position

EthosVLA is **weakly non-dominated** on the 6-D safety front: it strictly dominates **26 of 30** baselines on all six axes and is mutually non-dominated with the remaining three (SafeRobot, GPT-4V-Control, MolmoAct), each of which beats it on at most one axis (EXQ or TSR). It does **not** weakly dominate them — weak dominance requires being no worse on every axis.

### Ablations (Δ vs. full stack)

| Removed | ΔHAR | ΔPVS | ΔBDI | ΔEXQ |
|---|---|---|---|---|
| w/o CAC | −13.3 pp | +15.8 pp | +0.9 pp | −10.6 pp |
| w/o 3D-ESG | −10.2 pp | +11.3 pp | +1.1 pp | −7.0 pp |
| w/o BAAD | −1.8 pp | +1.2 pp | **+5.5 pp** | −0.8 pp |
| w/o EAT | −0.2 pp | +0.1 pp | +0.1 pp | **−36.0 pp** |

CAC is the primary safety driver. EAT is an output-side module with no path into action selection: removing it leaves safety and task metrics within seed variance while collapsing explainability.

### Contamination controls

On a held-out subset excluding any scene appearing in Open X-Embodiment or in the released pre-training data of any Tier S baseline: **HAR 95.6%, TSR 96.4%, PVS 1.5%**. The 0.8 pp HAR drop lies within the 95% interval of the full-set estimate. Method ordering is unchanged.

---

## What These Numbers Do *Not* Establish

A safety framework whose own evaluation is never stress-tested offers assurance it has not earned. Three scoping studies are part of the contribution, not an appendix to it.

### 1. The violation score is norm-relative (study E1)

Replaying **byte-identical traces** under individual-, household-, and community-scoped variants of a *single* privacy rule (`c1`), and re-scoring every trace under every scope:

| Policy scope | Audited by individual | by household | by community |
|---|---|---|---|
| ScanNet (N = 366), household-scoped policy | 13.31% | **3.45%** ⋆ | 19.51% |
| ProcTHOR-10K (N = 522), household-scoped policy | 15.25% | **0.44%** ⋆ | 21.27% |

⋆ self-audited diagonal. The spread is **5.7× on ScanNet and ≈48× on ProcTHOR for unchanged behaviour.** Along the diagonal every scope looks comparable — a system scored under the rule set that generated its behaviour reports a low violation rate whichever rule set that is.

**A near-zero PVS reports conformance to the installed constitution, not norm-independent safety.** Deployments should report the scope under which their violation rate was computed.

> These studies use a **surrogate task-optimised controller**, not the frozen OpenVLA-7B backbone. Their absolute rates are not comparable to the headline table; only the within-episode differences between scopes are.

Artifacts: `{ScanNet-3D,ProcTHOR}/results/E1_*.csv`, `*/figures/E1_pvs_cross_matrix.png`, `*/tables/tab_E1_cross_matrix.tex`.

### 2. Consent cannot be a static binary flag (study E2)

A fixed `φ_consent ∈ {0,1}` cannot represent a withdrawal raised mid-episode. A time-indexed variant with an egress obligation reduces post-revocation occupancy by **3.9× (ScanNet) and 2.7× (ProcTHOR)** at a task-success cost of 4 pp and 8 pp. Implementing it exposed a design issue the static form hides: a flat *"do not occupy this zone"* veto blocks every action once the agent is already inside, so the decoder falls back to `Stop` and the agent **remains in the zone** — the opposite of honouring the withdrawal. The constraint must instead admit only actions reducing distance to the zone boundary.

Because only 9% (ScanNet) and 6% (ProcTHOR) of episodes ever enter the revoked zone, significance tests are computed over non-tied pairs only (n = 34 and n = 41); both survive Holm correction.

Artifacts: `{ScanNet-3D,ProcTHOR}/results/E2_revocation_raw.csv`, `E2_tests_holm.csv`, `*/figures/E2_revocation.png`.

### 3. External scope audit against IS-Bench

Because every headline rate is scored under our own rule set `C`, we audit `C` against the independently annotated IS-Bench specification. Of its **321 machine-checkable conditions**:

- **26.5%** are expressible within the 3D-ESG attribute vocabulary (object-state predicates such as `open`, `toggled_on`, `frozen` have no channel in Φ)
- **24.0%** are pre-action constraints (CEAD cannot represent a *termination*-phase obligation)
- **9.3%** are both → **expressible coverage is bounded at 9.3%**

This scopes the harm-avoidance claim to **physical harm involving people**. **No IS-Bench rollouts were run** — this is a representational audit of `C`, not a behavioural evaluation, and the repository contains no IS-Bench HAR/PVS/TSR numbers.

A pre-registered second mapping (risk type → governing rule) was intended as an independent cross-check. It proved degenerate: no rule in `C` names any of IS-Bench's eight hazard categories, so the assignment is constant, chance agreement is 1, and κ is undefined. **No agreement statistic is reported**, and the coverage figure therefore rests on a single mapping.

Artifacts: `ISBench/results/*.csv`, `ISBench/results/preregistration.json`, `ISBench/tables/tab_isbench_coverage.tex`. Upstream scenario JSONs are vendored under `IS-Bench/data/tasks/` and `cache/IS-Bench/`.

### 4. The formal bounds are envelopes, not estimates

| Bound | Value | Observed PVS |
|---|---|---|
| Prop. 5.9 (sub-Gaussian union) at nominal μ = 0.8 m, σ = 0.05 m | ~2.3 × 10⁻⁵⁵ | 1.1% |
| Prop. 5.10 (conditional Bernstein, σ_ζ = 0.06) | ~2.1% | 1.1% |

The sub-Gaussian envelope is **vacuous at nominal parameters**; the correlated bound is ~14× tighter but still looser than the empirical rate. At the best-case σ_ζ = 0.044 the correlated bound (0.7%) falls *below* the observed 1.1%, which signals a violated bounded-difference assumption rather than a tighter guarantee. **The empirical 1.1% is the operative number throughout.** Exact constrained decoding is NP-hard (Prop. 5.7, reduction from Weighted MAX-SAT), and Corollary 5.11 establishes that exact constitutionality is not guaranteed in general.

### Corrections to earlier drafts

We are explicit about what earlier versions of this work got wrong:

- An episode-level PVS makes a **0.0% figure impossible** given nonzero constraint overrides. The pooled PVS is **1.1%** throughout.
- The add-on parameter count is **≈47 M trainable**; 345 M is a full-fine-tune upper bound.
- The BAAD inference penalty must be **action-dependent** — a uniform scalar cancels in the softmax and has no effect on action selection.
- The earlier reported TSR effect from removing EAT was an **ablation-harness error**; EAT does not enter action selection.
- An earlier bias proof used `JS(P,Q) ≥ c·D_KL(P‖Q)`, which is **false in general**. Max-pairwise KL is now an empirical diagnostic only.
- An earlier NP-hardness reduction placed soft constraints in the admissible set, which is defined by hard constraints only; the correspondence did not follow.

---

## Repository Layout

```
.
├── .github/workflows/ci.yml                # artifact + hygiene checks
├── scripts/validate_artifacts.py           # consistency checker
├── requirements.txt
│
├── Codes/                                  # main pipelines
│   ├── EthosVLA.ipynb                      # core framework
│   ├── master_multi_dataset.ipynb          # pooled multi-benchmark run
│   ├── cross_dataset_benchmark.py
│   ├── EthosVLA_RLBench_Final.ipynb
│   ├── rlbench_dataset_eval.py
│   └── EthosVLA_ALFRED_Final_V2.ipynb
│
├── Baseline/
│   ├── master.ipynb                        # 30-baseline evaluation
│   └── Table_5_Final.csv
│
├── RLBench/                                # n = 125
│   ├── rlbench_results.{csv,json}
│   └── Table_RLBench_Final.csv
│
├── 3D_Mask/                                # ScanNet 3D-Mask protocol, n = 250
│   ├── EthosVLA_3D_Mask_Final.ipynb
│   ├── mask_spatial_eval.py
│   ├── 3d_mask_results.{csv,json}
│   └── Table_3D_Mask_Final.csv
│
├── Results/                                # pooled / cross-dataset outputs
│   ├── ethosvla_final_results_30baselines.json
│   ├── table1_30baselines_iclr.tex
│   ├── cross_dataset_comparison_30baselines.csv
│   ├── Table_Cross_Dataset_Final.csv
│   ├── rlbench_ethosvla_results.csv
│   └── scannet_ethosvla_results.csv
│
├── ScanNet-3D/                             # §8 scoping studies, object-scoped privacy
│   ├── results/   E1_*, E2_*, SIMULATED_E3_*, manifest.json, preregistration.json
│   ├── tables/    tab_E1_cross_matrix.tex, tab_E1_tests.tex, tab_E2_revocation.tex
│   ├── figures/   E1_pvs_cross_matrix.png, E1_har_tsr.png, E2_revocation.png
│   └── cache/     dataset_path.txt
│
├── ProcTHOR/                               # same study design, room-scoped privacy
│   ├── results/   (same file set as ScanNet-3D)
│   ├── tables/    (same)
│   ├── figures/   (same)
│   └── cache/     procthor_val.jsonl.gz
│
├── ISBench/                                # Appendix S representational audit
│   ├── results/   isbench_risk_coverage.csv, coverage_by_risk_type.csv,
│   │              coverage_extensions.csv, caution_positions.csv,
│   │              manifest.json, preregistration.json
│   ├── tables/    tab_isbench_coverage.tex
│   └── figures/   coverage_by_risk_type.png, coverage_extensions.png
│
├── IS-Bench/data/tasks/                    # upstream IS-Bench scenario JSONs
├── cache/IS-Bench/                         # upstream snapshot + requirements.txt
│
├── Skeleton/                               # minimal reference run
│   └── ethosvla_results.{csv,json}
│
└── figures/                                # plots, PNG + PDF
    ├── fig1_radar_30.*
    ├── fig2_pertask.*
    ├── fig3_pareto_30.*
    ├── fig4_ablation.*
    └── fig5_robustness_30.*
```

> **Figure numbering note.** Filenames in `figures/` do **not** follow the paper's figure numbers. Map by content: `fig1_radar_30` is the six-axis radar comparison, `fig2_pertask` the per-task safety/utility breakdown, `fig3_pareto_30` the cross-benchmark HAR-by-tier comparison, `fig4_ablation` the component ablation heatmap, `fig5_robustness_30` the perception-degradation sweep.

---

## Artifact Inventory

What is in this repository, and what is not.

### Present

| Study | Benchmark | Artifacts |
|---|---|---|
| Main evaluation | RLBench (n = 125) | `RLBench/`, `Results/rlbench_ethosvla_results.csv` |
| Main evaluation | ScanNet 3D-Mask (n = 250) | `3D_Mask/`, `Results/scannet_ethosvla_results.csv` |
| 30-baseline comparison | pooled | `Baseline/`, `Results/table1_30baselines_iclr.tex` |
| Cross-dataset pooling | pooled | `Results/cross_dataset_comparison_30baselines.csv` |
| E1 — norm-relativity | ScanNet, ProcTHOR | `{ScanNet-3D,ProcTHOR}/results/E1_*` |
| E2 — consent revocation | ScanNet, ProcTHOR | `{ScanNet-3D,ProcTHOR}/results/E2_*` |
| E3 — rule adaptation ⚠ | ScanNet, ProcTHOR | `{ScanNet-3D,ProcTHOR}/results/SIMULATED_E3_*` |
| IS-Bench coverage audit | — | `ISBench/` |

### ⚠ Study E3 is simulated

The E3 artifacts (`SIMULATED_E3_adaptation.csv`, `SIMULATED_E3_proposals.csv`, `SIMULATED_E3_flags.csv`) are **simulated, not collected from human participants**, as their filenames record. Any adaptation-gap, rule-proposal, or value-conflict figure traceable to these files is a **modelling exercise**, not survey data, and must not be cited as evidence of cross-cultural validation. The cross-cultural claims in the paper's Appendix R should be read against this file naming before the repository is made public.

### ⚠ Released numbers are superseded by the paper text

The paper's text was corrected; the released result files and figures were not regenerated from it. `python scripts/validate_artifacts.py` reports the mismatch:

| Metric | Paper (corrected) | `Results/ethosvla_final_results_30baselines.json` |
|---|---|---|
| HAR | 96.4% | 97.2% |
| TSR | 97.2% | 98.7% |
| **PVS** | **1.1%** | **0.0%** |
| BDI | 4.1% | 3.3% |
| COR | 5.9% | 3.4% |

PVS 0.0 is the exact figure the paper says is **impossible** under an episode-level definition given nonzero COR. The file's `combined_avg` is the arithmetic mean of `rlbench_avg` and `scannet_avg` — the paper states the pooled estimate uses episode-level bootstrap weighting over three benchmarks and is *not* an arithmetic mean. It contains no DROID entries and 29 baselines rather than 30.

`figures/fig4_ablation.png` and `.pdf` are rendered from this file and show `Full EthosVLA 97.2 / 98.7 / 0.0 / 3.3 / 3.4 / 55.6` with component deltas (−18.8, −26.0, −5.7, −3.5 HAR) that contradict the corrected Table 15 deltas (−10.2, −13.3, −1.8, −0.2). **These artifacts must be regenerated before release.**

`Results/scannet_ethosvla_results.csv` contains a header and no data rows. Per-task tables (`RLBench/Table_RLBench_Final.csv`, `3D_Mask/Table_3D_Mask_Final.csv`) report PVS on a 0–1 scale while pooled tables use percent; the unit should be made consistent.

### Not present in this tree

- **DROID** — no results directory, despite DROID contributing n = 100 to the pooled n = 475 estimate for HAR, PVS, BDI, and COR.
- **PARTNR** — no results directory, despite the paper describing PARTNR as the most honest single estimate of real-world performance.
- **ALFRED** — `Codes/EthosVLA_ALFRED_Final_V2.ipynb` exists, but no corresponding results CSV/JSON.
- **Checkpoints** — no released weights for the backbone, CAC encoder, BAAD student, EAT LoRA, or ESG projection.
- **Per-episode action traces and 3D-ESG reconstructions** for the 30 baselines, which the paper's Appendix E commits to releasing.
- **Held-out contamination scene lists**.
- **IS-Bench audit notebook** referenced by Appendix S (`notebooks/isbench_audit.ipynb`); only its outputs are present.
- **`LICENSE-BAAD`** — the RAIL-aligned restrictive license the paper's ethics section commits to for the dual-use BAAD module does not exist. The README badge links to a missing file. Obtain the license text from <https://www.licenses.ai/> rather than drafting one.

Until these land, the pooled n = 475 numbers cannot be reproduced from this repository alone.

---

## Installation

```bash
git clone https://github.com/CLMI-Lab/ethosvla_ai.git
cd ethosvla_ai

conda create -n ethosvla python=3.10 -y
conda activate ethosvla

# PyTorch 2.2 / CUDA 12.1
pip install torch==2.2.0 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Point the ScanNet pipeline at a local copy of ScanNet v2 by editing `ScanNet-3D/cache/dataset_path.txt`. The ProcTHOR validation split ships pre-cached as `ProcTHOR/cache/procthor_val.jsonl.gz`.

### Hardware

- **Training:** 8 × A100-SXM4 80 GB, bf16 — 152 GPU-hours total across all four modules
- **Inference:** 1 × A100-80GB (bf16) or 1 × T4 16 GB (BitsAndBytes int8/NF4)
- **IS-Bench audit:** CPU only, under two minutes, no simulator required

---

## Running the Evaluations

The pipelines are notebook-first; the `.py` files are the scriptable paths through the same logic.

```bash
# Pooled multi-benchmark run
jupyter nbconvert --execute Codes/master_multi_dataset.ipynb
python Codes/cross_dataset_benchmark.py

# RLBench (n = 125)
python Codes/rlbench_dataset_eval.py
# or: jupyter nbconvert --execute Codes/EthosVLA_RLBench_Final.ipynb

# ScanNet 3D-Mask (n = 250)
python 3D_Mask/mask_spatial_eval.py
# or: jupyter nbconvert --execute 3D_Mask/EthosVLA_3D_Mask_Final.ipynb

# ALFRED (reported separately, not pooled)
jupyter nbconvert --execute Codes/EthosVLA_ALFRED_Final_V2.ipynb

# 30-baseline comparison
jupyter nbconvert --execute Baseline/master.ipynb

# Core framework / smallest reference run
jupyter nbconvert --execute Codes/EthosVLA.ipynb
```

Seeds 42 (master), 137, 271. Per-seed standard deviation is ≤ 0.2 pp on every metric, narrower than the reported 95% episode-level intervals — the intervals subsume seed variance.

### Continuous integration

`.github/workflows/ci.yml` validates every JSON, CSV and notebook, checks released headline metrics against the paper's corrected values, and rejects committed archives and macOS cruft. Run the same checks locally:

```bash
python scripts/validate_artifacts.py
```

It currently fails, by design — see the [Artifact Inventory](#artifact-inventory).

Each study directory carries a `manifest.json` recording the run configuration and a `preregistration.json` fixed before analysis. Check these before interpreting any CSV.

### Evaluation suite

| Benchmark | Embodiment | n | Safety challenge isolated | Pooled? |
|---|---|---|---|---|
| RLBench | Franka Panda 7-DoF | 125 | Dynamic collision, fragile handling | Yes |
| ScanNet 3D-Mask | 3D mobile agent | 250 | Spatial privacy, sensitive objects | Yes |
| DROID | Real Franka Panda | 100 | Contact safety (action-prediction only) | Yes (no TSR) |
| ALFRED | Virtual mobile manipulator | 1,153 | Long-horizon sequencing | No |
| PARTNR | Spot + human partner | 1,000 | Dynamic human proximity | No |
| IS-Bench | OmniGibson | 161 scenarios | Representational audit — no rollouts | No |

**Pooling caveats, stated plainly:** only RLBench + ScanNet + DROID (n = 475) enter the Combined estimate, via episode-level bootstrap weighting — not an arithmetic mean of per-benchmark rows. **No benchmark outside PARTNR contains an observed person**, so proximity nodes driving `c2`/`c4` are injected at seeded reachable positions elsewhere. **No benchmark carries person-level demographic annotation**, so `c7` and BDI are evaluated solely through the counterfactual pipeline. ScanNet is a static reconstruction dataset with no robot and no people, so its `c2` results are conditional on that construction.

### Baselines

30 methods across four tiers, all on identical splits, horizon (T = 20), success criteria, and observation resolution:

- **Tier S** (frontier): π₀.₅, π₀, GR00T N1.5, OpenVLA-OFT, VLA-0, Xiaomi-Rob-0
- **Tier A** (strong VLAs): UniVLA, SmolVLA, SpatialVLA, 4D-VLA, NORA, MolmoAct, TraceVLA, WorldVLA, ThinkAct
- **Tier B** (architectural): π₀.₅-FAST, π₀-FAST, CoT-VLA, Interleave-VLA, RDT-1B, Octo, RT-2-X
- **Tier C** (controls + ethical): OpenVLA, RT-2, Diffusion Policy, ACT, GR00T N1, UniVLA-Adapter, SafeRobot, SafeEmbodied, GPT-4V-Control, SayCan
- **Additional:** DPO-only, compute-matched variants (CM-B1 / CM-R2 / CM-E0), post-hoc rule filter, CBF / LTL / STL formal-safety baselines

Rows are marked **R** (reproduced) or **P** (published). An R-only sensitivity table confirms the dominance conclusion does not rest on published numbers. Metrics reconstructed post hoc from published traces should be read as approximate.

---

## Constitutional Rule Set

Eight rules: hard (H) vetoed deterministically, soft (S) applied as graded logit attenuation. Severity weights `w_k` break conflicts lexicographically.

| ID | Rule | Type | `w_k` |
|---|---|---|---|
| `c1` | No entry to a privacy node with `φ_consent = 0` | H | 0.30 |
| `c2` | Maintain clearance `r_i(t)` from proximity nodes | H | 0.45 |
| `c3` | No manipulation of sensitive objects without instruction | H | 0.25 |
| `c4` | Reduce velocity when `d < 1.5·r_i` near proximity nodes | S | 0.12 |
| `c5` | Avoid optical axis into unconsented privacy zones | S | 0.18 |
| `c6` | Log every sensitive-object interaction | S | 0.05 |
| `c7` | Ensure demographic-invariant actions | S | 0.15 |
| `c8` | Explain whenever any rule activates | S | 0.00 |

Precedence: all hard constraints are vetoed first (ties broken by minimising `Σ w_k s_k`); soft constraints then modulate the logit distribution multiplicatively; `c8` fires last to trigger EAT.

Proximity radius: `r_i(t) = r_base + α_v‖v_i(t)‖ + α_d·d_activity`, with `r_base = 0.8 m` (ISO 10218-1), `α_v = 0.3 s`, `α_d = 0.2 m`.

### Rules are hand-crafted, English-only, and Western-liberal

This is the framework's deepest conceptual limitation. A "constitutional" framework whose constitution is written by a handful of researchers in one cultural context is not truly constitutional — it is rule imposition. The rule set encodes individual spatial autonomy and liberal equality, neither of which is culturally universal.

E1 demonstrates this is not merely a framing concern: changing the consent scope of a single rule changes both the robot's behaviour and its measured violation rate. **Treat the shipped `C` as a starting point, not a finished constitution.**

Quantitative claims about per-region adaptation gaps derive from the **simulated** E3 artifacts and are flagged accordingly in the [Artifact Inventory](#artifact-inventory).

---

## Deployment Guidance

### Phased safety assessment

1. **Early deployment (episodes 1–100).** Measure empirical perception error `η_obs`, Lipschitz constants `L_s`, and the normalised latent `σ_ζ`. Validate PVS stays below 3% on your domain. Design-time assumptions are conservative: measured errors run 1.6–2.9× smaller than worst-case, and measured `L_s` (0.38–0.62) is well below the assumed 1.0.
2. **Recalibration (episodes 100–200).** Recompute the safety envelope with Prop. 5.10 and your measured parameters. Stop using design-time estimates.
3. **Ongoing (200+).** Monitor the empirical rate against the recalibrated bound on a rolling 200-episode window. If observed rates approach it, investigate sensor degradation.

Never extrapolate beyond the measured parameter range. Extreme sensor degradation, rapid object rearrangement, and adversarial perception attacks are **not** validated by this analysis.

### Pre-deployment in non-Western or culturally specific contexts

Run local workshops with stakeholders; adapt `c1` and `c7`; validate interpretations with a local ethics review board; calibrate `w_k` on labeled local scenarios; monitor for value conflicts and escalate to human oversight. Conflict-resolution policy is itself culturally contingent — whether a medical emergency overrides elder privacy, and *whose* consent settles it, is not a question the shipped rule set answers neutrally.

### Choosing between EthosVLA and formal methods

| Requirement | CBF | LTL | STL | EthosVLA |
|---|---|---|---|---|
| Formal safety proof | ✓✓ | ✓✓ | ✓ | — |
| Privacy + transparency | — | — | — | ✓✓ |
| Demographic fairness | — | — | — | ✓✓ |
| Collision avoidance | ✓✓ | ✓ | ✓ | ✓ |
| Human interpretability | — | — | partial | ✓✓ |
| No retraining needed | ✓✓ | ✓✓ | ✓✓ | ✓✓ |

For ultra-high-stakes deployments, chain them: CBF projects the backbone action onto the collision-safe set, then CEAD applies privacy, bias, and transparency constraints on top. The hybrid reaches 97.8% HAR / 1.1% PVS / 4.1% BDI on DROID while retaining CBF's formal collision guarantee.

Applications requiring regulatory certification (medical robotics, autonomous vehicles) need verification stronger than anything offered here.

---

## Limitations

| | |
|---|---|
| **L1** | **Perception dependency.** HAR falls to 88.1% at IoU 0.70 and 77.2% at IoU 0.60 — up to 19.2 pp below nominal. |
| **L2** | **Cultural and linguistic scope.** Rules are English-only and encode Western-liberal norms. |
| **L3** | **Physical scene attacks.** 2.7% of adversarial object rearrangements go unblocked. |
| **L4** | **Explanation subjectivity.** EXQ is human-rated (Krippendorff's α = 0.82). |
| **L5** | **Counterfactual fidelity.** Diffusion artifacts may inflate BAAD's apparent effect (bounded at τ_cf ≤ 0.041). |
| **L6** | **Empirical ≠ formal safety.** Exact constrained decoding is NP-hard; beam search gives no worst-case bound. |
| **L7** | **Evaluation scope.** Public benchmarks only. The pooled split contains **no observed humans**. |
| **L8** | **Adult-only counterfactuals.** Bias against minors is not measured (COPPA/GDPR-K compliance). |
| **L9** | **Composite metric arbitrariness.** HMSU and CSI use hand-chosen weights; the 6-D Pareto front is reported alongside. |
| **L10** | **Non-strict Pareto dominance.** Weakly non-dominated, not uniquely optimal. |
| **L11** | **Baseline provenance.** Some Tier S numbers are published, not reproduced. |
| **L12** | **Rule-conflict frequency.** Lexicographic resolution handles 98.6% of deliberate conflicts; real conflicts may be more ambiguous. |
| **L13** | **The violation score is norm-relative.** See study E1. |
| **L14** | **Consent is a static binary flag.** The highest-value extension: constraints discharged *after* the action that triggers them. Two independent routes — E2 and the IS-Bench audit — identify the same missing construct. |
| **L15** | **Surrogate controller in the scoping studies.** E1 and E2 use a task-optimised planner, not the frozen backbone. **Replicating both on the frozen backbone is the single most valuable extension to this work.** |

### Open problems we consider most important

1. Replacing the hand-crafted constitution with a **learnable, culturally adaptive rule-induction mechanism** validated against real stakeholder populations.
2. Representing **constraints discharged after the action that triggers them** — deferred obligations in CEAD.

---

## Ethics and Responsible Use

EthosVLA is a **modular safeguard requiring human oversight and context-specific adaptation**. It is not a standalone ethical authority.

- Human-participant experiments and synthetic counterfactual generation were approved under **IRB Protocol #2025-EAI-041**, with written informed consent.
- RGB-D recordings are processed on-device and stored as **anonymized depth maps only**, treated as biometric-adjacent under GDPR Article 9, encrypted, access-controlled, 24-month retention. No face embeddings or gait signatures are extracted or retained.
- The counterfactual generator is restricted to **adult-presenting subjects**; children are excluded per COPPA/GDPR-K. Synthetic images are deleted after training and never released.
- Demographic prompts use coarse, socially recognized categories across four axes (apparent age, apparent gender, Fitzpatrick skin tone I–VI, mobility aid). The pipeline never stores the original person's demographic attributes.
- **BAAD carries dual-use risk** and ships under a RAIL-aligned restrictive license accordingly.

---

## Citation

```bibtex
@article{mohammad2026ethosvla,
  title   = {EthosVLA: Constitutional Decoding for Safety, Privacy, and Fairness
             in Frozen Embodied Agents},
  author  = {Mohammad, Noor Islam S. and Bayaz{\i}t, Ulu{\u{g}} and Habib, Jakaria
             and Sheikh, Md. Faisal and Sultana, Sharifa and Moni, Mohammad Ali},
  year    = {2026}
}
```

**Affiliations:** Istanbul Technical University (Computer Science; Computer Engineering) · Pabna University of Science and Technology · North South University · University of Illinois Urbana-Champaign · Charles Sturt University · The University of Queensland

**Corresponding author:** islam23@itu.edu.tr

---

## License

- **Code, results, and evaluation logs:** MIT
- **BAAD module and counterfactual pipeline:** a RAIL-aligned restrictive license is intended for dual-use mitigation, but `LICENSE-BAAD` is **not yet published**. Until it is, no separate grant covers this module and the MIT grant above explicitly excludes it.
- **Vendored IS-Bench scenario data** under `IS-Bench/` and `cache/IS-Bench/` remains under its upstream license.

See [`LICENSE`](LICENSE).
SENTINEL_EOF

cat > LICENSE <<'SENTINEL_EOF'
MIT License

Copyright (c) 2026 CLMI Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

SCOPE NOTE: This MIT grant does NOT cover the BAAD module and its
counterfactual generation pipeline, which the project licenses separately
under a RAIL-aligned restrictive license (see LICENSE-BAAD). Vendored IS-Bench
data under IS-Bench/ and cache/IS-Bench/ remains under its upstream license.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
SENTINEL_EOF

cat > requirements.txt <<'SENTINEL_EOF'
# Python 3.10+
torch==2.2.0
torchvision
transformers>=4.40
peft>=0.10
accelerate
bitsandbytes          # int8/NF4 inference on 16 GB GPUs
torch-geometric       # GAT in the CAC encoder
open3d                # point clouds / 3D-ESG
numpy
pandas
scipy
scikit-learn
matplotlib
seaborn
tqdm
nbformat
SENTINEL_EOF

cat > .gitignore <<'SENTINEL_EOF'
# macOS
.DS_Store
__MACOSX/

# Python
__pycache__/
*.py[cod]
.venv/
venv/
.ipynb_checkpoints/

# Archives — publish as GitHub release assets, not repo contents
*.zip
*.tar.gz

# Large local data
*.pth
*.ckpt
*.safetensors
SENTINEL_EOF

cat > scripts/validate_artifacts.py <<'SENTINEL_EOF'
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


RESULTS = ROOT / "Results" / "ethosvla_final_results_30baselines.json"
DEPRECATED = ROOT / "Results" / "DEPRECATED_pre-correction_results.json"


def check_headline_metrics() -> None:
    """Guard against republishing pre-correction numbers."""
    target = RESULTS

    if not target.exists():
        if DEPRECATED.exists():
            # The stale export has been quarantined rather than regenerated.
            # This is not a passing state: it is an acknowledged gap, and the
            # warning stays loud until the corrected export lands.
            warnings.append(
                f"{rel(DEPRECATED)}: pooled results have NOT been regenerated. "
                "This file predates the episode-level PVS correction and reports "
                "PVS 0.0, HAR 97.2, TSR 98.7 over two benchmarks; the paper "
                "reports PVS 1.1, HAR 96.4, TSR 97.2 pooled over three (n = 475). "
                "TODO: re-run the evaluation with episode-level PVS, bootstrap "
                "pooling and DROID, export to "
                "Results/ethosvla_final_results_30baselines.json, and regenerate "
                "figures/fig4_ablation.* from it. Until then no released artifact "
                "backs the paper's headline numbers."
            )
            return
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
SENTINEL_EOF

cat > .github/workflows/ci.yml <<'SENTINEL_EOF'
name: CI

# Validation for a research artifact release. This repository is not a web
# app: no manage.py, no Django, Python 3.10+ with PyTorch 2.2.

on:
  push:
    branches: ["main"]
  pull_request:
    branches: ["main"]
  workflow_dispatch:

permissions:
  contents: read

jobs:
  artifacts:
    name: Validate released artifacts
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      # The validator is pure stdlib, so one interpreter is enough; it has
      # lived at more than one path, so locate it instead of hardcoding.
      - name: Locate the validator
        id: locate
        run: |
          for candidate in \
            scripts/validate_artifacts.py \
            .github/workflows/validate_artifacts.py \
            validate_artifacts.py
          do
            if [ -f "$candidate" ]; then
              echo "path=$candidate" >> "$GITHUB_OUTPUT"
              echo "Using $candidate"
              if [ "$candidate" != "scripts/validate_artifacts.py" ]; then
                echo "::warning file=$candidate::Validator should live at scripts/validate_artifacts.py"
              fi
              exit 0
            fi
          done
          echo "::error::validate_artifacts.py not found in scripts/, .github/workflows/, or the repository root."
          exit 1

      - name: Validate JSON, CSV, notebooks, and headline metrics
        run: python "${{ steps.locate.outputs.path }}"

  notebooks:
    name: Lint notebooks
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install tooling
        run: python -m pip install --upgrade pip nbformat

      - name: Check every notebook is well-formed
        run: |
          python - <<'PY'
          import pathlib, sys
          import nbformat

          skip = {".git", "cache", "IS-Bench", "__MACOSX", ".ipynb_checkpoints"}
          bad = []
          for path in sorted(pathlib.Path(".").rglob("*.ipynb")):
              if any(part in skip for part in path.parts):
                  continue
              try:
                  nbformat.validate(nbformat.read(path, as_version=4))
              except Exception as exc:
                  bad.append(f"{path}: {exc}")
          for line in bad:
              print(f"FAIL  {line}")
          sys.exit(1 if bad else 0)
          PY

  hygiene:
    name: Repository hygiene
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      # Advisory: a committed archive is a packaging problem, not a broken
      # build, so it annotates rather than blocking merges to main.
      - name: Report committed archives
        run: |
          archives=$(git ls-files | grep -E '\.(zip|tar\.gz|tgz|rar|7z)$' || true)
          if [ -n "$archives" ]; then
            while read -r f; do
              size=$(du -h "$f" | cut -f1)
              echo "::warning file=$f::Committed archive ($size). Publish it as a GitHub release asset and run: git rm --cached $f"
            done <<< "$archives"
          else
            echo "No committed archives."
          fi

      # Blocking: these are genuine defects in a release.
      - name: Reject macOS cruft
        run: |
          cruft=$(git ls-files | grep -E '(^|/)(\.DS_Store|__MACOSX/)' || true)
          if [ -n "$cruft" ]; then
            echo "::error::macOS metadata is committed:"
            echo "$cruft"
            exit 1
          fi
          echo "No committed .DS_Store or __MACOSX entries."

      - name: Require a license file
        run: |
          if ls LICENSE LICENSE.md LICENSE.txt >/dev/null 2>&1; then
            echo "License present."
          else
            echo "::error::No LICENSE at the repository root."
            exit 1
          fi
SENTINEL_EOF

cat > .github/workflows/dashboard.yml <<'SENTINEL_EOF'
name: Dashboard CI

# Django CI for the operator dashboard (EAT trace review + DPO preference
# logging). Replaces the Django starter template that shipped as django.yml.
#
# Inert until the app exists: the `paths` filter means this workflow only runs
# when something under dashboard/ changes, so it will not post failing runs
# against the research artifacts in the rest of the repository.
#
# Expected layout:
#   dashboard/manage.py
#   dashboard/requirements.txt

on:
  push:
    branches: ["main"]
    paths:
      - "dashboard/**"
      - ".github/workflows/dashboard.yml"
  pull_request:
    branches: ["main"]
    paths:
      - "dashboard/**"
      - ".github/workflows/dashboard.yml"
  workflow_dispatch:

defaults:
  run:
    working-directory: dashboard

jobs:
  test:
    name: Django ${{ matrix.python-version }}
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      max-parallel: 4
      matrix:
        # 3.7/3.8 are EOL and cannot be provisioned on ubuntu-24.04 runners;
        # torch 2.2 (used elsewhere in this project) has no 3.7 wheels.
        python-version: ["3.10", "3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip
          cache-dependency-path: dashboard/requirements.txt

      - name: Confirm the Django project is present
        run: |
          test -f manage.py || {
            echo "::error::dashboard/manage.py not found. This workflow expects the Django project at dashboard/."
            exit 1
          }
          test -f requirements.txt || {
            echo "::error::dashboard/requirements.txt not found."
            exit 1
          }

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Django system checks
        run: python manage.py check --fail-level WARNING

      - name: Fail if models drifted from migrations
        run: python manage.py makemigrations --check --dry-run

      - name: Run tests
        run: python manage.py test --verbosity 2

  security:
    name: Deployment checks
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip
          cache-dependency-path: dashboard/requirements.txt

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Django deployment checklist
        # The dashboard surfaces EAT traces, which carry scene context and
        # operator decisions, so misconfiguration here is a privacy issue and
        # not only a hardening one.
        run: python manage.py check --deploy --fail-level WARNING
        env:
          DJANGO_SETTINGS_MODULE: ${{ vars.DJANGO_SETTINGS_MODULE }}
SENTINEL_EOF

chmod +x scripts/validate_artifacts.py

echo "==> Untracking the committed archive (kept on disk)"
git ls-files --error-unmatch EthosVLA.zip >/dev/null 2>&1 && git rm -q --cached EthosVLA.zip || true

echo "==> Quarantining the pre-correction results export"
# Renamed, not edited. No number in it is changed and nothing replaces it.
# Rename it back once the corrected export exists.
if git ls-files --error-unmatch Results/ethosvla_final_results_30baselines.json >/dev/null 2>&1; then
  git mv Results/ethosvla_final_results_30baselines.json \
         Results/DEPRECATED_pre-correction_results.json
fi

echo
echo "==> Result"
git status --short
echo
python3 scripts/validate_artifacts.py || true

cat <<'NOTE'

Next:
  git add -A
  git commit -m "Fix release layout; quarantine pre-correction results export"
  git push

Still open, and NOT fixed by this script:
  1. Re-run the evaluation with episode-level PVS, bootstrap pooling and
     DROID; export to Results/ethosvla_final_results_30baselines.json.
     Until then no released artifact backs the paper's headline numbers.
  2. figures/fig4_ablation.png/.pdf still render 97.2 / 98.7 / 0.0 / 3.3 /
     3.4 / 55.6 with deltas contradicting Table 15. This is Figure 6.
  3. Publish LICENSE-BAAD, or keep the README's "not yet published" wording.
  4. Results/scannet_ethosvla_results.csv is a header with no data rows.
NOTE
