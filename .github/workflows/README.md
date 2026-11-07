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
