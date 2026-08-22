# Information-Geometric Context Window Governance and the Probabilistic Theory of Long-Context Collapse in Large Language Models  

### A Unified Framework: Observer Entropy, Extreme Value Theory, and the CPL 4.24 Phase-Aware Governor  

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22056414.svg)](https://doi.org/10.5281/zenodo.22056414)  
[![Dashboard](https://img.shields.io/badge/Dashboard-Live-10b981?style=flat-square)](https://khomyakov-vladimir.github.io/llm-context-window-governance/)  
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/) — Scientific article and associated documentation  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) — Source code and simulation scripts  
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)  

**Version 4.24** | Zenodo  

> Khomyakov, V. (2026). *Information-Geometric Context Window Governance and the Probabilistic Theory of Long-Context Collapse in Large Language Models* (4.24). Zenodo. https://doi.org/10.5281/zenodo.22056414  

---

## 🔬 Interactive Dashboard (Live Simulation)  

An interactive CPL Context Governor simulation is available online:  

👉 **https://khomyakov-vladimir.github.io/llm-context-window-governance/**  

The dashboard implements the discrete-time governor dynamics (context length, entropy, and stability updates under the phase-classifier policy) and allows real-time parameter exploration (σ, ρ, α_tight, κ_L, γ, T). It reflects the CPL simulator baseline hosted in this repository; see the note under [Repository Contents](#repository-contents) regarding which `simulator.py` state the dashboard currently tracks.

---

## Abstract  

This paper presents a unified theoretical framework connecting two complementary approaches to long-context degradation in Transformer-based large language models (LLMs): the information-geometric theory of **observer entropy** and the probabilistic theory of **attention collapse** via Extreme Value Theory (EVT).  

The central object is the observer entropy  

$$S_{\mathrm{obs}}(p_\theta, \varepsilon) = D_{\mathrm{KL}}\!\left(p_\theta \,\Big\|\, \widetilde{\Pi_\varepsilon p_\theta}\right),$$

whose quadratic scaling law — the **Bridge Theorem** — states

$$S_{\mathrm{obs}} = \tfrac{1}{2}\varepsilon^2\, v(\theta)^\top I(\theta)\, v(\theta) + O(\varepsilon^3),$$

where $I(\theta)$ is the Fisher information matrix.

Defining signal strength $\mu_s = \tfrac{1}{\sqrt{d}}\mathrm{Tr}(W_Q^\top W_K \Sigma_{qr})$ and effective margin $\mu_L = \mu_s - \sigma\sqrt{2\log L_{\mathrm{eff}}}$, the paper establishes a two-sided bound

$$c_1\mu_L \frac{e^{\mu_L}}{L} \le S_{\mathrm{obs}}(L) \le c_2 \frac{e^{\mu_L}}{L}$$

valid in the pre-collapse regime $\mu_L > \log 2 + 1$, and a one-sided upper bound $S_{\mathrm{obs}}(L) \le c_2 e^{\mu_L}/L$ for all sufficiently large $L$.

This bound holds in a two-sided form in the pre-collapse regime and as a one-sided upper bound for all sufficiently large $L$; the paper is explicit that the matching lower bound in the deep-collapse regime ($\mu_L \le \log 2 + 1$) remains an open problem (see Remark 4.9). This yields the **Fundamental Impossibility Theorem**: for any finite $\mu_s$, observer entropy decays to zero as $L \to \infty$, establishing long-context collapse as an information-theoretic inevitability under softmax attention. The theorem relies only on the (proved) upper bound and is therefore unaffected by the open lower-bound problem.

Using Gumbel convergence for weakly dependent logit maxima with Gaussian marginals (Leadbetter conditions), the paper derives a closed-form **Probabilistic Risk Law**:

$$P(\mathcal{F}_L) \approx 1 - \exp\!\left(-\exp\!\left(-\frac{\mu_s - \sigma\sqrt{2\log L_{\mathrm{eff}}}}{a_L}\right)\right),$$

and conjectures that the critical length $L_{\mathrm{crit}}$ is a heavy-tailed random variable (formal proof incomplete; see Remark 4.14 in the paper).

The **CPL 4.0 phase-aware governor** emerges as the principled control-theoretic response, with a Master Theorem guaranteeing a hard context cap, entropy contraction, and a sub-linear fragmentation bound $N_F(T) = O(\sqrt{T\log(1/\delta_0)})$.

Numerical verification via needle-in-a-haystack (NIAH) simulation ($\mu_s = 3.0$, $\sigma = 1.0$, $n = 2000$ trials per $L$, $L \in \{128, 256, 512, 1024, 2048, 4096, 8192\}$, seed = 42) confirms the $\varepsilon^2$ scaling law and EVT structure ($R^2 = 0.9986$ first-order; $R^2 = 0.9989$ second-order).

---

## Repository Contents  

```
llm-context-window-governance/  
    ├── .github/workflows/pages.yml     # GitHub Pages build for the live dashboard  
    ├── dashboard/                       # Interactive CPL governor simulation (JS)  
    ├── LICENSE  
    ├── README.md  
    ├── niah_experiment.py               # NIAH simulation & EVT diagnostics  
    │                                     # (also reproduced in full in Appendix D of the paper)  
    ├── niah_experiment.sha256           # SHA-256 checksum of niah_experiment.py  
    ├── simulator.py                     # CPL 4.0 discrete-time governor simulator  
    └── output/  
           ├── results_sim.csv                                   # Raw simulation data (14,000 trials)  
           ├── figure1_margin_scaling_sim.pdf                    # Figure 1: margin vs. sqrt(2 log L)  
           ├── figure2_failure_prob_sim.pdf                      # Figure 2: P(failure) vs. log L  
           ├── figure3_collapse_margin_sim.pdf                   # Figure 3: P(failure) vs. margin  
           ├── figure4_scatter_sim.pdf                           # Figure 4: scatter margin × failure  
           ├── figure2b_corrected_sim.pdf                        # Figure 2b: 2nd-order EVT linearization  
           ├── figure2c_bias_sim.pdf                             # Figure 2c: finite-size bias  
           ├── figure2d_collapse_sim.pdf                         # Figure 2d: distributional collapse  
           ├── figure2e_qq_sim.pdf                               # Figure 2e: QQ-plot vs. Gumbel(0,1)  
           └── figure2f_gev_sim.pdf                              # Figure 2f: GEV shape parameter xi  
```

**SHA-256 of `niah_experiment.py`** (as reproduced in Appendix D of the paper, and in `niah_experiment.sha256`):
`82e28f2e10fbcb0fd5d3035f0e99af1e8b3e58e274e640f0a1c1d4d8a6991732`

> **Note on `simulator.py` / dashboard:** this repository has iterated through several internal revisions of the discrete-time governor simulator alongside the paper series (v1.0 → v4.24). The version of `simulator.py` presently in the repository root, and the model the live dashboard reflects, are being reconciled; treat the **paper itself** (Section 5, "Control-Theoretic Response: CPL 4.0 Governor") as the authoritative specification of the governor equations in case of any discrepancy with the simulator code or dashboard behavior.

---

## Main Theoretical Contributions  

### 1. Bridge Theorem (Proved)  
Observer entropy satisfies $S_{\mathrm{obs}}(p_\theta, \varepsilon) = \tfrac{1}{2}\varepsilon^2 v(\theta)^\top I(\theta) v(\theta) + O(\varepsilon^3)$, connecting finite-resolution semantic coarse-graining to the Fisher–Rao geometry.  

### 2. Fundamental Impossibility Theorem (Proved conditionally)  
For any finite signal strength $\mu_s < \infty$: $\lim_{L \to \infty} S_{\mathrm{obs}}(L) = 0$. Long-context retention is an information-theoretic impossibility under softmax attention.  

### 3. Gumbel Convergence for Attention Maxima (Proved conditionally)  
Under Gaussian marginals and Leadbetter weak-dependence conditions:  
$$\frac{M_L - b_L}{a_L} \xrightarrow{d} \mathrm{Gumbel}(0,1), \quad b_L = \sigma\sqrt{2\log L_{\mathrm{eff}}},\; a_L = \frac{\sigma}{\sqrt{2\log L_{\mathrm{eff}}}}.$$  

### 4. Probabilistic Risk Law (Proved conditionally)  
Closed-form S-shaped failure probability curve in $\log L$ (Corollary 4.12 in the paper), matching empirically observed threshold collapse behavior.  

### 5. Stochastic Critical Length (Conjecture — proof incomplete)  
$\log L_{\mathrm{crit}} \approx (\mu_s - X - \log\varepsilon_0)^2 / (2\sigma^2)$, $X \sim \mathrm{Gumbel}(0,1)$, implying a heavy-tailed distribution for collapse timing. Fixed-point derivation incomplete; see Remark 4.14 in the paper.  

### 6. CPL 4.0 Master Theorem (Proved conditionally)  
The phase-aware governor guarantees: (i) hard context cap $L_k \le L_{\mathrm{cap}}$; (ii) entropy contraction; (iii) sub-linear fragmentation bound $N_F(T) = O(\sqrt{T\log(1/\delta_0)})$.  

---

## Result Status at a Glance

Version 4.24 makes the epistemic status of every main result explicit rather than implicit. Full detail is in §7 (Classification) and the referenced remarks of the paper.

| Result | Status | Key open condition |
|---|---|---|
| Bridge Theorem | **Proved** | Partition-adapted family |
| Riemannian reformulation | **Proved** | Bridge Theorem |
| Sufficient conditions (partition-adapted families) | **Proved** | Conditions A1–A4 |
| Gumbel convergence for attention maxima | Proved conditionally | Leadbetter conditions; Gaussian marginals (Remark 3.4 on non-Gaussian extension) |
| Two-sided observer-entropy bounds | Proved conditionally | Lower bound restricted to pre-collapse regime $\mu_L > \log 2 + 1$ (Remark 4.9) |
| Fundamental Impossibility Theorem | Proved conditionally | Relies only on the upper bound; unaffected by the Remark 4.9 gap |
| Probabilistic Risk Law | Proved conditionally | Gumbel convergence |
| Stochastic critical length $L_{\mathrm{crit}}$ | **Conjecture — proof incomplete** | Fixed-point derivation not completed (Remark 4.14) |
| CPL 4.0 Master Theorem | Proved conditionally | Spectral Response Assumption |
| Resolution–information trade-off | **Proved** | Bridge Theorem, positive-definite Fisher information |
| Signal-loss scaling ($\mu_s \sim \sqrt{d}\cdot\mathcal{L}^{-1/2}$) | Modelling axiom | Not proved; motivated heuristically |

Two points worth flagging explicitly for readers of the abstract:
- The **partition-based** observer entropy $S_{\mathrm{obs}}(p_\theta,\varepsilon)$ (Def. 2.4) and the **attention-uniform** $S_{\mathrm{obs}}(L)$ (Def. 3.7) are connected *interpretatively*, not by a proved equivalence proposition — this is listed as open problem (2)/(8) in §7.2.
- The lower bound in the two-sided entropy inequality is proved only for $\mu_L > \log 2 + 1$; for larger $L$ only the upper bound holds, though this does not weaken the Impossibility Theorem (see Remark on this in §3).

---

## Identified Open Problems  

The following gaps are explicitly documented in the paper:  

1. Extension to non-Gaussian sub-Gaussian logit marginals (domain-of-attraction argument required).  
2. Formal identification of partition-based $S_{\mathrm{obs}}(p_\theta,\varepsilon)$ and attention-uniform $S_{\mathrm{obs}}(L)$ definitions.  
3. Lower bound on $S_{\mathrm{obs}}(L)$ in the deep collapse regime $\mu_L \le \log 2 + 1$.  
4. Completion of the fixed-point derivation for the distributional form of $L_{\mathrm{crit}}$.  
5. Quantification of the convergence rate $\xi(L) \to 0$ from the pre-asymptotic Weibull-class regime to the asymptotic Gumbel limit.  
6. Formal proof of the scaling conjecture $\mu_s \sim \sqrt{d} \cdot \mathcal{L}^{-1/2}$.  
7. Extension to multi-head aggregation and multiple relevant tokens.  

---

## Simulation: NIAH EVT Diagnostic Results  

Script: `niah_experiment.py`  
Run command used for this version:  
```bash  
python niah_experiment.py --mode simulation --n_trials 2000 --mu_s 3.0 --seed 42  
```

Parameters: $\mu_s = 3.0$, $\sigma = 1.0$, $n = 2000$ trials per $L$, seed = 42.  

### Per-length summary  

| $L$  | Mean margin | $P(\text{failure})$ | $n$ trials |  
|------|-------------|----------------------|------------|  
| 128  | 0.4489      | 0.3390               | 2000       |  
| 256  | 0.1796      | 0.4465               | 2000       |  
| 512  | −0.0443     | 0.5130               | 2000       |  
| 1024 | −0.2449     | 0.5835               | 2000       |  
| 2048 | −0.4482     | 0.6605               | 2000       |  
| 4096 | −0.6528     | 0.7380               | 2000       |  
| 8192 | −0.7942     | 0.7690               | 2000       |  

### EVT diagnostic summary  

| Test | Result | Interpretation |  
|------|--------|----------------|  
| Scaling $R^2$ (1st order) | 0.9986 | EVT structure holds |  
| Scaling $R^2$ (2nd order) | 0.9989 | 2nd-order consistent |  
| OLS intercept (1st order) | $\hat\mu_s = 3.866$ | bias $= +0.866$ |  
| OLS intercept (2nd order) | $\hat\mu_s = 2.916$ | bias $= -0.084$ (90% reduction) |  
| Mean bias 1st order | $+0.482$ | large, systematic |  
| Mean bias 2nd order | $-0.122$ | small; finite-size origin |  
| KS test vs. Gumbel(0,1) | stat $= 0.394$, $p < 10^{-10}$ | pre-asymptotic regime; not a theory failure |  
| GEV shape $\xi$ (95% bootstrap CI) | $-0.138$ $[-0.296, -0.126]$ | finite-sample artefact; $\xi \to 0$ as $L\to\infty$ |  

**Note on KS test:** At $n = 14{,}000$ pooled trials the test resolves deviations of order $1/\sqrt{n} \approx 0.008$. Rejection quantifies the pre-asymptotic distance to the Gumbel limit, not a failure of the asymptotic theory. Note also that the pooled quantity normalises the *margin* $\mu = z_{\mathrm{rel}} - M_L$, not $M_L$ directly — the margin's KS statistic includes variance from $z_{\mathrm{rel}}$ itself; normalising $M_L$ alone gives KS ≈ 0.054, much closer to the Gumbel limit (see the paper's remark on margin normalisation).  

**Note on GEV $\xi < 0$:** The Gaussian distribution belongs to $\mathrm{MDA}(\mathrm{Gumbel})$ (Gnedenko–de Haan criterion); $\xi \to 0$ as $L \to \infty$. The value $\xi \approx -0.14$ quantifies the distance of the range $L \in [128, 8192]$ from the asymptotic regime rather than indicating a heavy-tailed (Fréchet) failure mode.  

---

## Installation  

```bash  
pip install numpy pandas matplotlib scipy scikit-learn  
```

Dependencies used for the simulation in this version:  

| Package | Role |  
|---------|------|  
| `numpy` | Numerical computations |  
| `pandas` | Data aggregation, CSV output |  
| `matplotlib` | Figure generation |  
| `scipy` | `genextreme`, `kstest`, `curve_fit`, `expit` |  
| `scikit-learn` | `IsotonicRegression` |  

> *Exact package versions were recorded in the original simulation environment.  
> Since the simulation script is available upon request, environment details   
> can be provided to researchers seeking to replicate the computational setup.*  

---

## Reproducibility  

All simulation outputs are fully deterministic under the fixed seed:  

```bash  
python niah_experiment.py --mode simulation --n_trials 2000 --mu_s 3.0 --seed 42  
```

This reproduces:  
- `results_sim.csv` (14,000 rows × 7 columns)  
- All nine PDF figures (`figure1_*` through `figure2f_*`)  

> **Note**: `niah_experiment.py` is included directly in this repository (with its SHA-256 checksum in `niah_experiment.sha256`) and its full source is also reproduced in Appendix D of the paper, so no separate request is needed to inspect or rerun it.  
All reported results are deterministic under the specified parameters and can be independently validated using the provided `results_sim.csv` and figure files.  

---

## Relation to Companion Manuscripts  

This paper is version 4.24 of a series and supersedes versions 1.0 through 4.2.  

| Version | Title | DOI |  
|---------|-------|-----|  
| 4.2 | Information-Geometric Context Window Governance and the Probabilistic Theory of Long-Context Collapse in Large Language Models | [10.5281/zenodo.20925845](https://doi.org/10.5281/zenodo.20925845) |  
| 4.0 | Information-Geometric Context Window Governance and the Probabilistic Theory of Long-Context Collapse in Large Language Models | [10.5281/zenodo.19568493](https://doi.org/10.5281/zenodo.19568493) |  
| 3.0 | Information-Geometric Context Window Governance and the Probabilistic Theory of Long-Context Collapse in Large Language Models | [10.5281/zenodo.19352770](https://doi.org/10.5281/zenodo.19352770) |  
| 2.0 | Information-Geometric Context Window Governance for Large Language Models via Observer Entropy and the Cognitive Phase Law (CPL 4.0) | [10.5281/zenodo.19177363](https://doi.org/10.5281/zenodo.19177363) |  
| 1.0 | Phase-Aware Context Window Governance for Large Language Models: A CPL-Based Engineering Framework | [10.5281/zenodo.18784361](https://doi.org/10.5281/zenodo.18784361) |  

> This repository tracks the latest version (4.24) of the framework only. Earlier preprint versions (1.0–4.2) are preserved as immutable Zenodo records with their own DOIs (see table above); this repo does not maintain per-version code snapshots or subdirectories.

Companion theoretical manuscripts cited in this paper:  
- Khomyakov (2026a). *KL-Geometric Structure of Observer Entropy*. [doi:10.5281/zenodo.19202244](https://doi.org/10.5281/zenodo.19202244)  
- Khomyakov (2026b). *Information-Geometric Context Window Governance for LLMs via Observer Entropy and the CPL 4.0*. [doi:10.5281/zenodo.19177363](https://doi.org/10.5281/zenodo.19177363)  
- Khomyakov (2026c). *Information-Geometric Context Window Governance and the Probabilistic Theory of Long-Context Collapse in Large Language Models*. [doi:10.5281/zenodo.19352770](https://doi.org/10.5281/zenodo.19352770)  

Code repositories:  
- https://github.com/Khomyakov-Vladimir/observer-entropy-bridge  
- https://github.com/Khomyakov-Vladimir/llm-context-window-governance  

---

## Citation

```bibtex
@misc{khomyakov_2026_22056414,  
  author       = {Khomyakov, Vladimir},  
  title        = {Information-Geometric Context Window Governance  
                   and the Probabilistic Theory of Long-Context  
                   Collapse in Large Language Models},  
  year         = 2026,  
  publisher    = {Zenodo},  
  version      = {4.24},  
  doi          = {10.5281/zenodo.22056414},  
  url          = {https://doi.org/10.5281/zenodo.22056414},  
}
```

---

## Keywords  

observer entropy, information geometry, Fisher information, Kullback–Leibler divergence, extreme value theory, Gumbel distribution, attention mechanism, transformer models, long-context degradation, context window governance, CPL 4.0, needle-in-a-haystack, probabilistic risk law, softmax attention collapse  

---

## License  

- **Scientific article and associated documentation** (PDF, figures, LaTeX sources):  
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)  
- **Source code and simulation scripts** (`niah_experiment.py`, `simulator.py`, `dashboard/`):  
  [MIT License](https://opensource.org/licenses/MIT)  

---

## Author  

**Vladimir Khomyakov**  
Independent Researcher  
ORCID: [0009-0006-3074-9145](https://orcid.org/0009-0006-3074-9145)  
