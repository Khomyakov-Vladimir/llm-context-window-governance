# Phase-Aware Context Window Governance for LLM Inference

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19177363.svg)](https://doi.org/10.5281/zenodo.19177363)  
[![Dashboard](https://img.shields.io/badge/Dashboard-Live-10b981?style=flat-square)](https://khomyakov-vladimir.github.io/llm-context-window-governance/)  
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/) — Scientific article and associated documentation  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) — Source code and simulation scripts  

**Based on:** Khomyakov, V. (2026). *Phase-Aware Context Window Governance for Large Language Models: A CPL-Based Engineering Framework (2.0).* Zenodo.  
[doi:10.5281/zenodo.19177363](https://doi.org/10.5281/zenodo.19177363)

---

## 🔬 Interactive Dashboard (Live Simulation)  

An interactive CPL Context Governor simulation is available online:  

👉 **https://khomyakov-vladimir.github.io/llm-context-window-governance/**  

The dashboard implements discrete Eqs. (9), (11), (14) with policy (18–19) and allows real-time parameter exploration (σ, ρ, α_tight, κ_L, γ, T).  

---

## Problem Statement  

Large Language Models exhibit semantic drift, coherence loss, and latency spikes as the context window fills up. Existing mitigation strategies — sliding windows, RoPE extensions, fixed-interval summarization — operate on heuristic length thresholds and ignore the model's internal behavioral state. This framework applies the Cognitive Phase Law (CPL 4.0) to trigger context compression *only when the model actually enters a degraded phase*, replacing blind length-based policies with state-aware governance.  

---

## Core Idea  

CPL 4.0 defines three cognitive phases — **Coherence (C)**, **Reorganization (R)**, and **Fragmentation (F)** — through observable entropy, semantic stability, and the rate of entropy change. This framework maps that logic onto the LLM inference loop: at each request–response turn, the model's phase is classified, and context management actions plus decoding parameters are selected accordingly.  

## Agent State (per step k)  

```
x_k = (L_k, Ĥ_k, Ŝ_k, D̂_k, z_k, c_k)
```

| Component | Description |  
|-----------|-------------|  
| `L_k` | Context length (tokens) |  
| `Ĥ_k` | Observer entropy (KL divergence of output distribution from its coarse-grained projection: `S_obs(p_θ, ε)`) |  
| `Ŝ_k` | Semantic stability ∈ [0, 1] (e.g., cosine similarity of successive embeddings) |  
| `D̂_k` | Discrete entropy derivative: `|Ĥ_k − Ĥ_{k−1}|` |  
| `z_k` | Current phase: C / R / F |  
| `c_k` | Compressed memory size |  

## Phase Classifier  

Phases are assigned using the **"Reorganization first"** priority rule:  

```
if  D̂_k ≥ γ                        →  z_k = R   (Reorganization)  
elif Ĥ_k < H_c  and  Ŝ_k > S_c    →  z_k = C   (Coherence)  
else                                →  z_k = F   (Fragmentation)  
```

Canonical thresholds (from CPL 4.0; require recalibration for LLMs):  
`H_c = ln(3) ≈ 1.099 nats`,  `S_c = 0.7`,  `γ = 0.1`.  

## Context Management Policy  

The action at each step is determined jointly by the phase and the current context length:  

| Condition | Action (`m_k`) | Token release |  
|-----------|-----------------|---------------|  
| `z_k = F` | **chunk** (extract + compress) | `r_recover` (aggressive) |  
| `z_k = R` | **summarize** | `r_rescue` |  
| `z_k = C` and `L_k > L_warn` | **summarize** | `r_rescue` |  
| `z_k = C` and `L_k ≤ L_warn` | **keep** (no action) | 0 |  

Context thresholds are ordered as: `L_recover < L_warn < L_cap < L_practical ≤ L_max`.  

## Adaptive Decoding  

```
if z_k ∈ {R, F}  or  L_k > L_warn:  
    θ_k = θ_tight      # reduced temperature / top-p  
else:  
    θ_k = θ_base        # standard parameters  
```

## Formal Guarantees  

1. **Hard context invariant.** Given correctly configured thresholds, `L_k ≤ L_cap` for all `k` — the model never enters the catastrophic-latency regime.  
2. **Entropy contraction.** Under tight decoding, expected entropy decreases by at least `α · Δ` per step while `Ĥ_k` exceeds its target by `Δ`.  
3. **Bounded degradation.** The number of Fragmentation steps grows as `O(√T)`, not linearly with the horizon `T`.  

See the [preprint](https://doi.org/10.5281/zenodo.19177363) for complete proofs and assumptions.  

## Inference Pipeline Overview  

```
┌──────────────────────────────────────────────┐  
│              Inference Loop                  │  
│                                              │  
│  1. Receive user request  (U_k tokens)       │  
│  2. Compute surrogates:                      │  
│     · Ĥ_k  (observer entropy S_obs(p_θ, ε))  │  
│     · Ŝ_k  (embedding cosine similarity)     │  
│     · D̂_k = |Ĥ_k − Ĥ_{k−1}|                  │  
│  3. Classify phase  z_k                      │  
│  4. Select action  (m_k, θ_k)                │  
│  5. Apply context management:                │  
│     · keep / summarize / chunk+retrieve      │  
│  6. Set decoding parameters                  │  
│  7. Generate response  (Y_k tokens)          │  
│  8. Update  L_{k+1}, Ĥ_{k+1}, Ŝ_{k+1}        │  
│  9. Log metrics                              │  
│                                              │  
│  → Repeat for step k+1                       │  
└──────────────────────────────────────────────┘  
```

## Implementation Note  

The phase classifier can be integrated into any inference pipeline (Transformers, LangChain, vLLM, etc.) in a few lines:  

```python  
def get_cpl_phase(  
    entropy: float,  
    stability: float,  
    delta_entropy: float,  
    H_c: float = 1.099,  
    S_c: float = 0.7,  
    gamma: float = 0.1,  
) -> str:  
    """Classify the current cognitive phase per CPL 4.0.  

    Args:  
        entropy:       Observer entropy S_obs(p_θ, ε) — KL divergence from coarse-grained projection (Ĥ_k).
        stability:     Semantic stability score (Ŝ_k ∈ [0, 1]).  
        delta_entropy:  |Ĥ_k − Ĥ_{k−1}|, discrete entropy derivative.  
        H_c:           Entropy threshold (default: ln 3 ≈ 1.099 nats).  
        S_c:           Stability threshold (default: 0.7).  
        gamma:         Entropy-rate threshold (default: 0.1).  

    Returns:  
        "C" (Coherence), "R" (Reorganization), or "F" (Fragmentation).  
    """  
    if delta_entropy >= gamma:  
        return "R"  # Reorganization — entropy is changing rapidly  
    elif entropy < H_c and stability > S_c:  
        return "C"  # Coherence — stable, low-entropy regime  
    return "F"      # Fragmentation — degraded state  


def select_action(phase: str, context_length: int, L_warn: int):  
    """Select context-management action and decoding mode.  

    Returns:  
        (action, decoding_mode) where action ∈ {keep, summarize, chunk}  
        and decoding_mode ∈ {base, tight}.  
    """  
    if phase == "F":  
        return "chunk", "tight"  
    elif phase == "R":  
        return "summarize", "tight"  
    elif context_length > L_warn:  
        return "summarize", "tight"  
    return "keep", "base"  
```

> **Note.** The canonical thresholds (`H_c`, `S_c`, `γ`) are derived from human-observer calibration in CPL 4.0. For production LLM deployment, these values must be recalibrated from operational logs. Empirical validation remains future work — see the preprint for a full discussion of assumptions and limitations.  

## Calibration  

All thresholds (`H_c`, `S_c`, `γ`, `L_warn`, `L_cap`, `δ`, `σ`) are designed to be calibrated from operational telemetry. The preprint provides the formal structure; empirical fitting to specific models and workloads is left as future work.  

## References  

- **KL-Geom MITF 2.4:** Khomyakov, V. (2026). *KL-Geometric Structure of Observer Entropy: A Minimal Information-Theoretic Framework (2.4).* Zenodo [https://doi.org/10.5281/zenodo.19080663](https://doi.org/10.5281/zenodo.19080663)  
- **CPL 4.0:** Khomyakov, V. (2025). *The Law of Cognitive Phases (CPL): Formalizing Observer State Transitions in Subjective Physics (4.0).* Zenodo [https://doi.org/10.5281/zenodo.17788635](https://doi.org/10.5281/zenodo.17788635)  
- **MMSF 12.0:** Khomyakov, V. (2025). *Cognitive Projection and Observer Entropy: A Minimal Model of Subjective Physics (12.0).* Zenodo [https://doi.org/10.5281/zenodo.17407408](https://doi.org/10.5281/zenodo.17407408)  
- **This framework:** Khomyakov, V. (2026). *Phase-Aware Context Window Governance for Large Language Models: A CPL-Based Engineering Framework (2.0).* Zenodo [https://doi.org/10.5281/zenodo.19177363](https://doi.org/10.5281/zenodo.19177363)  

Author identification and project information:  
[ORCID: 0009-0006-3074-9145](https://orcid.org/0009-0006-3074-9145)  

## License  

- **Scientific article and associated documentation** (PDF, figures, LaTeX sources):  
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)  
- **Source code and simulation scripts**:  
  [MIT License](https://opensource.org/licenses/MIT)  
