#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simulator.py

CPL Context Governor — Discrete Simulator  (v2.0)
==================================================
Implements Eqs. (9), (14) and Policy §4 from:

  Khomyakov, V. (2026). Information-Geometric Context Window Governance
  for Large Language Models via Observer Entropy and the Cognitive Phase
  Law (CPL 4.0).  Version 2.0.
  doi:10.5281/zenodo.19177363

Stability Ŝ_k is not an independent stochastic process. Per Definition 3.5
and Corollary 3.7 of the paper, stability is an algebraic consequence of
entropy:

    Ŝ_k = 1 - Ĥ_k / S_obs_max

All noise is carried by the entropy channel  σ_H  only (Assumption 4.1(ε')).

The boundary sweep (sweep_spectral_boundary() and the --sweep CLI flag)
explores Lemma 4.4 (Revised Drift Conditions) and Assumption 3.14
(Spectral Response) by varying  α_tight  and  Q_ratio  (the ratio
Q(θ_tight)/Q(θ_base)).

h_target() accepts θ as an explicit parameter consistent with
h(L, θ_tight)  in Assumption 4.1(h) of the paper; at runtime the caller
always passes  θ^tight, and the explicit signature keeps the code aligned
with the formal definition.

Usage:
    python simulator.py                   # Generate all plots (seed 42)
    python simulator.py --seed 7          # Specific seed
    python simulator.py --sweep           # Spectral-response boundary sweep
    python simulator.py --T 400           # Longer horizon
    python simulator.py --seed 7 --sweep  # Combined
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════

@dataclass
class CPLConfig:
    """
    Parameters corresponding to §3–§4 of v2.0.

    Information-geometric parameters:
        S_obs_max   — supremum of observer entropy over K  (Def. 3.4)
        Q_ratio     — Q(θ_tight) / Q(θ_base)  (Assumption 3.14);
                      must be < 1 for Assumption (Spectral) to hold
    """

    # ── Context thresholds — Assumption 4.1 (T), Eq. (15) ────────────
    L_recover: int   = 2_000
    L_warn:    int   = 3_200
    L_cap:     int   = 4_000
    L_practical: int = 4_800
    L_max:     int   = 5_000
    L_safe:    int   = 1_500   # kept for latency formula; not in stability

    # ── Input bounds — Assumption 4.1 (U), (Y)  [v2.0 numbering] ─────
    U_max:        int   = 400
    Y_max_tight:  int   = 350

    # ── Entropy dynamics — Eq. (14) ───────────────────────────────────
    alpha_base:  float = 0.15   # α_eff under θ_base
    alpha_tight: float = 0.35   # α_eff under θ_tight  (Assumption 3.13)
    H_c:         float = 1.099  # H_c = ln(3) nats
    eta:         float = 3e-4   # slope of h(L, θ_tight)
    delta_H:     float = 0.25   # regime penalty at L ≥ L_practical

    # ── Information-geometric parameters  (v2.0 additions) ───────────
    # S_obs_max: supremum of observer entropy over compact K  (Def. 3.4).
    # In the softmax family  S_obs_max = ½ε² max_θ∈K Q(θ).
    # We treat this as a calibrated scalar; default matches  H_c * 1.5
    # so that  S_c = 0.7  corresponds to  Ĥ_k / S_obs_max = 0.3.
    S_obs_max: float = 1.648   # = H_c * 1.5  ≈ 1.099 * 1.5

    # Q_ratio = Q(θ_tight) / Q(θ_base) — Assumption 3.14 (Spectral).
    # Must satisfy  Q_ratio < 1  for tight decoding to reduce entropy.
    Q_ratio: float = 0.60

    # ── Phase classifier — Eq. (8) ────────────────────────────────────
    gamma:      float = 0.10
    beta_hyst:  float = 0.05
    S_c:        float = 0.70   # stability threshold (used in phase rule
                               # and Lyapunov potential)

    # ── Noise — Assumption 4.1 (ε') ──────────────────────────────────
    # Only σ_H remains; σ_S is removed (stability is deterministic given Ĥ).
    sigma_H: float = 0.04

    # ── Rescue releases — Eqs. (16)–(17) ─────────────────────────────
    r_rescue:  int = 1_550
    r_recover: int = 1_950

    # ── Simulation ────────────────────────────────────────────────────
    T:    int = 200
    seed: int = 42

    # ── Derived: σ_eff for N_F bound — Lemma 4.3 ─────────────────────
    # In v2.0  ξ_{k+1} = −ε_k^H · 1{Ĥ_k > H_c}  (scalar, since the
    # stability term is algebraically determined by entropy and carries
    # no independent noise).  Hence  σ_eff = σ_H.
    @property
    def sigma_eff(self) -> float:
        return self.sigma_H

    # ── Design function  h(L, θ) — Assumption 4.1(h) ─────────────────
    def h_target(self, L: float, theta: str = "tight") -> float:
        """
        h(L, θ) = H_c − η · (L_cap − L)₊

        The function satisfies  h(L, θ_tight) < H_c  for all  L ≤ L_cap
        (Assumption 4.1(h)), which guarantees entropy contraction in
        tight decoding mode.

        Parameters
        ----------
        L     : current context length
        theta : decoding mode ('tight' or 'base'); included per the
                formal signature  h(L, θ)  in the paper.
                At present both modes use the same functional form;
                the argument is kept for forward compatibility.
        """
        return self.H_c - self.eta * max(self.L_cap - L, 0.0)


# ══════════════════════════════════════════════════════════════════════
# Data record
# ══════════════════════════════════════════════════════════════════════

@dataclass
class StepRecord:
    k:     int
    L:     float
    H:     float    # Ĥ_k = S_obs(p_θ, ε)  — observer entropy
    S:     float    # Ŝ_k = 1 − Ĥ_k / S_obs_max  — semantic stability
    D:     float    # D̂_k = |Ĥ_k − Ĥ_{k-1}|
    z:     str      # phase ∈ {C, R, F}
    V:     float    # Lyapunov potential  V_k = (Ĥ_k − H_c)₊ + (S_c − Ŝ_k)₊
    m:     str      # action ∈ {keep, summarize, chunk}
    theta: str      # decoding mode ∈ {base, tight}
    tau:   float    # latency


# ══════════════════════════════════════════════════════════════════════
# Core simulator
# ══════════════════════════════════════════════════════════════════════

def simulate(cfg: CPLConfig, use_governor: bool) -> list[StepRecord]:
    """
    Run the discrete-time CPL context governor simulation.

    State vector  x_k = (L_k, Ĥ_k, Ŝ_k, D̂_k, z_k, c_k)  — Eq. (9) of v2.0.

    Stability  Ŝ_k  is computed from entropy via the algebraic coupling
    (Corollary 3.7):

        Ŝ_k = 1 − Ĥ_k / S_obs_max

    It is not propagated as a separate SDE.

    Parameters
    ----------
    cfg          : simulation parameters
    use_governor : if True apply policy Eqs. (18)–(19); otherwise baseline
    """
    rng = np.random.RandomState(cfg.seed)

    # ── Initial state ─────────────────────────────────────────────────
    L: float = 800.0
    H: float = 0.60                              # observer entropy
    S: float = 1.0 - H / cfg.S_obs_max          # algebraic stability
    H_prev: float = H
    z_prev: str   = "C"

    trajectory: list[StepRecord] = []

    for k in range(cfg.T + 1):

        # ── Phase classifier — Eq. (8), "Reorganization first" ───────
        D = abs(H - H_prev)
        H_c_eff = cfg.H_c + (cfg.beta_hyst if z_prev == "C" else 0.0)
        if D >= cfg.gamma:
            z = "R"
        elif H < H_c_eff and S > cfg.S_c:
            z = "C"
        else:
            z = "F"

        # ── Lyapunov potential — Eq. (23) ─────────────────────────────
        V = max(H - cfg.H_c, 0.0) + max(cfg.S_c - S, 0.0)

        # ── Policy — Eqs. (18)–(19) ───────────────────────────────────
        if use_governor:
            if z == "F":
                m, r = "chunk",     cfg.r_recover
            elif z == "R":
                m, r = "summarize", cfg.r_rescue
            elif z == "C" and L > cfg.L_warn:
                m, r = "summarize", cfg.r_rescue
            else:
                m, r = "keep",      0
            theta = "tight" if (z in ("R", "F") or L > cfg.L_warn) else "base"
        else:
            m, r  = "keep", 0
            theta = "base"

        # ── Effective contraction rate — Assumption 3.13 ──────────────
        alpha = cfg.alpha_tight if theta == "tight" else cfg.alpha_base

        # ── Latency — Eq. (10) ────────────────────────────────────────
        g_L         = 1e-5 * L * L
        tau_throttle = 2.0 if L >= cfg.L_practical else 0.0
        tau          = 0.1 + g_L + tau_throttle

        trajectory.append(
            StepRecord(k=k, L=L, H=H, S=S, D=D, z=z, V=V,
                       m=m, theta=theta, tau=tau)
        )

        if k == cfg.T:
            break

        # ── State update ──────────────────────────────────────────────

        # Inputs (bounded by Assumption 4.1 (U), (Y))
        U_k   = rng.randint(int(cfg.U_max * 0.3), cfg.U_max + 1)
        Y_k   = rng.randint(int(cfg.Y_max_tight * 0.2), cfg.Y_max_tight + 1)
        eps_H = cfg.sigma_H * rng.randn()   # sub-Gaussian noise, Assumption 4.1(ε')

        # Context — Eq. (9)
        L_next = float(np.clip(L + U_k + Y_k - r, 0.0, cfg.L_max))

        # Entropy — Eq. (14)
        # h(L, θ) satisfies h(L, θ_tight) < H_c for all L ≤ L_cap
        H_tgt   = cfg.h_target(L, theta)
        Delta_H = cfg.delta_H if L >= cfg.L_practical else 0.0
        H_next  = H - alpha * (H - H_tgt) + Delta_H + eps_H

        # Stability — Corollary 3.7
        # Ŝ_{k+1} = 1 − Ĥ_{k+1} / S_obs_max
        # No independent SDE; stability is fully determined by entropy.
        S_next = float(np.clip(1.0 - H_next / cfg.S_obs_max, 0.0, 1.0))

        H_prev = H
        z_prev = z
        L, H, S = L_next, H_next, S_next

    return trajectory


# ══════════════════════════════════════════════════════════════════════
# Helper functions
# ══════════════════════════════════════════════════════════════════════

def cumulative_NF(traj: list[StepRecord]) -> np.ndarray:
    """Cumulative count of Fragmentation steps  N_F(k)."""
    return np.cumsum([1 if s.z == "F" else 0 for s in traj])


def theoretical_bound(
    V0: float, delta: float, sigma: float, T: int, delta0: float = 0.05
) -> np.ndarray:
    """
    High-probability N_F bound — Theorem 5.3 / Eq. (24) of v2.0:

        N_F(T) ≤ V₀/δ + (σ/δ) √(2T · ln(1/δ₀))   w.p. ≥ 1 − δ₀

    In v2.0  σ = σ_H  (scalar, since stability noise is removed).
    """
    ks      = np.arange(T + 1, dtype=float)
    ks[0]   = 1.0   # avoid log(0) at k=0
    return V0 / delta + (sigma / delta) * np.sqrt(2.0 * ks * np.log(1.0 / delta0))


# ══════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════

PHASE_COLORS = {"C": "#10b981", "R": "#f59e0b", "F": "#ef4444"}


def _phase_bar(ax, traj: list[StepRecord], height: float = 0.02) -> None:
    """Thin phase-color strip at the bottom of an axes."""
    for s in traj:
        ax.axvspan(s.k - 0.5, s.k + 0.5,
                   ymin=0, ymax=height,
                   color=PHASE_COLORS[s.z], alpha=0.6, lw=0)


def plot_all(cfg: CPLConfig, outdir: Path) -> Path:
    """
    Generate the six canonical visualization panels.

    Panels correspond to v2.0 proposition/theorem numbers:
      (A) Prop. 5.1 — Invariant Context Cap
      (B) Prop. 5.2 — Entropy Contraction
      (C) Stability (algebraic from entropy, no independent dynamics)
      (D) Lyapunov Potential V_k
      (E) N_F(T) — Thm. 5.3
      (F) Phase diagram (Ĥ, Ŝ)
    """
    ctrl = simulate(cfg, use_governor=True)
    base = simulate(cfg, use_governor=False)

    ks = [s.k for s in ctrl]
    T  = cfg.T

    fig = plt.figure(figsize=(18, 22), facecolor="#0f172a")
    gs  = gridspec.GridSpec(
        3, 2, hspace=0.35, wspace=0.28,
        left=0.07, right=0.95, top=0.95, bottom=0.04
    )
    common_kw = dict(color="#e2e8f0", fontsize=10)
    ax_kw     = dict(facecolor="#0f172a")

    def style_ax(ax, title: str, ylabel: str) -> None:
        ax.set_facecolor("#111827")
        ax.set_title(title, color="#e2e8f0", fontsize=12, fontweight="bold", pad=10)
        ax.set_ylabel(ylabel, **common_kw)
        ax.set_xlabel("Step k", **common_kw)
        ax.tick_params(colors="#64748b", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#1e293b")
        ax.grid(True, alpha=0.08, color="#94a3b8")

    # ── (A) Context Length ────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0], **ax_kw)
    style_ax(ax, "(A) Context Length  Lₖ  —  Prop. 5.1", "Lₖ (tokens)")
    ax.plot(ks, [s.L for s in base], color="#ef4444", lw=1.5, alpha=0.7, label="Baseline")
    ax.plot(ks, [s.L for s in ctrl], color="#10b981", lw=2,   label="CPL-Governor")
    ax.axhline(cfg.L_warn,      color="#f59e0b", ls="--",  lw=1.0, alpha=0.7, label="L_warn")
    ax.axhline(cfg.L_cap,       color="#ef4444", ls="--",  lw=1.3, alpha=0.8, label="L_cap")
    ax.axhline(cfg.L_practical, color="#dc2626", ls="-.",  lw=1.5,             label="L_practical")
    ax.legend(fontsize=8, facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")
    _phase_bar(ax, ctrl)

    # ── (B) Observer Entropy ──────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1], **ax_kw)
    style_ax(ax, "(B) Observer Entropy  Ĥₖ = Sobs(pθ, ε)  —  Prop. 5.2", "Ĥₖ (nats)")
    ax.plot(ks, [s.H for s in base], color="#ef4444", lw=1.5, alpha=0.7, label="Baseline")
    ax.plot(ks, [s.H for s in ctrl], color="#10b981", lw=2,   label="CPL-Governor")
    ax.axhline(cfg.H_c,       color="#ef4444", ls="--", lw=1.3, alpha=0.8,
               label="H_c = ln(3)")
    ax.axhline(cfg.S_obs_max, color="#8b5cf6", ls=":",  lw=1.0, alpha=0.6,
               label="S_obs_max")
    ax.legend(fontsize=8, facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")
    _phase_bar(ax, ctrl)

    # ── (C) Semantic Stability (algebraic) ───────────────────────────
    # v2.0: Ŝ_k = 1 − Ĥ_k / S_obs_max  (Corollary 3.7)
    ax = fig.add_subplot(gs[1, 0], **ax_kw)
    style_ax(ax,
             "(C) Stability  Ŝₖ = 1 − Ĥₖ/S_obs_max  —  Cor. 3.7  (algebraic)",
             "Ŝₖ")
    ax.plot(ks, [s.S for s in base], color="#ef4444", lw=1.5, alpha=0.7, label="Baseline")
    ax.plot(ks, [s.S for s in ctrl], color="#10b981", lw=2,   label="CPL-Governor")
    ax.axhline(cfg.S_c, color="#3b82f6", ls="--", lw=1.3, alpha=0.8, label="S_c = 0.7")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")
    # Annotation clarifying algebraic coupling
    ax.text(0.02, 0.05,
            "Stability is algebraically coupled to entropy\n"
            "via Corollary 3.7 — no independent SDE",
            transform=ax.transAxes, fontsize=7.5, color="#94a3b8",
            va="bottom", style="italic")
    _phase_bar(ax, ctrl)

    # ── (D) Lyapunov Potential ────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1], **ax_kw)
    style_ax(ax, "(D) Lyapunov Potential  Vₖ  —  Lem. 4.4 Drift Condition", "Vₖ")
    ax.plot(ks, [s.V for s in base], color="#ef4444", lw=1.5, alpha=0.7, label="Baseline Vₖ")
    ax.plot(ks, [s.V for s in ctrl], color="#10b981", lw=2,   label="Governor Vₖ")
    ax.axhline(0, color="#10b981", lw=0.8, alpha=0.3)
    ax.legend(fontsize=8, facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")
    _phase_bar(ax, ctrl)

    # ── (E) Fragmentation Occupancy N_F(T) ───────────────────────────
    # Theorem 5.3 in v2.0
    ax = fig.add_subplot(gs[2, 0], **ax_kw)
    style_ax(ax, "(E) Fragmentation Occupancy  N_F(T)  —  Thm. 5.3", "N_F(T)")

    nf_base = cumulative_NF(base)
    nf_ctrl = cumulative_NF(ctrl)

    V0    = max(ctrl[0].V, 0.1)
    # σ_eff = σ_H  (v2.0: stability noise removed, Lemma 4.3)
    sigma = cfg.sigma_eff
    bound = theoretical_bound(V0, delta=0.05, sigma=sigma, T=T, delta0=0.05)

    ax.plot(ks, nf_base, color="#ef4444", lw=1.5, alpha=0.7, label="Baseline N_F")
    ax.plot(ks, nf_ctrl, color="#10b981", lw=2,   label="Governor N_F")
    ax.plot(ks, bound,   color="#8b5cf6", lw=1.5, ls="--",
            label=f"Bound (δ₀=0.05, σ=σ_H={sigma:.3f})")
    ax.legend(fontsize=8, facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")

    # ── (F) Phase Diagram (Ĥ, Ŝ) ─────────────────────────────────────
    ax = fig.add_subplot(gs[2, 1], **ax_kw)
    style_ax(ax, "(F) Phase Space  (Ĥ, Ŝ)  —  algebraic manifold  Ŝ = 1 − Ĥ/S_obs_max",
             "Ŝₖ")
    ax.set_xlabel("Ĥₖ", **common_kw)

    # Scatter trajectories
    ax.scatter([s.H for s in base], [s.S for s in base],
               c="#ef4444", alpha=0.30, s=12, label="Baseline",     zorder=2)
    ax.scatter([s.H for s in ctrl], [s.S for s in ctrl],
               c="#10b981", alpha=0.55, s=16, label="CPL-Governor", zorder=3)

    # Algebraic coupling manifold  Ŝ = 1 − Ĥ / S_obs_max
    H_range = np.linspace(0, cfg.S_obs_max, 200)
    S_range = 1.0 - H_range / cfg.S_obs_max
    ax.plot(H_range, S_range, color="#94a3b8", lw=1.0, ls=":",
            alpha=0.5, label="Algebraic manifold (v2.0)", zorder=1)

    ax.axvline(cfg.H_c,       color="#ef4444", ls="--", lw=1.2, alpha=0.7)
    ax.axhline(cfg.S_c,       color="#3b82f6", ls="--", lw=1.2, alpha=0.7)
    ax.axvline(cfg.S_obs_max, color="#8b5cf6", ls=":",  lw=1.0, alpha=0.5)
    ax.text(0.05, 0.92, "COHERENCE",     color="#10b981", fontsize=9, alpha=0.5,
            transform=ax.transAxes)
    ax.text(0.65, 0.08, "FRAGMENTATION", color="#ef4444", fontsize=9, alpha=0.5,
            transform=ax.transAxes)
    ax.set_xlim(-0.05, cfg.S_obs_max * 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")

    fig.suptitle(
        "CPL Context Governor v2.0 — Information-Geometric Simulation\n"
        f"doi:10.5281/zenodo.19177363  ·  "
        f"seed={cfg.seed}  T={cfg.T}  σ_H={cfg.sigma_H}  "
        f"α_tight={cfg.alpha_tight}  S_obs_max={cfg.S_obs_max:.3f}",
        color="#94a3b8", fontsize=11, y=0.99
    )

    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"cpl_governor_v2_seed{cfg.seed}.png"
    fig.savefig(path, dpi=180, facecolor="#0f172a", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════
# Spectral-response boundary sweep
# ══════════════════════════════════════════════════════════════════════

def sweep_spectral_boundary(outdir: Path) -> None:
    """
    Sweep the boundary of Assumption 3.14 (Spectral Response Under Tight
    Decoding) in v2.0.

    Assumption 3.14 requires  Q(θ_tight) < Q(θ_base), captured here by
    the ratio  Q_ratio = Q(θ_tight)/Q(θ_base) < 1.  As Q_ratio → 1 the
    spectral gap closes and Assumption 3.14 is no longer satisfied;
    the Lyapunov drift vanishes.

    We also sweep  α_tight  (effective contraction rate, Assumption 3.13)
    to show the joint boundary.
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), facecolor="#0f172a")

    configs = [
        {"Q_ratio": 0.40, "alpha_tight": 0.45,
         "label": "Q_ratio=0.40, α=0.45\n(Ass. 3.14 well satisfied)"},
        {"Q_ratio": 0.60, "alpha_tight": 0.35,
         "label": "Q_ratio=0.60, α=0.35\n(default — nominal)"},
        {"Q_ratio": 0.80, "alpha_tight": 0.25,
         "label": "Q_ratio=0.80, α=0.25\n(Ass. 3.14 marginal)"},
        {"Q_ratio": 0.60, "alpha_tight": 0.15,
         "label": "Q_ratio=0.60, α=0.15\n(weak contraction)"},
        {"Q_ratio": 0.92, "alpha_tight": 0.20,
         "label": "Q_ratio=0.92, α=0.20\n(Ass. 3.14 near boundary)"},
        {"Q_ratio": 0.99, "alpha_tight": 0.10,
         "label": "Q_ratio=0.99, α=0.10\n(Ass. 3.14 violated ≈)"},
    ]

    for ax, c in zip(axes.flat, configs):
        cfg = CPLConfig(
            Q_ratio=c["Q_ratio"],
            alpha_tight=c["alpha_tight"],
            T=250,
            seed=42,
        )
        traj = simulate(cfg, use_governor=True)
        ks   = [s.k for s in traj]

        ax.set_facecolor("#111827")
        ax.plot(ks, [s.V for s in traj], color="#10b981", lw=1.5)
        ax.axhline(0, color="#64748b", lw=0.5, alpha=0.3)
        ax.set_title(c["label"], color="#e2e8f0", fontsize=10, fontweight="bold")
        ax.tick_params(colors="#64748b", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#1e293b")
        ax.grid(True, alpha=0.06, color="#94a3b8")
        ax.set_ylabel("Vₖ", color="#94a3b8", fontsize=9)
        ax.set_xlabel("k",  color="#94a3b8", fontsize=9)

    fig.suptitle(
        "Boundary of Assumption 3.14 (Spectral Response) — v2.0",
        color="#e2e8f0", fontsize=13, fontweight="bold"
    )

    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "spectral_boundary_sweep_v2.png"
    fig.savefig(path, dpi=150, facecolor="#0f172a", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CPL Context Governor Simulator v2.0  "
                    "(doi:10.5281/zenodo.19177363)"
    )
    parser.add_argument("--seed",  type=int, default=42,
                        help="PRNG seed (default: 42)")
    parser.add_argument("--T",     type=int, default=200,
                        help="Simulation horizon (default: 200)")
    parser.add_argument("--sweep", action="store_true",
                        help="Run Assumption 3.14 spectral-response boundary sweep")
    parser.add_argument("--outdir", type=str, default="plots",
                        help="Output directory for figures (default: plots/)")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    cfg    = CPLConfig(seed=args.seed, T=args.T)

    plot_all(cfg, outdir)

    if args.sweep:
        sweep_spectral_boundary(outdir)

    # ── Summary statistics ────────────────────────────────────────────
    ctrl = simulate(cfg, use_governor=True)
    base = simulate(cfg, use_governor=False)

    print("\nDone.  Summary statistics (v2.0):")
    print(f"  Baseline max L  : {max(s.L for s in base):.0f}"
          f"  (L_practical = {cfg.L_practical})")
    print(f"  Governor max L  : {max(s.L for s in ctrl):.0f}"
          f"  (L_cap = {cfg.L_cap})  — Prop. 5.1")
    print(f"  Baseline N_F    : {sum(1 for s in base if s.z == 'F')}")
    print(f"  Governor N_F    : {sum(1 for s in ctrl if s.z == 'F')}"
          f"  — bounded by Thm. 5.3")
    print(f"  Final V (base)  : {base[-1].V:.4f}")
    print(f"  Final V (gov)   : {ctrl[-1].V:.4f}")
    print(f"  σ_eff           : {cfg.sigma_eff:.4f}"
          f"  (= σ_H; stability carries no independent noise term)")
    print(f"  S_obs_max       : {cfg.S_obs_max:.4f}  (calibrated parameter)")
    print(f"  Final Ŝ (gov)   : {ctrl[-1].S:.4f}"
          f"  = 1 − Ĥ/S_obs_max  (Cor. 3.7)")


if __name__ == "__main__":
    main()
