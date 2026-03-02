#!/usr/bin/env python3
"""
CPL Context Governor — Discrete Simulator
==========================================
Implements Eqs. (9), (11), (14) and Policy §4 from:
  Khomyakov, V. (2026). Phase-Aware Context Window Governance for LLMs.
  doi:10.5281/zenodo.18784361

Usage:
    python simulator.py              # Generate all plots
    python simulator.py --seed 42    # Specific seed
    python simulator.py --sweep      # Parameter sweep (boundary of Lemma 5.3)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════
# Configuration (all thresholds from the paper)
# ══════════════════════════════════════════════════════════════════════

@dataclass
class CPLConfig:
    """Parameters corresponding to §3–§4 of the paper."""
    # Context thresholds (Eq. 15)
    L_recover: int = 2000
    L_warn: int = 3200
    L_cap: int = 4000
    L_practical: int = 4800
    L_max: int = 5000
    L_safe: int = 1500

    # Input bounds — Assumption 3.1 (U), (Y)
    U_max: int = 400
    Y_max_tight: int = 350

    # Entropy dynamics (Eq. 14)
    alpha_base: float = 0.15
    alpha_tight: float = 0.35
    H_c: float = 1.099       # ln(3) nats
    eta: float = 0.0003       # slope of h(L, θ)
    delta_H: float = 0.25     # regime penalty at L >= L_practical

    # Stability dynamics (Eq. 11)
    rho: float = 0.92
    b_C: float = 0.06
    b_R: float = 0.03
    b_F: float = 0.05
    kappa_L: float = 0.00004
    kappa_P: float = 0.15
    S_c: float = 0.7

    # Phase classifier (Eq. 8)
    gamma: float = 0.1
    beta_hyst: float = 0.05

    # Noise std
    sigma_H: float = 0.04
    sigma_S: float = 0.025

    # Rescue releases (Eqs. 16–17)
    r_rescue: int = 1550
    r_recover: int = 1950

    # Simulation
    T: int = 200
    seed: int = 42

    def h_target(self, L: float) -> float:
        """Design function h(L, θ_tight) = H_c - η·(L_cap - L)_+  (Eq. 12–13)."""
        return self.H_c - self.eta * max(self.L_cap - L, 0)


# ══════════════════════════════════════════════════════════════════════
# Simulator
# ══════════════════════════════════════════════════════════════════════

@dataclass
class StepRecord:
    k: int
    L: float
    H: float
    S: float
    D: float
    z: str        # "C", "R", "F"
    V: float      # Lyapunov potential (Eq. 23)
    m: str        # "keep", "summarize", "chunk"
    theta: str    # "base", "tight"
    tau: float    # latency


def simulate(cfg: CPLConfig, use_governor: bool) -> list[StepRecord]:
    """Run the discrete-time simulation.

    Parameters
    ----------
    cfg : CPLConfig
    use_governor : bool
        If True, apply policy (18)–(19). Otherwise baseline (no action, base θ).
    """
    rng = np.random.RandomState(cfg.seed)

    L = 800.0
    H = 0.6
    S = 0.85
    H_prev = H
    z_prev = "C"

    trajectory = []

    for k in range(cfg.T + 1):
        D = abs(H - H_prev)

        # Phase classifier (Eq. 8)
        H_c_eff = cfg.H_c + (cfg.beta_hyst if z_prev == "C" else 0.0)
        if D >= cfg.gamma:
            z = "R"
        elif H < H_c_eff and S > cfg.S_c:
            z = "C"
        else:
            z = "F"

        # Lyapunov potential (Eq. 23)
        V = max(H - cfg.H_c, 0) + max(cfg.S_c - S, 0)

        # Policy (Eqs. 18–19)
        if use_governor:
            if z == "F":
                m, r = "chunk", cfg.r_recover
            elif z == "R":
                m, r = "summarize", cfg.r_rescue
            elif z == "C" and L > cfg.L_warn:
                m, r = "summarize", cfg.r_rescue
            else:
                m, r = "keep", 0
            theta = "tight" if (z in ("R", "F") or L > cfg.L_warn) else "base"
        else:
            m, r = "keep", 0
            theta = "base"

        alpha = cfg.alpha_tight if theta == "tight" else cfg.alpha_base

        # Latency (Eq. 10)
        g_L = 1e-5 * L * L
        tau_throttle = 2.0 if L >= cfg.L_practical else 0.0
        tau = 0.1 + g_L + tau_throttle

        trajectory.append(StepRecord(k=k, L=L, H=H, S=S, D=D, z=z, V=V, m=m, theta=theta, tau=tau))

        if k == cfg.T:
            break

        # ── State update ──
        U_k = rng.randint(int(cfg.U_max * 0.3), cfg.U_max + 1)
        Y_k = rng.randint(int(cfg.Y_max_tight * 0.2), cfg.Y_max_tight + 1)
        eps_H = cfg.sigma_H * rng.randn()
        eps_S = cfg.sigma_S * rng.randn()

        # Context (Eq. 9)
        L_next = np.clip(L + U_k + Y_k - r, 0, cfg.L_max)

        # Entropy (Eq. 14)
        H_tgt = cfg.h_target(L)
        Delta_H = cfg.delta_H if L >= cfg.L_practical else 0.0
        H_next = H - alpha * (H - H_tgt) + Delta_H + eps_H

        # Stability (Eq. 11)
        b_z = cfg.b_C if z == "C" else (-cfg.b_R if z == "R" else -cfg.b_F)
        S_next = np.clip(
            cfg.rho * S + b_z - cfg.kappa_L * max(L - cfg.L_safe, 0)
            - cfg.kappa_P * (1.0 if L >= cfg.L_practical else 0.0) + eps_S,
            0.0, 1.0
        )

        H_prev = H
        z_prev = z
        L, H, S = float(L_next), float(H_next), float(S_next)

    return trajectory


def cumulative_NF(traj: list[StepRecord]) -> np.ndarray:
    """Cumulative count of Fragmentation steps."""
    return np.cumsum([1 if s.z == "F" else 0 for s in traj])


def theoretical_bound(V0: float, delta: float, sigma: float, T: int, delta0: float = 0.05) -> np.ndarray:
    """Eq. 24: V0/δ + (σ/δ)√(2T ln(1/δ₀))."""
    ks = np.arange(T + 1, dtype=float)
    ks[0] = 1  # avoid log(0)
    return V0 / delta + (sigma / delta) * np.sqrt(2 * ks * np.log(1 / delta0))


# ══════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════

PHASE_COLORS = {"C": "#10b981", "R": "#f59e0b", "F": "#ef4444"}


def _phase_bar(ax, traj, y_pos, height=0.02):
    """Draw a thin color bar of phases below the x-axis."""
    for s in traj:
        ax.axvspan(s.k - 0.5, s.k + 0.5, ymin=0, ymax=height,
                   color=PHASE_COLORS[s.z], alpha=0.6, lw=0)


def plot_all(cfg: CPLConfig, outdir: Path):
    """Generate all 6 visualization panels."""
    ctrl = simulate(cfg, use_governor=True)
    base = simulate(cfg, use_governor=False)

    ks = [s.k for s in ctrl]
    T = cfg.T

    fig = plt.figure(figsize=(18, 22), facecolor="#0f172a")
    gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.28,
                           left=0.07, right=0.95, top=0.95, bottom=0.04)

    common_kw = dict(color="#e2e8f0", fontsize=10)
    ax_kw = dict(facecolor="#0f172a")

    def style_ax(ax, title, ylabel):
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
    ax.plot(ks, [s.L for s in ctrl], color="#10b981", lw=2, label="CPL-Governor")
    ax.axhline(cfg.L_warn, color="#f59e0b", ls="--", lw=1, alpha=0.7, label="L_warn")
    ax.axhline(cfg.L_cap, color="#ef4444", ls="--", lw=1.3, alpha=0.8, label="L_cap")
    ax.axhline(cfg.L_practical, color="#dc2626", ls="-.", lw=1.5, label="L_practical")
    ax.legend(fontsize=8, facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")
    _phase_bar(ax, ctrl, 0)

    # ── (B) Entropy ──────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1], **ax_kw)
    style_ax(ax, "(B) Entropy  Ĥₖ  —  Prop. 5.2", "Ĥₖ (nats)")
    ax.plot(ks, [s.H for s in base], color="#ef4444", lw=1.5, alpha=0.7, label="Baseline")
    ax.plot(ks, [s.H for s in ctrl], color="#10b981", lw=2, label="CPL-Governor")
    ax.axhline(cfg.H_c, color="#ef4444", ls="--", lw=1.3, alpha=0.8, label="H_c = ln(3)")
    ax.legend(fontsize=8, facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")
    _phase_bar(ax, ctrl, 0)

    # ── (C) Stability ────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0], **ax_kw)
    style_ax(ax, "(C) Stability  Ŝₖ  —  Degradation vs Recovery", "Ŝₖ")
    ax.plot(ks, [s.S for s in base], color="#ef4444", lw=1.5, alpha=0.7, label="Baseline")
    ax.plot(ks, [s.S for s in ctrl], color="#10b981", lw=2, label="CPL-Governor")
    ax.axhline(cfg.S_c, color="#3b82f6", ls="--", lw=1.3, alpha=0.8, label="S_c = 0.7")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")
    _phase_bar(ax, ctrl, 0)

    # ── (D) Lyapunov Potential ───────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1], **ax_kw)
    style_ax(ax, "(D) Lyapunov Potential  Vₖ  —  Drift (Assumption D)", "Vₖ")
    ax.plot(ks, [s.V for s in base], color="#ef4444", lw=1.5, alpha=0.7, label="Baseline Vₖ")
    ax.plot(ks, [s.V for s in ctrl], color="#10b981", lw=2, label="Governor Vₖ")
    ax.axhline(0, color="#10b981", lw=0.8, alpha=0.3)
    ax.legend(fontsize=8, facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")
    _phase_bar(ax, ctrl, 0)

    # ── (E) Fragmentation Occupancy ──────────────────────────────────
    ax = fig.add_subplot(gs[2, 0], **ax_kw)
    style_ax(ax, "(E) Fragmentation Occupancy  N_F(T)  —  Thm. 5.4", "N_F(T)")
    nf_base = cumulative_NF(base)
    nf_ctrl = cumulative_NF(ctrl)
    V0 = max(ctrl[0].V, 0.1)
    # σ_eff = √(σ_H² + σ_S²) — composite sub-Gaussian parameter of ξ_{k+1}
    # per Lemma 5.3: ξ = −ε^H·𝟙{Ĥ>Hc} + ε^S·𝟙{Ŝ<Sc}, independent components
    sigma_eff = np.sqrt(cfg.sigma_H**2 + cfg.sigma_S**2)
    bound = theoretical_bound(V0, delta=0.05, sigma=sigma_eff, T=T, delta0=0.05)
    ax.plot(ks, nf_base, color="#ef4444", lw=1.5, alpha=0.7, label="Baseline N_F")
    ax.plot(ks, nf_ctrl, color="#10b981", lw=2, label="Governor N_F")
    ax.plot(ks, bound, color="#8b5cf6", lw=1.5, ls="--", label="Bound (δ₀=0.05)")
    ax.legend(fontsize=8, facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")

    # ── (F) Phase Diagram (Ĥ, Ŝ) ───────────────────────────────────
    ax = fig.add_subplot(gs[2, 1], **ax_kw)
    style_ax(ax, "(F) Phase Space  (Ĥ, Ŝ)", "Ŝₖ")
    ax.set_xlabel("Ĥₖ", **common_kw)
    ax.scatter([s.H for s in base], [s.S for s in base],
               c="#ef4444", alpha=0.3, s=12, label="Baseline", zorder=2)
    ax.scatter([s.H for s in ctrl], [s.S for s in ctrl],
               c="#10b981", alpha=0.5, s=16, label="CPL-Governor", zorder=3)
    ax.axvline(cfg.H_c, color="#ef4444", ls="--", lw=1.2, alpha=0.7)
    ax.axhline(cfg.S_c, color="#3b82f6", ls="--", lw=1.2, alpha=0.7)
    ax.text(0.2, 0.92, "COHERENCE", color="#10b981", fontsize=9, alpha=0.5, transform=ax.transAxes)
    ax.text(0.7, 0.08, "FRAGMENTATION", color="#ef4444", fontsize=9, alpha=0.5, transform=ax.transAxes)
    ax.set_xlim(-0.1, 2.8)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")

    fig.suptitle(
        "CPL Context Governor — Collapse Damping Simulation\n"
        f"seed={cfg.seed}  T={cfg.T}  σ_H={cfg.sigma_H}  ρ={cfg.rho}  α_tight={cfg.alpha_tight}",
        color="#94a3b8", fontsize=11, y=0.99
    )

    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"cpl_governor_seed{cfg.seed}.png"
    fig.savefig(path, dpi=180, facecolor="#0f172a", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════
# Parameter sweep — boundary of Lemma 5.3 condition (ii)
# ══════════════════════════════════════════════════════════════════════

def sweep_drift_boundary(outdir: Path):
    """Demonstrate where Assumption (D) breaks by sweeping b_F and kappa_L."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), facecolor="#0f172a")

    configs = [
        {"b_F": 0.02, "kappa_L": 0.00002, "label": "b_F=0.02, κ_L=2e-5\n(D holds)"},
        {"b_F": 0.05, "kappa_L": 0.00004, "label": "b_F=0.05, κ_L=4e-5\n(default)"},
        {"b_F": 0.12, "kappa_L": 0.00004, "label": "b_F=0.12, κ_L=4e-5\n(D marginal)"},
        {"b_F": 0.05, "kappa_L": 0.00015, "label": "b_F=0.05, κ_L=1.5e-4\n(D weakened)"},
        {"b_F": 0.15, "kappa_L": 0.0001, "label": "b_F=0.15, κ_L=1e-4\n(D breaks)"},
        {"b_F": 0.20, "kappa_L": 0.00015, "label": "b_F=0.20, κ_L=1.5e-4\n(D violated)"},
    ]

    for ax, c in zip(axes.flat, configs):
        cfg = CPLConfig(b_F=c["b_F"], kappa_L=c["kappa_L"], T=250, seed=42)
        traj = simulate(cfg, use_governor=True)
        ks = [s.k for s in traj]
        ax.set_facecolor("#111827")
        ax.plot(ks, [s.V for s in traj], color="#10b981", lw=1.5)
        ax.axhline(0, color="#64748b", lw=0.5, alpha=0.3)
        ax.set_title(c["label"], color="#e2e8f0", fontsize=10, fontweight="bold")
        ax.tick_params(colors="#64748b", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#1e293b")
        ax.grid(True, alpha=0.06, color="#94a3b8")
        ax.set_ylabel("Vₖ", color="#94a3b8", fontsize=9)
        ax.set_xlabel("k", color="#94a3b8", fontsize=9)

    fig.suptitle(
        "Boundary of Lemma 5.3 — When does Assumption (D) break?",
        color="#e2e8f0", fontsize=13, fontweight="bold"
    )
    path = outdir / "drift_boundary_sweep.png"
    fig.savefig(path, dpi=150, facecolor="#0f172a", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="CPL Context Governor Simulator")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--T", type=int, default=200)
    parser.add_argument("--sweep", action="store_true", help="Run drift boundary sweep")
    parser.add_argument("--outdir", type=str, default="plots")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    cfg = CPLConfig(seed=args.seed, T=args.T)

    plot_all(cfg, outdir)

    if args.sweep:
        sweep_drift_boundary(outdir)

    print("\nDone. Summary statistics:")
    ctrl = simulate(cfg, use_governor=True)
    base = simulate(cfg, use_governor=False)
    print(f"  Baseline max L:  {max(s.L for s in base):.0f}  (L_practical = {cfg.L_practical})")
    print(f"  Governor max L:  {max(s.L for s in ctrl):.0f}  (L_cap = {cfg.L_cap})")
    print(f"  Baseline N_F:    {sum(1 for s in base if s.z == 'F')}")
    print(f"  Governor N_F:    {sum(1 for s in ctrl if s.z == 'F')}")
    print(f"  Final V (base):  {base[-1].V:.4f}")
    print(f"  Final V (gov):   {ctrl[-1].V:.4f}")


if __name__ == "__main__":
    main()
