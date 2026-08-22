"""
=============================================================================
Needle-in-a-Haystack: Empirical Validation of the Logarithmic Collapse Law
=============================================================================

Paper: "Information-Geometric Context Window Governance and the
        Probabilistic Theory of Long-Context Collapse in Large Language Models"
Author: Vladimir Khomyakov

This script supports TWO modes:
  1. SIMULATION  — synthetic data, no model required, instant results
  2. REAL MODEL  — HuggingFace transformer (LLaMA / Mistral / GPT-2 etc.)

Outputs:
  - results.csv                 (raw trial data)
  - figure1_margin_scaling.pdf  (margin vs sqrt(2 log L))
  - figure2_failure_prob.pdf    (P(failure) vs log L)
  - figure3_collapse_margin.pdf (P(failure) vs margin)
  - figure4_scatter.pdf         (scatter: margin × failure)
  - figure2b_corrected.pdf      (2nd-order EVT linearization)
  - figure2c_bias.pdf           (finite-size bias: 1st vs 2nd order)
  - figure2d_collapse.pdf       (EVT distributional collapse)
  - figure2e_qq.pdf             (QQ-plot vs Gumbel(0,1) + KS test)
  - figure2f_gev.pdf            (GEV shape xi + bootstrap CI)

Usage:
  # Simulation only (no GPU needed):
  python niah_experiment.py --mode simulation --n_trials 2000 --mu_s 3.0 --seed 42

  # Real model:
  python niah_experiment.py --mode real --model_name gpt2

  # Both:
  python niah_experiment.py --mode both --model_name gpt2
=============================================================================
"""

import argparse
import os
import random
import string
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.special import expit
from scipy.optimize import curve_fit
from scipy.stats import genextreme, kstest
from sklearn.isotonic import IsotonicRegression

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

# Context lengths to sweep
CONTEXT_LENGTHS = [128, 256, 512, 1024, 2048, 4096, 8192]

# Trials per context length
N_TRIALS = 2000

# Signal strength (used in simulation to calibrate logits)
# mu_s=4.0 gives a clear transition at L~2k–8k; decrease for earlier collapse
MU_S = 3.0        # signal strength (theory parameter)
SIGMA = 1.0       # sub-Gaussian noise parameter

# Key format: random 6-digit number
KEY_LENGTH = 6

# Output directory
OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)

# Plot style
STYLE = {
    "figure.figsize": (6, 4),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "lines.linewidth": 1.8,
    "lines.markersize": 5,
}
plt.rcParams.update(STYLE)

# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def random_key():
    """Generate a random numeric key."""
    return "".join(random.choices(string.digits, k=KEY_LENGTH))


def random_word(length=5):
    """Generate a random lowercase word."""
    return "".join(random.choices(string.ascii_lowercase, k=length))


def build_context(total_tokens: int, key: str) -> tuple[str, int]:
    """
    Build a synthetic 'haystack' context of approximately `total_tokens` words,
    with KEY inserted at a random position.

    Returns:
        context (str): full text
        key_position (int): approximate token index of the key
    """
    words = [random_word() for _ in range(total_tokens - 1)]
    insert_pos = random.randint(0, len(words))
    words.insert(insert_pos, f"KEY:{key}")
    return " ".join(words), insert_pos


# ─────────────────────────────────────────────────────────────────────────────
# MODE 1: SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def simulate_logits(L: int, mu_s: float, sigma: float) -> tuple[float, np.ndarray]:
    """
    Simulate attention logits under the signal-noise model:
      z_{i*} ~ N(mu_s, sigma^2)     (relevant token)
      z_i    ~ N(0,   sigma^2)      (noise tokens, i != i*)

    Returns:
        z_rel   (float): logit of the relevant token
        z_noise (ndarray): logits of all noise tokens
    """
    z_rel = np.random.normal(mu_s, sigma)
    z_noise = np.random.normal(0.0, sigma, size=L - 1)
    return z_rel, z_noise


def run_simulation(lengths=CONTEXT_LENGTHS, n_trials=N_TRIALS,
                   mu_s=MU_S, sigma=SIGMA) -> pd.DataFrame:
    """
    Run the simulation experiment.
    For each (L, trial): simulate logits, compute margin, determine success.
    """
    print("\n[SIMULATION] Starting...")

    if n_trials < 1000:
        print(f"  [NOTE] n_trials={n_trials}. For monotonous P(failure) it is recommended"
              f" --n_trials 1000+. For small n, local inversions (noise) are possible.")
    records = []

    for L in lengths:
        for trial in range(n_trials):
            z_rel, z_noise = simulate_logits(L, mu_s, sigma)
            margin = z_rel - z_noise.max()
            # success = relevant token dominates (margin > 0)
            success = int(margin > 0)
            records.append({
                "mode": "simulation",
                "L": L,
                "trial": trial,
                "z_rel": z_rel,
                "z_noise_max": z_noise.max(),
                "margin": margin,
                "success": success,
            })

        n_success = sum(r["success"] for r in records if r["L"] == L)
        print(f"  L={L:5d}  P(success)={n_success/n_trials:.3f}")

    print("[SIMULATION] Done.\n")
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# MODE 2: REAL MODEL
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_name: str):
    """Load a HuggingFace causal LM and tokenizer."""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
    except ImportError:
        raise ImportError(
            "transformers and torch are required for real-model mode.\n"
            "Install: pip install transformers torch"
        )

    print(f"[REAL] Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        output_attentions=True,
        torch_dtype="auto",
    )
    model.eval()
    print(f"[REAL] Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, tokenizer


def extract_attention_logits(model, tokenizer, context: str, key_token_pos: int):
    """
    Run a forward pass and extract attention logits from the last layer, last head.

    Returns:
        z_rel   (float): mean attention logit at key_token_pos
        z_noise (ndarray): attention logits at all other positions
        pred_key (str): model's predicted key (greedy decode of last few tokens)
    """
    import torch

    inputs = tokenizer(context, return_tensors="pt", truncation=True, max_length=8192)
    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]

    # Clip key position to tokenized length
    key_pos = min(key_token_pos, seq_len - 2)

    with torch.no_grad():
        outputs = model(**inputs)

    # Extract attention weights: shape (batch, heads, seq, seq)
    # Use last layer, last head, last query token (attending to all keys)
    attn = outputs.attentions[-1][0]      # (heads, seq, seq)
    last_head = attn[-1]                  # (seq, seq)
    last_query = last_head[-1]            # (seq,)  — attention from last token

    # Convert attention weights back to approximate logits via log
    attn_np = last_query.cpu().numpy().astype(float)
    attn_np = np.clip(attn_np, 1e-12, None)
    logits = np.log(attn_np)  # log-attention as proxy for pre-softmax logit

    z_rel = logits[key_pos]
    z_noise = np.concatenate([logits[:key_pos], logits[key_pos+1:]])

    # Greedy generation of the answer
    prompt = context + " The KEY is:"
    gen_inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192)
    with torch.no_grad():
        gen_ids = model.generate(
            gen_inputs["input_ids"],
            max_new_tokens=10,
            do_sample=False,
        )
    generated = tokenizer.decode(gen_ids[0][gen_inputs["input_ids"].shape[1]:],
                                  skip_special_tokens=True)
    return z_rel, z_noise, generated.strip()


def run_real_model(model_name: str,
                   lengths=CONTEXT_LENGTHS,
                   n_trials=N_TRIALS) -> pd.DataFrame:
    """
    Run the real-model experiment using a HuggingFace model.
    """
    model, tokenizer = load_model(model_name)
    print(f"\n[REAL] Starting experiment: {model_name}")
    records = []

    for L in lengths:
        n_success = 0
        for trial in range(n_trials):
            key = random_key()
            context, key_pos = build_context(L, key)
            query_context = context + f"\nWhat is the KEY? Answer:"

            try:
                z_rel, z_noise, pred = extract_attention_logits(
                    model, tokenizer, query_context, key_pos
                )
                margin = float(z_rel - z_noise.max())
                success = int(key in pred)
            except Exception as e:
                print(f"    [warn] L={L} trial={trial} error: {e}")
                margin = float("nan")
                success = 0
                pred = ""

            n_success += success
            records.append({
                "mode": "real",
                "model": model_name,
                "L": L,
                "trial": trial,
                "z_rel": float(z_rel) if not isinstance(z_rel, float) else z_rel,
                "z_noise_max": float(z_noise.max()),
                "margin": margin,
                "success": success,
                "prediction": pred[:30],
            })

        print(f"  L={L:5d}  P(success)={n_success/n_trials:.3f}")

    print("[REAL] Done.\n")
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS & PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-L statistics."""
    agg = df.groupby("L").agg(
        mean_margin=("margin", "mean"),
        std_margin=("margin", "std"),
        failure_prob=("success", lambda x: 1 - x.mean()),
        n_trials=("success", "count"),
    ).reset_index()
    agg["se_failure"] = np.sqrt(
        agg["failure_prob"] * (1 - agg["failure_prob"]) / agg["n_trials"]
    )
    return agg


def gumbel_cdf(log_L, mu_s, sigma):
    """
    Gumbel risk law: P(F_L) ≈ 1 - exp(-exp(-(mu_s - sigma*sqrt(2*log(L))) / aL))
    where aL = sigma / sqrt(2 * log(L))
    Used for fitting.
    """
    L = np.exp(log_L)
    bL = sigma * np.sqrt(2 * np.log(np.maximum(L, 2)))
    aL = sigma / np.sqrt(2 * np.log(np.maximum(L, 2)) + 1e-9)
    z = (mu_s - bL) / (aL + 1e-9)
    return 1.0 - np.exp(-np.exp(-z))


def plot_figure1(agg: pd.DataFrame, label: str, suffix: str):
    """Figure 1: Mean margin vs sqrt(2 log L)."""
    L = agg["L"].values.astype(float)
    margin = agg["mean_margin"].values
    std = agg["std_margin"].values
    n = agg["n_trials"].values
    se = std / np.sqrt(n)

    theory = -np.sqrt(2 * np.log(L))

    fig, ax = plt.subplots()
    ax.errorbar(L, margin, yerr=1.96 * se,
                fmt="o-", color="#1f77b4", capsize=3,
                label=f"Empirical ({label})")
    ax.plot(L, theory, "--", color="#d62728",
            label=r"Theory: $-\sqrt{2\log L}$")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xscale("log")
    ax.set_xlabel("Context length $L$")
    ax.set_ylabel(r"Mean margin $\mu(L)$")
    ax.set_title("Margin scaling vs. context length")
    ax.legend()
    path = os.path.join(OUT_DIR, f"figure1_margin_scaling_{suffix}.pdf")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_figure2(agg: pd.DataFrame, label: str, suffix: str):
    """Figure 2: Failure probability vs log L, with Gumbel fit + isotonic smoothing."""
    L = agg["L"].values.astype(float)
    fp = agg["failure_prob"].values
    se = agg["se_failure"].values

    # Isotonic regression: ensures monotonic growth of P(failure) with L
    # Necessary for a small number of trials, when noise produces local inversions
    ir = IsotonicRegression(increasing=True, out_of_bounds="clip")
    fp_monotone = ir.fit_transform(np.log(L), fp)

    # Fit Gumbel CDF to isotonic smoothed values
    mask = (fp_monotone > 0.01) & (fp_monotone < 0.99)
    gumbel_fit = None
    if mask.sum() >= 3:
        try:
            popt, _ = curve_fit(
                gumbel_cdf,
                np.log(L[mask]), fp_monotone[mask],
                p0=[MU_S, SIGMA],
                bounds=([0.1, 0.01], [100.0, 100.0]),
                maxfev=5000,
            )
            gumbel_fit = popt
        except Exception:
            pass

    L_dense = np.logspace(np.log10(L.min()), np.log10(L.max()), 300)

    fig, ax = plt.subplots()
    # Raw points (with CI)
    ax.errorbar(L, fp, yerr=1.96 * se,
                fmt="o", color="#1f77b4", capsize=3, alpha=0.5,
                label=f"Empirical ({label})")
    # Isotonically smoothed curve
    ax.plot(L, fp_monotone, "o-", color="#1f77b4",
            label="Isotonic smoothing")
    if gumbel_fit is not None:
        fp_fit = gumbel_cdf(np.log(L_dense), *gumbel_fit)
        ax.plot(L_dense, fp_fit, "--", color="#d62728",
                label=r"Gumbel fit $\hat{\mu}_s$=%.2f, $\hat{\sigma}$=%.2f"
                      % tuple(gumbel_fit))
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Context length $L$")
    ax.set_ylabel(r"$P(\text{failure})$")
    ax.set_title("Failure probability vs. context length")
    ax.legend()
    path = os.path.join(OUT_DIR, f"figure2_failure_prob_{suffix}.pdf")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path, gumbel_fit


def plot_figure3(df: pd.DataFrame, label: str, suffix: str):
    """Figure 3: P(failure) vs margin (binned)."""
    df_clean = df.dropna(subset=["margin"])
    bins = np.linspace(df_clean["margin"].quantile(0.02),
                       df_clean["margin"].quantile(0.98), 30)
    df_clean = df_clean.copy()
    df_clean["margin_bin"] = pd.cut(df_clean["margin"], bins)
    grouped = df_clean.groupby("margin_bin", observed=False)["success"].agg(["mean", "count"])
    grouped["failure"] = 1 - grouped["mean"]
    grouped["se"] = np.sqrt(
        grouped["failure"] * (1 - grouped["failure"]) / grouped["count"]
    )
    centers = np.array([iv.mid for iv in grouped.index])

    # Logistic fit
    try:
        def logistic(x, x0, k):
            return expit(-k * (x - x0))
        popt, _ = curve_fit(logistic, centers, grouped["failure"].values,
                            p0=[0.0, 2.0], maxfev=5000)
        x_dense = np.linspace(centers.min(), centers.max(), 300)
        y_fit = logistic(x_dense, *popt)
        fit_label = r"Logistic fit ($x_0$=%.2f)" % popt[0]
    except Exception:
        x_dense, y_fit, fit_label = None, None, None

    fig, ax = plt.subplots()
    ax.errorbar(centers, grouped["failure"].values,
                yerr=1.96 * grouped["se"].values,
                fmt="o", color="#1f77b4", capsize=2, markersize=4,
                label=f"Empirical ({label})")
    if x_dense is not None:
        ax.plot(x_dense, y_fit, "--", color="#d62728", label=fit_label)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle=":",
               label=r"$\mu=0$ (collapse threshold)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(r"Margin $\mu = z_\mathrm{rel} - \max z_\mathrm{noise}$")
    ax.set_ylabel(r"$P(\text{failure})$")
    ax.set_title("Collapse transition vs. margin")
    ax.legend()
    path = os.path.join(OUT_DIR, f"figure3_collapse_margin_{suffix}.pdf")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_figure4(df: pd.DataFrame, label: str, suffix: str):
    """Figure 4: Scatter — margin × failure, coloured by L."""
    df_clean = df.dropna(subset=["margin"]).copy()
    sample = df_clean.sample(min(3000, len(df_clean)), random_state=42)

    fig, ax = plt.subplots()
    sc = ax.scatter(
        sample["margin"],
        1 - sample["success"],
        c=np.log2(sample["L"].astype(float)),
        cmap="viridis",
        alpha=0.4,
        s=8,
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(r"$\log_2 L$")
    ax.axvline(0, color="#d62728", linewidth=1.2, linestyle="--",
               label=r"$\mu = 0$")
    ax.set_xlabel(r"Margin $\mu$")
    ax.set_ylabel("Failure (0 / 1)")
    ax.set_title(f"Scatter: margin × failure ({label})")
    ax.legend()
    path = os.path.join(OUT_DIR, f"figure4_scatter_{suffix}.pdf")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# EVT SECOND-ORDER CORRECTIONS  (Figures 2b-corr, 2c, 2d, 2e, 2f)
# ─────────────────────────────────────────────────────────────────────────────

def _evt_correction(logL: np.ndarray) -> np.ndarray:
    """
    Second-order correction to E[max of L i.i.d. N(0,1)]:
        b_L^(2) = sqrt(2 log L) - (log log L + log(4π)) / (2 sqrt(2 log L))
    Valid for L ≥ 3 (log log L > 0).
    Reference: Leadbetter et al. (1983), standard Gaussian EVT asymptotics.
    """
    safe_logL = np.maximum(logL, np.log(3))          # guard against L < 3
    first  = np.sqrt(2 * safe_logL)
    second = (np.log(safe_logL) + np.log(4 * np.pi)) / (2 * first)
    return first - second                              # x_corr = b_L^(2) / sigma


def plot_figure2b_corrected(agg: pd.DataFrame, label: str, suffix: str,
                             mu_s: float = MU_S, sigma: float = SIGMA):
    """
    Figure 2b (corrected EVT):
    Mean margin vs second-order EVT predictor x_corr.

    Under the corrected model:
        mu(L) ≈ mu_s - sigma * x_corr
    so OLS should give intercept ≈ mu_s, slope ≈ -sigma — without fitting them.
    """
    L      = agg["L"].values.astype(float)
    mu_emp = agg["mean_margin"].values
    se     = agg["std_margin"].values / np.sqrt(agg["n_trials"].values)
    logL   = np.log(L)

    x_corr = _evt_correction(logL)          # second-order predictor

    # OLS (diagnostic only — parameters NOT estimated from data here)
    slope, intercept = np.polyfit(x_corr, mu_emp, 1)
    x_dense = np.linspace(x_corr.min(), x_corr.max(), 200)
    y_pred  = intercept + slope * x_corr
    r2 = 1 - np.sum((mu_emp - y_pred) ** 2) / np.sum((mu_emp - mu_emp.mean()) ** 2)

    fig, ax = plt.subplots()
    ax.errorbar(x_corr, mu_emp, yerr=1.96 * se,
                fmt="o", color="#1f77b4", capsize=3,
                label=f"Empirical ({label})")
    ax.plot(x_dense, intercept + slope * x_dense, "-", color="#1f77b4",
            label=fr"OLS: $\hat{{\mu}}_s={intercept:.3f}$, "
                  fr"$\hat{{\sigma}}={-slope:.3f}$, $R^2={r2:.4f}$")
    # Theory line — uses true generative parameters (no fitting)
    ax.plot(x_dense, mu_s - sigma * x_dense, "--", color="#d62728",
            label=fr"Theory ($\mu_s={mu_s}$, $\sigma={sigma}$): "
                  r"$\mu_s - \sigma\,x_{\mathrm{corr}}$")
    ax.axhline(0, color="gray", lw=0.8, ls=":")

    for xi, li, mi in zip(x_corr, L, mu_emp):
        ax.annotate(f"$L={int(li)}$", (xi, mi),
                    textcoords="offset points", xytext=(4, 4),
                    fontsize=7, color="#555")

    ax.set_xlabel(r"$x_{\mathrm{corr}} = \sqrt{2\log L} - "
                  r"\frac{\log\log L + \log(4\pi)}{2\sqrt{2\log L}}$")
    ax.set_ylabel(r"Mean margin $\mu(L)$")
    ax.set_title("Corrected EVT linearization (2nd-order)")
    ax.legend(fontsize=8)

    path = os.path.join(OUT_DIR, f"figure2b_corrected_{suffix}.pdf")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path, (intercept, slope, r2)


def plot_figure2c_bias(agg: pd.DataFrame, label: str, suffix: str,
                       mu_s: float = MU_S, sigma: float = SIGMA):
    """
    Figure 2c: Finite-size bias vs L.
    bias_k(L) = mu_emp(L) - mu_theory^(k)(L),  k = 1 (first-order), 2 (second-order).

    Expectation:
      bias_1(L) — large and systematic (positive, decreasing slowly)
      bias_2(L) — small and converging toward zero as L grows
    """
    L      = agg["L"].values.astype(float)
    mu_emp = agg["mean_margin"].values
    se     = agg["std_margin"].values / np.sqrt(agg["n_trials"].values)
    logL   = np.log(L)

    mu_1 = mu_s - sigma * np.sqrt(2 * logL)
    mu_2 = mu_s - sigma * _evt_correction(logL)

    bias_1 = mu_emp - mu_1
    bias_2 = mu_emp - mu_2

    fig, ax = plt.subplots()
    ax.errorbar(L, bias_1, yerr=1.96 * se, fmt="o-", capsize=3,
                color="#d62728", label=r"Bias: 1st-order EVT")
    ax.errorbar(L, bias_2, yerr=1.96 * se, fmt="s-", capsize=3,
                color="#2ca02c", label=r"Bias: 2nd-order EVT")
    ax.axhline(0, linestyle="--", linewidth=1, color="gray")
    ax.set_xscale("log")
    ax.set_xlabel("Context length $L$")
    ax.set_ylabel(r"$\mu_{\mathrm{emp}}(L) - \mu_{\mathrm{theory}}(L)$")
    ax.set_title("Finite-size bias: 1st vs 2nd order EVT")
    ax.legend()

    path = os.path.join(OUT_DIR, f"figure2c_bias_{suffix}.pdf")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")

    # Numeric summary
    print(f"  [bias summary]  1st-order: mean={bias_1.mean():.4f}  "
          f"2nd-order: mean={bias_2.mean():.4f}")
    return path, (bias_1, bias_2)


def plot_figure2d_collapse(df: pd.DataFrame, label: str, suffix: str,
                           mu_s: float = MU_S, sigma: float = SIGMA):
    """
    Figure 2d: EVT distributional collapse.
    After normalising each trial's margin as
        y = (mu - (mu_s - b_L_corr)) / a_L
    the empirical CDFs for all L should collapse onto the Gumbel(0,1) CDF.

    Uses second-order b_L (corrected) for centering.
    a_L = sigma / sqrt(2 log L)  (first-order scale — standard).
    """
    df_c = df.dropna(subset=["margin"]).copy()
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, df_c["L"].nunique()))

    fig, ax = plt.subplots()

    for (L_val, color) in zip(sorted(df_c["L"].unique()), colors):
        subset = df_c[df_c["L"] == L_val]
        mu     = subset["margin"].values
        logL   = np.log(float(L_val))

        b_L = sigma * _evt_correction(np.array([logL]))[0]  # corrected centering
        a_L = sigma / np.sqrt(2 * logL)                      # standard scale

        y = (mu - (mu_s - b_L)) / a_L
        y_sorted = np.sort(y)
        cdf = np.arange(1, len(y_sorted) + 1) / len(y_sorted)
        ax.plot(y_sorted, cdf, alpha=0.7, color=color, label=f"L={int(L_val)}")

    y_dense   = np.linspace(-5, 6, 400)
    gumbel_cdf_vals = np.exp(-np.exp(-y_dense))
    ax.plot(y_dense, gumbel_cdf_vals, "k--", linewidth=2.2,
            label="Gumbel(0,1) CDF")

    ax.set_xlabel("Normalised margin $y$ (EVT coordinates)")
    ax.set_ylabel("CDF")
    ax.set_title("EVT distributional collapse (2nd-order centering)")
    ax.legend(fontsize=7, ncol=2)

    path = os.path.join(OUT_DIR, f"figure2d_collapse_{suffix}.pdf")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_figure2e_qq(df: pd.DataFrame, label: str, suffix: str,
                     mu_s: float = MU_S, sigma: float = SIGMA):
    """
    Figure 2e: QQ-plot — pooled normalised margins vs Gumbel(0,1).
    All L are pooled after EVT normalisation (corrected centering).

    Caution: this is a visual/diagnostic tool.
    The KS p-value reported uses the pooled sample; because the normalisation
    parameters (mu_s, sigma) are treated as known (generative model), the test
    is not biased by estimation — but pooling across L introduces mild
    non-iid structure. Interpret p-value as indicative, not exact.
    """
    df_c  = df.dropna(subset=["margin"]).copy()
    y_all = []

    for L_val in df_c["L"].unique():
        subset = df_c[df_c["L"] == L_val]
        mu     = subset["margin"].values
        logL   = np.log(float(L_val))

        b_L = sigma * _evt_correction(np.array([logL]))[0]
        a_L = sigma / np.sqrt(2 * logL)
        y_all.append((mu - (mu_s - b_L)) / a_L)

    y_all   = np.concatenate(y_all)
    y_sorted = np.sort(y_all)
    n        = len(y_sorted)

    p        = (np.arange(1, n + 1) - 0.5) / n
    q_theory = -np.log(-np.log(p))           # Gumbel(0,1) quantiles

    # KS test (parameters known from generative model — not estimated from data)
    ks_stat, ks_p = kstest(y_all, lambda x: np.exp(-np.exp(-x)))

    fig, ax = plt.subplots()
    ax.scatter(q_theory, y_sorted, s=4, alpha=0.3, color="#1f77b4",
               label="Empirical quantiles")
    q_min, q_max = q_theory.min(), q_theory.max()
    ax.plot([q_min, q_max], [q_min, q_max], "--", color="#d62728",
            linewidth=2, label="Ideal: Gumbel(0,1)")
    ax.set_xlabel("Theoretical Gumbel quantiles")
    ax.set_ylabel("Empirical quantiles")
    ax.set_title(f"QQ-plot vs Gumbel(0,1)\n"
                 f"KS stat={ks_stat:.4f}, p={ks_p:.4f}  ({label})")
    ax.legend()

    path = os.path.join(OUT_DIR, f"figure2e_qq_{suffix}.pdf")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}  [KS stat={ks_stat:.4f}, p={ks_p:.4f}]")
    return path, (ks_stat, ks_p)


def plot_figure2f_gev(df: pd.DataFrame, label: str, suffix: str,
                      mu_s: float = MU_S, sigma: float = SIGMA,
                      n_bootstrap: int = 500):
    """
    Figure 2f: GEV shape parameter estimation (EVT class identification).
    Fits GEV(xi, loc, scale) to pooled normalised margins.

    xi ≈ 0  → Gumbel (light tail)   — predicted by theory
    xi > 0  → Fréchet (heavy tail)  — would refute Gaussian noise assumption

    Bootstrap CI is computed over xi to make the claim quantitative.
    Note: scipy uses the sign convention shape = -xi.
    """
    df_c  = df.dropna(subset=["margin"]).copy()
    y_all = []

    for L_val in df_c["L"].unique():
        subset = df_c[df_c["L"] == L_val]
        mu     = subset["margin"].values
        logL   = np.log(float(L_val))
        b_L    = sigma * _evt_correction(np.array([logL]))[0]
        a_L    = sigma / np.sqrt(2 * logL)
        y_all.append((mu - (mu_s - b_L)) / a_L)

    y_all = np.concatenate(y_all)

    # Point estimate
    c_hat, loc_hat, scale_hat = genextreme.fit(y_all)
    xi_hat = -c_hat          # xi = -c in scipy convention

    # Parametric bootstrap CI for xi
    rng      = np.random.default_rng(42)
    xi_boot  = []
    for _ in range(n_bootstrap):
        sample = genextreme.rvs(c_hat, loc=loc_hat, scale=scale_hat,
                                size=len(y_all), random_state=rng)
        try:
            cb, _, _ = genextreme.fit(sample)
            xi_boot.append(-cb)
        except Exception:
            pass
    xi_boot  = np.array(xi_boot)
    xi_lo    = np.percentile(xi_boot, 2.5)
    xi_hi    = np.percentile(xi_boot, 97.5)

    print(f"  [GEV] xi={xi_hat:.4f}  95% CI=[{xi_lo:.4f}, {xi_hi:.4f}]  "
          f"loc={loc_hat:.4f}  scale={scale_hat:.4f}")

    # QQ vs fitted GEV
    y_sorted = np.sort(y_all)
    n        = len(y_sorted)
    p        = (np.arange(1, n + 1) - 0.5) / n
    q_gev    = genextreme.ppf(p, c_hat, loc=loc_hat, scale=scale_hat)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left panel: QQ vs fitted GEV
    ax = axes[0]
    ax.scatter(q_gev, y_sorted, s=4, alpha=0.3, color="#1f77b4",
               label="Empirical vs fitted GEV")
    q_min, q_max = q_gev.min(), q_gev.max()
    ax.plot([q_min, q_max], [q_min, q_max], "--", color="#d62728",
            linewidth=2, label="Ideal")
    ax.set_xlabel("GEV theoretical quantiles")
    ax.set_ylabel("Empirical quantiles")
    ax.set_title(f"QQ vs fitted GEV\n"
                 fr"$\xi={xi_hat:.3f}$ [{xi_lo:.3f}, {xi_hi:.3f}]")
    ax.legend(fontsize=8)

    # Right panel: bootstrap distribution of xi
    ax2 = axes[1]
    ax2.hist(xi_boot, bins=40, color="#1f77b4", alpha=0.7, density=True)
    ax2.axvline(xi_hat, color="#d62728", linewidth=2, label=fr"$\xi={xi_hat:.3f}$")
    ax2.axvline(0,      color="gray",    linewidth=1.5, linestyle="--", label=r"$\xi=0$ (Gumbel)")
    ax2.axvline(xi_lo,  color="#ff7f0e", linewidth=1, linestyle=":",  label="95% CI")
    ax2.axvline(xi_hi,  color="#ff7f0e", linewidth=1, linestyle=":")
    ax2.set_xlabel(r"Bootstrap $\xi$")
    ax2.set_ylabel("Density")
    ax2.set_title(f"Bootstrap CI for GEV shape $\\xi$ ({label})")
    ax2.legend(fontsize=8)

    path = os.path.join(OUT_DIR, f"figure2f_gev_{suffix}.pdf")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path, (xi_hat, xi_lo, xi_hi)


def print_evt_table(results: dict):
    """Print a summary table of all EVT diagnostic results."""
    print("\n" + "=" * 65)
    print("  EVT DIAGNOSTIC TABLE")
    print("=" * 65)
    rows = [
        ("Scaling (R²)",       f"{results['r2_1st']:.4f}",
         "✓ EVT structure holds" if results['r2_1st'] > 0.99 else "– check"),
        ("Corrected scaling R²", f"{results['r2_2nd']:.4f}",
         "✓ 2nd-order consistent" if results['r2_2nd'] >= results['r2_1st'] - 0.001 else "– marginal"),
        ("Intercept (1st)",    f"{results['ic_1st']:.3f}",
         f"bias={results['ic_1st'] - results['mu_s']:+.3f}"),
        ("Intercept (2nd)",    f"{results['ic_2nd']:.3f}",
         f"bias={results['ic_2nd'] - results['mu_s']:+.3f}  (reduced)"),
        ("Mean bias 1st",      f"{results['mean_bias_1']:.4f}", ""),
        ("Mean bias 2nd",      f"{results['mean_bias_2']:.4f}", "smaller → finite-size"),
        ("KS p-value",         f"{results['ks_p']:.4f}",
         "✓ Gumbel not rejected" if results['ks_p'] > 0.05 else "⚠ Gumbel rejected"),
        ("GEV xi",             f"{results['xi']:.4f}  "
                               f"[{results['xi_lo']:.4f}, {results['xi_hi']:.4f}]",
         "✓ Gumbel class (xi≈0)" if abs(results['xi']) < 0.1 else "⚠ non-Gumbel"),
    ]
    fmt = "  {:<22} {:<28} {}"
    print(fmt.format("Test", "Result", "Conclusion"))
    print("  " + "-" * 61)
    for r in rows:
        print(fmt.format(*r))
    print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_and_plot(df: pd.DataFrame, label: str, suffix: str,
                 mu_s: float = MU_S, sigma: float = SIGMA):
    """Aggregate, plot all figures, save CSV."""
    csv_path = os.path.join(OUT_DIR, f"results_{suffix}.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    agg = aggregate(df)

    print("\n[PLOTS] Generating original figures...")
    plot_figure1(agg, label, suffix)
    _, gumbel_fit = plot_figure2(agg, label, suffix)
    plot_figure3(df, label, suffix)
    plot_figure4(df, label, suffix)

    print("\n[PLOTS] Generating EVT diagnostic figures...")

    # Figure 2b corrected — first-order baseline for comparison
    L      = agg["L"].values.astype(float)
    mu_emp = agg["mean_margin"].values
    logL   = np.log(L)
    x_1st  = np.sqrt(2 * logL)
    sl_1, ic_1 = np.polyfit(x_1st, mu_emp, 1)
    y_1st  = ic_1 + sl_1 * x_1st
    r2_1st = 1 - np.sum((mu_emp - y_1st) ** 2) / np.sum((mu_emp - mu_emp.mean()) ** 2)

    _, (ic_2nd, sl_2nd, r2_2nd) = plot_figure2b_corrected(
        agg, label, suffix, mu_s=mu_s, sigma=sigma)

    _, (bias_1, bias_2) = plot_figure2c_bias(
        agg, label, suffix, mu_s=mu_s, sigma=sigma)

    plot_figure2d_collapse(df, label, suffix, mu_s=mu_s, sigma=sigma)

    _, (ks_stat, ks_p) = plot_figure2e_qq(
        df, label, suffix, mu_s=mu_s, sigma=sigma)

    _, (xi, xi_lo, xi_hi) = plot_figure2f_gev(
        df, label, suffix, mu_s=mu_s, sigma=sigma)

    print("\n[SUMMARY]")
    print(agg[["L", "mean_margin", "failure_prob", "n_trials"]].to_string(index=False))

    print_evt_table({
        "mu_s":        mu_s,
        "r2_1st":      r2_1st,
        "r2_2nd":      r2_2nd,
        "ic_1st":      ic_1,
        "ic_2nd":      ic_2nd,
        "mean_bias_1": float(bias_1.mean()),
        "mean_bias_2": float(bias_2.mean()),
        "ks_p":        ks_p,
        "xi":          xi,
        "xi_lo":       xi_lo,
        "xi_hi":       xi_hi,
    })


def main():
    parser = argparse.ArgumentParser(
        description="NIAH Experiment: Logarithmic Collapse Law Validation"
    )
    parser.add_argument(
        "--mode",
        choices=["simulation", "real", "both"],
        default="simulation",
        help="Experiment mode (default: simulation)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2",
        help="HuggingFace model name for real mode (default: gpt2)",
    )
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=CONTEXT_LENGTHS,
        help="Context lengths to sweep",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=N_TRIALS,
        help="Number of trials per context length",
    )
    parser.add_argument(
        "--mu_s",
        type=float,
        default=MU_S,
        help="Signal strength for simulation (default: 3.0)",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=SIGMA,
        help="Noise std for simulation (default: 1.0)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    print("=" * 65)
    print("  NIAH Experiment — Logarithmic Collapse Law Validation")
    print("=" * 65)
    print(f"  Mode      : {args.mode}")
    print(f"  Lengths   : {args.lengths}")
    print(f"  Trials    : {args.n_trials}")
    print(f"  Output    : {OUT_DIR}/")
    print("=" * 65)

    if args.mode in ("simulation", "both"):
        df_sim = run_simulation(
            lengths=args.lengths,
            n_trials=args.n_trials,
            mu_s=args.mu_s,
            sigma=args.sigma,
        )
        run_and_plot(df_sim, label="simulation", suffix="sim",
                     mu_s=args.mu_s, sigma=args.sigma)

    if args.mode in ("real", "both"):
        df_real = run_real_model(
            model_name=args.model_name,
            lengths=args.lengths,
            n_trials=args.n_trials,
        )
        run_and_plot(df_real, label=args.model_name, suffix="real")

    print("\n✓ All outputs saved to:", os.path.abspath(OUT_DIR))


if __name__ == "__main__":
    main()
