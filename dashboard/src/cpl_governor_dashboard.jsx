import { useState, useMemo } from "react";
import {
  Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, ReferenceLine, Scatter, ScatterChart,
  ComposedChart,
} from "recharts";

// ─────────────────────────────────────────────────────────────────────────────
// Simulation Engine — v2.0
// Implements Eqs. (9), (14) and Policy §4 from:
//   Khomyakov, V. (2026). Information-Geometric Context Window Governance
//   for Large Language Models via Observer Entropy and the Cognitive Phase
//   Law (CPL 4.0). Version 2.0. doi:10.5281/zenodo.19177363
//
// Key changes relative to v1.0 dashboard
// ─────────────────────────────────────────
// [FIX-1] Stability Ŝ_k is NO LONGER an independent stochastic process.
//         Per Definition 3.5 and Corollary 3.7, stability is algebraically
//         coupled to entropy:
//             Ŝ_k = 1 − Ĥ_k / S_obs_max
//         The v1.0 parameters rho, b_C, b_R, b_F, kappa_L, kappa_P, sigma_S
//         are REMOVED from DEFAULT_PARAMS and simulate().
//
// [FIX-2] σ_eff = σ_H only (not √(σ_H² + σ_S²)).
//         In v2.0, ξ_{k+1} = −ε_k^H · 𝟙{Ĥ_k > H_c} is a scalar noise term
//         (Lemma 4.3). The independent stability noise σ_S is removed.
//
// [FIX-3] DEFAULT_PARAMS now includes S_obs_max (Def. 3.4) and Q_ratio
//         (Assumption 3.14) instead of the removed SDE parameters.
//
// [FIX-4] Theorem/Lemma references updated to v2.0 numbering:
//         "Lemma 5.3" → "Lemma 4.3" (composite SG)
//         "Thm. 5.4"  → "Thm. 5.3"  (fragmentation bound)
//
// [FIX-5] Assumption reference updated from "Assumption 3.1" to
//         "Assumption 4.1 (U), (Y)" per v2.0 numbering.
//
// [FIX-6] Phase diagram (Ĥ, Ŝ) now shows the algebraic manifold
//         Ŝ = 1 − Ĥ / S_obs_max as a dashed reference line.
//
// [FIX-7] Stability tab subtitle updated to reflect algebraic coupling
//         and Corollary 3.7 reference.
// ─────────────────────────────────────────────────────────────────────────────

const DEFAULT_PARAMS = {
  // Context thresholds — Assumption 4.1 (T), Eq. (15)
  L_recover:   2000,
  L_warn:      3200,
  L_cap:       4000,
  L_practical: 4800,
  L_max:       5000,
  L_safe:      1500,  // kept for latency formula only

  // Input bounds — Assumption 4.1 (U), (Y)  [FIX-5]
  U_max:       400,
  Y_max_tight: 350,

  // Entropy dynamics — Eq. (14)
  alpha_base:  0.15,
  alpha_tight: 0.35,   // α_eff under θ_tight — Assumption 3.13
  H_c:         1.099,  // H_c = ln(3) nats
  eta:         0.0003, // slope of h(L, θ_tight) — Assumption 4.1(h)
  delta_H:     0.25,   // regime penalty at L ≥ L_practical

  // Information-geometric parameters (v2.0) — [FIX-3]
  // S_obs_max: supremum of observer entropy over compact K (Def. 3.4).
  // Default: H_c × 1.5 ≈ 1.648, so S_c = 0.7 ↔ Ĥ/S_obs_max = 0.3
  S_obs_max:   1.648,

  // Q_ratio = Q(θ_tight) / Q(θ_base) — Assumption 3.14 (Spectral Response).
  // Must be < 1 for tight decoding to reduce observer entropy.
  Q_ratio:     0.60,

  // Phase classifier — Eq. (8)
  S_c:         0.70,
  gamma:       0.10,
  beta_hyst:   0.05,

  // Noise — Assumption 4.1 (ε')
  // Only σ_H remains; σ_S removed because stability is algebraically
  // determined by entropy (Corollary 3.7).  [FIX-1]
  sigma_H:     0.04,

  // Rescue releases — Eqs. (16)–(17)
  r_rescue:    1550,
  r_recover:   1950,

  // Simulation
  T:    200,
  seed: 42,
};

// ─── Seeded PRNG (Mulberry32) ─────────────────────────────────────────────────
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussianRandom(rng) {
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

// ─── Core simulator ───────────────────────────────────────────────────────────
function simulate(params, useGovernor) {
  const p = { ...DEFAULT_PARAMS, ...params };
  const rng = mulberry32(p.seed);

  // Design function h(L, θ_tight) = H_c − η·(L_cap − L)₊
  // Satisfies h(L, θ_tight) < H_c for all L ≤ L_cap — Assumption 4.1(h)
  const h_target = (L) => p.H_c - p.eta * Math.max(p.L_cap - L, 0);

  // Initial state
  let L = 800;
  let H = 0.60;
  // Algebraic stability — Corollary 3.7  [FIX-1]
  let S = Math.max(0, Math.min(1, 1 - H / p.S_obs_max));
  let H_prev = H;
  let z_prev = "C";

  const trajectory = [];

  for (let k = 0; k <= p.T; k++) {
    const D = Math.abs(H - H_prev);

    // Phase classifier (Eq. 8) — "Reorganization first"
    const H_c_eff = p.H_c + (z_prev === "C" ? p.beta_hyst : 0);
    let z;
    if (D >= p.gamma)                   { z = "R"; }
    else if (H < H_c_eff && S > p.S_c) { z = "C"; }
    else                                { z = "F"; }

    // Lyapunov potential — Eq. (23)
    const V = Math.max(H - p.H_c, 0) + Math.max(p.S_c - S, 0);

    // Policy — Eqs. (18)–(19)
    let m, r, theta_mode;
    if (useGovernor) {
      if      (z === "F")                    { m = "chunk";     r = p.r_recover; }
      else if (z === "R")                    { m = "summarize"; r = p.r_rescue;  }
      else if (z === "C" && L > p.L_warn)   { m = "summarize"; r = p.r_rescue;  }
      else                                   { m = "keep";      r = 0;           }
      theta_mode = (z === "R" || z === "F" || L > p.L_warn) ? "tight" : "base";
    } else {
      m = "keep"; r = 0;
      theta_mode = "base";
    }

    // Effective contraction rate — Assumption 3.13
    const alpha = theta_mode === "tight" ? p.alpha_tight : p.alpha_base;

    // Latency — Eq. (10)
    const g_L         = 1e-5 * L * L;
    const tau_throttle = L >= p.L_practical ? 2.0 : 0;
    const tau          = 0.1 + g_L + tau_throttle;

    trajectory.push({ k, L, H, S, D, z, V, m, theta_mode, tau, isF: z === "F" ? 1 : 0 });

    if (k === p.T) break;

    // ── State update ──────────────────────────────────────────────────────────

    // Inputs bounded by Assumption 4.1 (U), (Y)
    const U_k   = Math.floor(rng() * p.U_max * 0.7 + p.U_max * 0.3);
    const Y_k   = Math.floor(rng() * p.Y_max_tight * 0.6 + p.Y_max_tight * 0.2);
    const eps_H = p.sigma_H * gaussianRandom(rng);
    // NOTE: eps_S is NOT generated. Stability noise removed in v2.0.  [FIX-1]

    // Context — Eq. (9)
    const L_next = Math.max(0, Math.min(p.L_max, L + U_k + Y_k - r));

    // Entropy — Eq. (14)
    const H_tgt   = h_target(L);
    const Delta_H = L >= p.L_practical ? p.delta_H : 0;
    const H_next  = H - alpha * (H - H_tgt) + Delta_H + eps_H;

    // Stability — Corollary 3.7 (algebraic coupling, NOT a separate SDE) [FIX-1]
    //   Ŝ_{k+1} = 1 − Ĥ_{k+1} / S_obs_max
    const S_next = Math.max(0, Math.min(1, 1 - H_next / p.S_obs_max));

    H_prev = H;
    z_prev = z;
    L = L_next;
    H = H_next;
    S = S_next;
  }

  return trajectory;
}

// ─── Cumulative fragmentation count ──────────────────────────────────────────
function computeNF(traj) {
  let count = 0;
  return traj.map((pt) => {
    if (pt.z === "F") count++;
    return { ...pt, NF: count };
  });
}

// ─── Theoretical N_F bound — Thm. 5.3 / Eq. (24)  [FIX-4] ──────────────────
// N_F(T) ≤ V₀/δ + (σ/δ)·√(2T·ln(1/δ₀))   w.p. ≥ 1 − δ₀
// σ = σ_H only (scalar ξ_{k+1}; stability noise removed in v2.0)  [FIX-2]
function theoreticalBound(V0, delta, sigma, T, delta0) {
  return V0 / delta + (sigma / delta) * Math.sqrt(2 * T * Math.log(1 / delta0));
}

// ─── Phase colours ────────────────────────────────────────────────────────────
const PHASE_COLORS = { C: "#10b981", R: "#f59e0b", F: "#ef4444" };
const PHASE_LABELS = { C: "Coherence", R: "Reorganization", F: "Fragmentation" };

// ─── Custom tooltip ───────────────────────────────────────────────────────────
function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: "rgba(15,23,42,0.95)", border: "1px solid rgba(148,163,184,0.2)",
      borderRadius: 8, padding: "10px 14px", fontSize: 12, color: "#e2e8f0",
      lineHeight: 1.6,
    }}>
      <div style={{ fontWeight: 700, marginBottom: 4, color: "#94a3b8" }}>Step {label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color || "#e2e8f0" }}>
          {p.name}: <span style={{ fontWeight: 600 }}>
            {typeof p.value === "number" ? p.value.toFixed(3) : p.value}
          </span>
        </div>
      ))}
    </div>
  );
}

// ─── Phase strip ──────────────────────────────────────────────────────────────
function PhaseStrip({ data }) {
  const w = 600;
  const h = 14;
  const stepW = w / data.length;
  return (
    <svg width={w} height={h} style={{ marginLeft: 55, marginTop: -2, display: "block" }}>
      {data.map((d, i) => (
        <rect key={i} x={i * stepW} y={0} width={stepW + 0.5} height={h}
          fill={PHASE_COLORS[d.z]} opacity={0.7} />
      ))}
    </svg>
  );
}

// ─── Phase space diagram (Ĥ vs Ŝ) with algebraic manifold  [FIX-6] ───────────
function PhaseDiagram({ controlled, uncontrolled, params }) {
  const p = { ...DEFAULT_PARAMS, ...params };

  // Algebraic manifold points: Ŝ = 1 − Ĥ / S_obs_max  (Corollary 3.7)
  const manifoldPoints = Array.from({ length: 60 }, (_, i) => {
    const H = (i / 59) * p.S_obs_max;
    return { H, S: Math.max(0, 1 - H / p.S_obs_max) };
  });

  return (
    <div style={{ position: "relative" }}>
      <ResponsiveContainer width="100%" height={360}>
        <ScatterChart margin={{ top: 20, right: 30, bottom: 30, left: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" />
          <XAxis dataKey="H" type="number" name="Ĥ" domain={[0, Math.max(2.5, p.S_obs_max * 1.05)]}
            label={{ value: "Ĥ (observer entropy)", position: "insideBottom", offset: -10,
                     fill: "#94a3b8", fontSize: 11 }}
            tick={{ fill: "#64748b", fontSize: 10 }} stroke="rgba(148,163,184,0.2)" />
          <YAxis dataKey="S" type="number" name="Ŝ" domain={[0, 1]}
            label={{ value: "Ŝ (stability)", angle: -90, position: "insideLeft",
                     fill: "#94a3b8", fontSize: 11 }}
            tick={{ fill: "#64748b", fontSize: 10 }} stroke="rgba(148,163,184,0.2)" />
          <ReferenceLine x={p.H_c}       stroke="#ef4444" strokeDasharray="6 3" strokeWidth={1.5} />
          <ReferenceLine y={p.S_c}       stroke="#3b82f6" strokeDasharray="6 3" strokeWidth={1.5} />
          <ReferenceLine x={p.S_obs_max} stroke="#8b5cf6" strokeDasharray="4 4" strokeWidth={1} />
          <Tooltip content={<CustomTooltip />} />
          {/* Algebraic manifold Ŝ = 1 − Ĥ/S_obs_max  [FIX-6] */}
          <Scatter name="Algebraic manifold (Cor. 3.7)" data={manifoldPoints}
            fill="none" line={{ stroke: "#94a3b8", strokeWidth: 1, strokeDasharray: "3 3" }}
            shape={() => null} legendType="line" />
          <Scatter name="Uncontrolled" data={uncontrolled} fill="#ef4444" opacity={0.35} r={3} />
          <Scatter name="CPL-Governor" data={controlled}  fill="#10b981" opacity={0.6}  r={3} />
          <Legend wrapperStyle={{ fontSize: 11, color: "#94a3b8" }} />
        </ScatterChart>
      </ResponsiveContainer>
      <div style={{ position: "absolute", top: 28, left: 70, color: "#10b981",
                    fontSize: 10, fontWeight: 700, opacity: 0.6 }}>COHERENCE (C)</div>
      <div style={{ position: "absolute", top: 28, right: 50, color: "#ef4444",
                    fontSize: 10, fontWeight: 700, opacity: 0.6 }}>FRAGMENTATION (F)</div>
      <div style={{ position: "absolute", top: 60, right: 50, color: "#8b5cf6",
                    fontSize: 9, opacity: 0.5 }}>S_obs_max</div>
    </div>
  );
}

// ─── Main dashboard ───────────────────────────────────────────────────────────
export default function CPLDashboard() {
  const [params, setParams]     = useState(DEFAULT_PARAMS);
  const [activeTab, setActiveTab] = useState("context");
  const [showSettings, setShowSettings] = useState(false);

  const controlled   = useMemo(() => computeNF(simulate(params, true)),  [params]);
  const uncontrolled = useMemo(() => computeNF(simulate(params, false)), [params]);

  // Theoretical N_F bound — Thm. 5.3 / Eq. (24)
  // σ_eff = σ_H  (v2.0: stability noise removed — Lemma 4.3)  [FIX-2]
  const V0        = Math.max(controlled[0]?.V || 0.1, 0.1);
  const delta_drift = 0.05;
  const sigma_eff = params.sigma_H;  // scalar σ_H only  [FIX-2]

  const boundData = useMemo(() => controlled.map((pt) => ({
    k: pt.k,
    bound: theoreticalBound(V0, delta_drift, sigma_eff, Math.max(pt.k, 1), 0.05),
  })), [controlled, V0, sigma_eff]);

  const updateParam = (key, val) =>
    setParams((p) => ({ ...p, [key]: val }));

  const tabs = [
    { id: "context",       label: "L_k Context"   },
    { id: "entropy",       label: "Ĥ_k Entropy"   },
    { id: "stability",     label: "Ŝ_k Stability"  },
    { id: "lyapunov",      label: "V_k Lyapunov"   },
    { id: "fragmentation", label: "N_F(T)"          },
    { id: "phase",         label: "(Ĥ, Ŝ) Phase"   },
  ];

  const chartStyle = {
    background: "rgba(15,23,42,0.6)",
    borderRadius: 12,
    border: "1px solid rgba(148,163,184,0.1)",
    padding: "16px 8px 8px 8px",
    marginBottom: 16,
  };
  const refLineStyle = { strokeDasharray: "8 4", strokeWidth: 1.5 };

  // ── Summary stats (v2.0) ───────────────────────────────────────────────────
  const finalCtrl   = controlled[controlled.length - 1];
  const finalBase   = uncontrolled[uncontrolled.length - 1];
  const summaryStats = [
    { label: "Max L (Baseline)",  value: Math.max(...uncontrolled.map(d => d.L)).toFixed(0), color: "#ef4444" },
    { label: "Max L (Governor)",  value: Math.max(...controlled.map(d => d.L)).toFixed(0),   color: "#10b981" },
    { label: "N_F Baseline",      value: finalBase?.NF  || 0,                                 color: "#ef4444" },
    { label: "N_F Governor",      value: finalCtrl?.NF  || 0,                                 color: "#10b981" },
    { label: "Final V (Baseline)", value: finalBase?.V.toFixed(3) || "–",                     color: "#ef4444" },
    { label: "Final V (Governor)", value: finalCtrl?.V.toFixed(3) || "–",                     color: "#10b981" },
    // v2.0: show final algebraic stability (= 1 − Ĥ/S_obs_max)
    { label: "Final Ŝ (Governor)", value: finalCtrl?.S.toFixed(4) || "–",                     color: "#10b981" },
    { label: "σ_eff (= σ_H)",     value: sigma_eff.toFixed(4),                                color: "#8b5cf6" },
  ];

  return (
    <div style={{
      minHeight: "100vh",
      background: "linear-gradient(145deg, #0a0f1c 0%, #0f172a 40%, #111827 100%)",
      color: "#e2e8f0",
      fontFamily: "'JetBrains Mono', 'SF Mono', 'Fira Code', monospace",
      padding: "20px 24px",
    }}>

      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <div style={{ marginBottom: 20 }}>
        <div style={{ display: "flex", alignItems: "baseline", gap: 12, flexWrap: "wrap" }}>
          <h1 style={{
            fontSize: 22, fontWeight: 800, margin: 0,
            background: "linear-gradient(135deg, #10b981, #3b82f6, #8b5cf6)",
            WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
            letterSpacing: "-0.5px",
          }}>
            CPL Context Governor
          </h1>
          <span style={{ fontSize: 11, color: "#64748b", letterSpacing: 1 }}>
            v2.0 · INFORMATION-GEOMETRIC
          </span>
        </div>
        <p style={{ fontSize: 12, color: "#64748b", margin: "6px 0 0 0", maxWidth: 720 }}>
          Discrete simulation of Eqs. (9), (14) with policy §4 · Baseline vs CPL-Governor ·
          doi:10.5281/zenodo.19177363
        </p>

        {/* v2.0 badge row */}
        <div style={{ display: "flex", gap: 8, marginTop: 8, flexWrap: "wrap" }}>
          {[
            "Ŝ_k = 1 − Ĥ_k / S_obs_max  (Cor. 3.7)",
            "σ_eff = σ_H  (Lem. 4.3)",
            "Assumption 3.14 (Spectral)",
          ].map((txt) => (
            <span key={txt} style={{
              fontSize: 10, color: "#8b5cf6",
              border: "1px solid rgba(139,92,246,0.3)",
              borderRadius: 4, padding: "2px 7px",
              background: "rgba(139,92,246,0.07)",
            }}>{txt}</span>
          ))}
        </div>
      </div>

      {/* ── Controls ───────────────────────────────────────────────────────── */}
      <div style={{ display: "flex", gap: 8, marginBottom: 14, flexWrap: "wrap", alignItems: "center" }}>
        <button onClick={() => setShowSettings(!showSettings)} style={{
          background: showSettings ? "rgba(59,130,246,0.2)" : "rgba(148,163,184,0.08)",
          border: "1px solid rgba(148,163,184,0.2)", borderRadius: 8,
          color: "#e2e8f0", padding: "6px 14px", fontSize: 12, cursor: "pointer",
          fontFamily: "inherit",
        }}>⚙ Parameters</button>
        <button onClick={() => updateParam("seed", params.seed + 1)} style={{
          background: "rgba(148,163,184,0.08)",
          border: "1px solid rgba(148,163,184,0.2)", borderRadius: 8,
          color: "#e2e8f0", padding: "6px 14px", fontSize: 12, cursor: "pointer",
          fontFamily: "inherit",
        }}>↻ Seed ({params.seed})</button>
        <span style={{ fontSize: 11, color: "#64748b", marginLeft: 4 }}>T = {params.T}</span>
      </div>

      {/* ── Parameter sliders (v2.0 — SDE parameters removed) ──────────────── */}
      {showSettings && (
        <div style={{
          ...chartStyle, padding: 16,
          display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(230px, 1fr))", gap: 12,
        }}>
          {/* v2.0 parameters */}
          {[
            { key: "sigma_H",    label: "σ_H  — entropy noise (only, v2.0)",     min: 0,      max: 0.20,   step: 0.005  },
            { key: "alpha_tight",label: "α_tight  — contraction rate (Ass. 3.13)",min: 0.05,   max: 0.80,   step: 0.01   },
            { key: "S_obs_max",  label: "S_obs_max  — sup observer entropy",      min: 0.5,    max: 3.0,    step: 0.05   },
            { key: "Q_ratio",    label: "Q_ratio  — Q(θ_tight)/Q(θ_base) (Ass. 3.14)", min: 0.1, max: 1.0, step: 0.05  },
            { key: "gamma",      label: "γ  — reorg. threshold",                  min: 0.01,   max: 0.50,   step: 0.01   },
            { key: "T",          label: "Horizon T",                              min: 50,     max: 500,    step: 10     },
          ].map(({ key, label, min, max, step }) => (
            <div key={key}>
              <div style={{ fontSize: 11, color: "#94a3b8", marginBottom: 4 }}>
                {label}:{" "}
                <span style={{ color: "#e2e8f0", fontWeight: 600 }}>{params[key]}</span>
              </div>
              <input type="range" min={min} max={max} step={step}
                value={params[key]}
                onChange={(e) => updateParam(key, parseFloat(e.target.value))}
                style={{ width: "100%", accentColor: "#8b5cf6" }} />
            </div>
          ))}
          <div style={{ gridColumn: "1/-1", fontSize: 10, color: "#475569",
                        borderTop: "1px solid rgba(148,163,184,0.08)", paddingTop: 8 }}>
            Removed in v2.0 (SDE stability): rho, b_C, b_R, b_F, kappa_L, kappa_P, sigma_S
          </div>
        </div>
      )}

      {/* ── Tab navigation ─────────────────────────────────────────────────── */}
      <div style={{
        display: "flex", gap: 4, marginBottom: 14, flexWrap: "wrap",
        borderBottom: "1px solid rgba(148,163,184,0.1)", paddingBottom: 8,
      }}>
        {tabs.map((tab) => (
          <button key={tab.id} onClick={() => setActiveTab(tab.id)} style={{
            background: activeTab === tab.id ? "rgba(59,130,246,0.15)" : "transparent",
            border: activeTab === tab.id ? "1px solid rgba(59,130,246,0.3)" : "1px solid transparent",
            borderRadius: 6, color: activeTab === tab.id ? "#60a5fa" : "#64748b",
            padding: "5px 12px", fontSize: 11, cursor: "pointer", fontFamily: "inherit",
            fontWeight: activeTab === tab.id ? 700 : 400, transition: "all 0.15s",
          }}>{tab.label}</button>
        ))}
      </div>

      {/* ── Charts ─────────────────────────────────────────────────────────── */}
      <div>

        {/* (A) Context Length — Prop. 5.1 */}
        {activeTab === "context" && (
          <div style={chartStyle}>
            <div style={{ fontSize: 13, fontWeight: 700, color: "#e2e8f0", marginBottom: 4, paddingLeft: 8 }}>
              Context length L_k — Invariant cap (Prop. 5.1)
            </div>
            <div style={{ fontSize: 11, color: "#64748b", marginBottom: 12, paddingLeft: 8 }}>
              Baseline exceeds L_practical → catastrophic latency. Governor maintains L_k ≤ L_cap for all k.
            </div>
            <ResponsiveContainer width="100%" height={320}>
              <ComposedChart margin={{ top: 5, right: 30, bottom: 5, left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.08)" />
                <XAxis dataKey="k" data={uncontrolled} tick={{ fill: "#64748b", fontSize: 10 }} stroke="rgba(148,163,184,0.15)" />
                <YAxis domain={[0, params.L_max + 200]} tick={{ fill: "#64748b", fontSize: 10 }} stroke="rgba(148,163,184,0.15)" />
                <Tooltip content={<CustomTooltip />} />
                <ReferenceLine y={params.L_warn}      stroke="#f59e0b" {...refLineStyle} label={{ value: "L_warn",      fill: "#f59e0b", fontSize: 10, position: "right" }} />
                <ReferenceLine y={params.L_cap}       stroke="#ef4444" {...refLineStyle} label={{ value: "L_cap",       fill: "#ef4444", fontSize: 10, position: "right" }} />
                <ReferenceLine y={params.L_practical} stroke="#dc2626" strokeWidth={2} strokeDasharray="4 2"
                  label={{ value: "L_practical", fill: "#dc2626", fontSize: 10, position: "right" }} />
                <Line data={uncontrolled} dataKey="L" name="Baseline"     stroke="#ef4444" strokeWidth={2}   dot={false} opacity={0.8} />
                <Line data={controlled}   dataKey="L" name="CPL-Governor" stroke="#10b981" strokeWidth={2}   dot={false} />
                <Legend wrapperStyle={{ fontSize: 11, color: "#94a3b8" }} />
              </ComposedChart>
            </ResponsiveContainer>
            <div style={{ display: "flex", gap: 16, fontSize: 11, color: "#64748b", padding: "4px 8px" }}>
              <span>Phase (Governor):</span>
              {Object.entries(PHASE_LABELS).map(([k, v]) => (
                <span key={k} style={{ color: PHASE_COLORS[k] }}>■ {v}</span>
              ))}
            </div>
            <PhaseStrip data={controlled} />
          </div>
        )}

        {/* (B) Observer Entropy — Prop. 5.2 */}
        {activeTab === "entropy" && (
          <div style={chartStyle}>
            <div style={{ fontSize: 13, fontWeight: 700, color: "#e2e8f0", marginBottom: 4, paddingLeft: 8 }}>
              Observer entropy Ĥ_k = S_obs(p_θ, ε) — Contraction under tight decoding (Prop. 5.2)
            </div>
            <div style={{ fontSize: 11, color: "#64748b", marginBottom: 12, paddingLeft: 8 }}>
              α_tight = {params.alpha_tight} contracts Ĥ_k toward h(L, θ_tight) &lt; H_c.
              S_obs_max = {params.S_obs_max.toFixed(3)} shown as upper reference.
            </div>
            <ResponsiveContainer width="100%" height={320}>
              <ComposedChart margin={{ top: 5, right: 30, bottom: 5, left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.08)" />
                <XAxis dataKey="k" data={uncontrolled} tick={{ fill: "#64748b", fontSize: 10 }} stroke="rgba(148,163,184,0.15)" />
                <YAxis domain={[0, Math.max(3, params.S_obs_max * 1.1)]} tick={{ fill: "#64748b", fontSize: 10 }} stroke="rgba(148,163,184,0.15)" />
                <Tooltip content={<CustomTooltip />} />
                <ReferenceLine y={params.H_c}       stroke="#ef4444" {...refLineStyle} label={{ value: "H_c = ln(3)", fill: "#ef4444", fontSize: 10, position: "right" }} />
                <ReferenceLine y={params.S_obs_max} stroke="#8b5cf6" strokeDasharray="4 4" strokeWidth={1}
                  label={{ value: "S_obs_max",  fill: "#8b5cf6", fontSize: 10, position: "right" }} />
                <Line data={uncontrolled} dataKey="H" name="Baseline"     stroke="#ef4444" strokeWidth={1.5} dot={false} opacity={0.7} />
                <Line data={controlled}   dataKey="H" name="CPL-Governor" stroke="#10b981" strokeWidth={2}   dot={false} />
                <Legend wrapperStyle={{ fontSize: 11, color: "#94a3b8" }} />
              </ComposedChart>
            </ResponsiveContainer>
            <PhaseStrip data={controlled} />
          </div>
        )}

        {/* (C) Stability — algebraic coupling (Corollary 3.7)  [FIX-7] */}
        {activeTab === "stability" && (
          <div style={chartStyle}>
            <div style={{ fontSize: 13, fontWeight: 700, color: "#e2e8f0", marginBottom: 4, paddingLeft: 8 }}>
              Stability Ŝ_k = 1 − Ĥ_k / S_obs_max — Algebraic coupling (Cor. 3.7)
            </div>
            <div style={{ fontSize: 11, color: "#64748b", marginBottom: 12, paddingLeft: 8 }}>
              Stability is fully determined by entropy — no independent SDE. Changes in Ĥ_k
              directly drive Ŝ_k via the algebraic identity (Corollary 3.7, v2.0).
            </div>
            <ResponsiveContainer width="100%" height={320}>
              <ComposedChart margin={{ top: 5, right: 30, bottom: 5, left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.08)" />
                <XAxis dataKey="k" data={uncontrolled} tick={{ fill: "#64748b", fontSize: 10 }} stroke="rgba(148,163,184,0.15)" />
                <YAxis domain={[0, 1]} tick={{ fill: "#64748b", fontSize: 10 }} stroke="rgba(148,163,184,0.15)" />
                <Tooltip content={<CustomTooltip />} />
                <ReferenceLine y={params.S_c} stroke="#3b82f6" {...refLineStyle}
                  label={{ value: "S_c = 0.7", fill: "#3b82f6", fontSize: 10, position: "right" }} />
                <Line data={uncontrolled} dataKey="S" name="Baseline"     stroke="#ef4444" strokeWidth={1.5} dot={false} opacity={0.7} />
                <Line data={controlled}   dataKey="S" name="CPL-Governor" stroke="#10b981" strokeWidth={2}   dot={false} />
                <Legend wrapperStyle={{ fontSize: 11, color: "#94a3b8" }} />
              </ComposedChart>
            </ResponsiveContainer>
            <PhaseStrip data={controlled} />
          </div>
        )}

        {/* (D) Lyapunov potential — Lemma 4.4 */}
        {activeTab === "lyapunov" && (
          <div style={chartStyle}>
            <div style={{ fontSize: 13, fontWeight: 700, color: "#e2e8f0", marginBottom: 4, paddingLeft: 8 }}>
              Lyapunov potential V_k = (Ĥ_k − H_c)₊ + (S_c − Ŝ_k)₊ — Drift condition (Lem. 4.4)
            </div>
            <div style={{ fontSize: 11, color: "#64748b", marginBottom: 12, paddingLeft: 8 }}>
              Governor induces negative drift via entropy contraction (Assumption 3.14).
              Baseline: V_k diverges without return.
            </div>
            <ResponsiveContainer width="100%" height={320}>
              <ComposedChart margin={{ top: 5, right: 30, bottom: 5, left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.08)" />
                <XAxis dataKey="k" data={uncontrolled} tick={{ fill: "#64748b", fontSize: 10 }} stroke="rgba(148,163,184,0.15)" />
                <YAxis tick={{ fill: "#64748b", fontSize: 10 }} stroke="rgba(148,163,184,0.15)" />
                <Tooltip content={<CustomTooltip />} />
                <ReferenceLine y={0} stroke="#10b981" strokeWidth={1} opacity={0.35} />
                <Line data={uncontrolled} dataKey="V" name="Baseline V_k"  stroke="#ef4444" strokeWidth={1.5} dot={false} opacity={0.7} />
                <Line data={controlled}   dataKey="V" name="Governor V_k"  stroke="#10b981" strokeWidth={2}   dot={false} />
                <Legend wrapperStyle={{ fontSize: 11, color: "#94a3b8" }} />
              </ComposedChart>
            </ResponsiveContainer>
            <PhaseStrip data={controlled} />
          </div>
        )}

        {/* (E) Fragmentation occupancy — Thm. 5.3 / Eq. (24)  [FIX-4] */}
        {activeTab === "fragmentation" && (
          <div style={chartStyle}>
            <div style={{ fontSize: 13, fontWeight: 700, color: "#e2e8f0", marginBottom: 4, paddingLeft: 8 }}>
              Fragmentation occupancy N_F(T) vs theoretical bound (Thm. 5.3)
            </div>
            <div style={{ fontSize: 11, color: "#64748b", marginBottom: 12, paddingLeft: 8 }}>
              {/* Updated description: σ_eff = σ_H only (not √(σ_H²+σ_S²))  [FIX-2] */}
              N_F stays below O(√T) bound at δ₀ = 0.05.
              σ_eff = σ_H = {sigma_eff.toFixed(4)} — scalar noise (Lem. 4.3, v2.0).
            </div>
            <ResponsiveContainer width="100%" height={320}>
              <ComposedChart margin={{ top: 5, right: 30, bottom: 5, left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.08)" />
                <XAxis dataKey="k" data={controlled} tick={{ fill: "#64748b", fontSize: 10 }} stroke="rgba(148,163,184,0.15)" />
                <YAxis tick={{ fill: "#64748b", fontSize: 10 }} stroke="rgba(148,163,184,0.15)" />
                <Tooltip content={<CustomTooltip />} />
                <Line data={uncontrolled} dataKey="NF"    name="Baseline N_F"   stroke="#ef4444" strokeWidth={1.5} dot={false} opacity={0.7} />
                <Line data={controlled}   dataKey="NF"    name="Governor N_F"   stroke="#10b981" strokeWidth={2}   dot={false} />
                <Line data={boundData}    dataKey="bound" name="Bound (δ₀=0.05)" stroke="#8b5cf6" strokeWidth={1.5} strokeDasharray="6 3" dot={false} />
                <Legend wrapperStyle={{ fontSize: 11, color: "#94a3b8" }} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* (F) Phase space — with algebraic manifold  [FIX-6] */}
        {activeTab === "phase" && (
          <div style={chartStyle}>
            <div style={{ fontSize: 13, fontWeight: 700, color: "#e2e8f0", marginBottom: 4, paddingLeft: 8 }}>
              Phase space (Ĥ, Ŝ) — algebraic manifold Ŝ = 1 − Ĥ/S_obs_max (Cor. 3.7)
            </div>
            <div style={{ fontSize: 11, color: "#64748b", marginBottom: 12, paddingLeft: 8 }}>
              Both trajectories lie on the manifold. Uncontrolled drifts to F zone;
              Governor stays in C. Dashed gray line = Corollary 3.7 algebraic constraint.
            </div>
            <PhaseDiagram controlled={controlled} uncontrolled={uncontrolled} params={params} />
          </div>
        )}

      </div>

      {/* ── Summary stats (v2.0) ───────────────────────────────────────────── */}
      <div style={{
        display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(170px, 1fr))",
        gap: 8, marginTop: 8,
      }}>
        {summaryStats.map((stat, i) => (
          <div key={i} style={{
            background: "rgba(15,23,42,0.6)", border: "1px solid rgba(148,163,184,0.1)",
            borderRadius: 8, padding: "10px 14px",
          }}>
            <div style={{ fontSize: 9, color: "#64748b", textTransform: "uppercase", letterSpacing: 0.5 }}>
              {stat.label}
            </div>
            <div style={{ fontSize: 19, fontWeight: 800, color: stat.color, marginTop: 2 }}>
              {stat.value}
            </div>
          </div>
        ))}
      </div>

      {/* ── Footer (v2.0) ──────────────────────────────────────────────────── */}
      <div style={{ marginTop: 24, fontSize: 10, color: "#475569", lineHeight: 1.7,
                    borderTop: "1px solid rgba(148,163,184,0.08)", paddingTop: 12 }}>
        <div>CPL 4.0 · Khomyakov · doi:10.5281/zenodo.17788635</div>
        <div>Information-Geometric Context Window Governance v2.0 · doi:10.5281/zenodo.19177363</div>
        <div style={{ marginTop: 4, color: "#374151" }}>
          Simulation implements discrete Eqs. (9), (14) with policy (18)–(19).
          Ŝ_k = 1 − Ĥ_k / S_obs_max (Cor. 3.7). σ_eff = σ_H (Lem. 4.3).
          Seeded PRNG ensures reproducibility. All thresholds calibrable from operational logs.
        </div>
      </div>

    </div>
  );
}
