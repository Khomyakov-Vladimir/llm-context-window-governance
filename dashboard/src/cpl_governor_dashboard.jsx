import { useState, useEffect, useCallback, useMemo, useRef } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, ReferenceLine, Area, AreaChart, ScatterChart,
  Scatter, ComposedChart, Bar
} from "recharts";

// ─── Simulation Engine (Eqs. 9, 11, 14 from the paper) ───────────────

const DEFAULT_PARAMS = {
  // Context thresholds (Eq. 15)
  L_recover: 2000,
  L_warn: 3200,
  L_cap: 4000,
  L_practical: 4800,
  L_max: 5000,
  L_safe: 1500,

  // Input bounds (Assumption 3.1)
  U_max: 400,
  Y_max_tight: 350,

  // Entropy dynamics (Eq. 14)
  alpha_base: 0.15,
  alpha_tight: 0.35,
  H_c: 1.099,       // ln(3)
  eta: 0.0003,       // h(L, θ) slope
  delta_H: 0.25,     // regime penalty

  // Stability dynamics (Eq. 11)
  rho: 0.92,
  b_C: 0.06,
  b_R: 0.03,
  b_F: 0.05,
  kappa_L: 0.00004,
  kappa_P: 0.15,
  S_c: 0.7,

  // Phase classifier
  gamma: 0.1,
  beta_hyst: 0.05,

  // Noise
  sigma_H: 0.04,
  sigma_S: 0.025,

  // Rescue releases (Eqs. 16–17)
  r_rescue: 1550,
  r_recover: 1950,

  // Simulation
  T: 200,
  seed: 42,
};

// Seeded PRNG (Mulberry32)
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

function simulate(params, useGovernor) {
  const p = { ...DEFAULT_PARAMS, ...params };
  const rng = mulberry32(p.seed);

  // Design function h (Eq. 12–13): h(L, θ_tight) = H_c - η·(L_cap - L)+
  const h_target = (L) => p.H_c - p.eta * Math.max(p.L_cap - L, 0);

  // Initial state
  let L = 800;
  let H = 0.6;
  let S = 0.85;
  let H_prev = H;
  let z_prev = "C";

  const trajectory = [];

  for (let k = 0; k <= p.T; k++) {
    const D = Math.abs(H - H_prev);

    // Phase classifier (Eq. 8) — "Reorganization first"
    const H_c_eff = p.H_c + (z_prev === "C" ? p.beta_hyst : 0);
    let z;
    if (D >= p.gamma) {
      z = "R";
    } else if (H < H_c_eff && S > p.S_c) {
      z = "C";
    } else {
      z = "F";
    }

    // Lyapunov potential (Eq. 23)
    const V = Math.max(H - p.H_c, 0) + Math.max(p.S_c - S, 0);

    // Policy (Eq. 18–19)
    let m, r, theta_mode;
    if (useGovernor) {
      if (z === "F") {
        m = "chunk"; r = p.r_recover;
      } else if (z === "R") {
        m = "summarize"; r = p.r_rescue;
      } else if (z === "C" && L > p.L_warn) {
        m = "summarize"; r = p.r_rescue;
      } else {
        m = "keep"; r = 0;
      }
      theta_mode = (z === "R" || z === "F" || L > p.L_warn) ? "tight" : "base";
    } else {
      m = "keep"; r = 0;
      theta_mode = "base";
    }

    const alpha = theta_mode === "tight" ? p.alpha_tight : p.alpha_base;

    // Latency (Eq. 10)
    const g_L = 0.00001 * L * L;
    const tau_throttle = L >= p.L_practical ? 2.0 : 0;
    const tau = 0.1 + g_L + tau_throttle;

    trajectory.push({
      k, L, H, S, D, z, V, m, theta_mode, tau,
      isF: z === "F" ? 1 : 0,
    });

    if (k === p.T) break;

    // --- State update ---
    const U_k = Math.floor(rng() * p.U_max * 0.7 + p.U_max * 0.3);
    const Y_k = Math.floor(rng() * p.Y_max_tight * 0.6 + p.Y_max_tight * 0.2);
    const eps_H = p.sigma_H * gaussianRandom(rng);
    const eps_S = p.sigma_S * gaussianRandom(rng);

    // Context update (Eq. 9)
    const L_next_raw = L + U_k + Y_k - r;
    const L_next = Math.max(0, Math.min(p.L_max, L_next_raw));

    // Entropy update (Eq. 14)
    const H_tgt = h_target(L);
    const Delta_H = L >= p.L_practical ? p.delta_H : 0;
    const H_next = H - alpha * (H - H_tgt) + Delta_H + eps_H;

    // Stability update (Eq. 11)
    const b_z = z === "C" ? p.b_C : (z === "R" ? -p.b_R : -p.b_F);
    const S_next_raw = p.rho * S + b_z
      - p.kappa_L * Math.max(L - p.L_safe, 0)
      - p.kappa_P * (L >= p.L_practical ? 1 : 0)
      + eps_S;
    const S_next = Math.max(0, Math.min(1, S_next_raw));

    H_prev = H;
    z_prev = z;
    L = L_next;
    H = H_next;
    S = S_next;
  }

  return trajectory;
}

// ─── Compute cumulative fragmentation count ────────────────────────
function computeNF(traj) {
  let count = 0;
  return traj.map((pt) => {
    if (pt.z === "F") count++;
    return { ...pt, NF: count };
  });
}

// ─── Theoretical bound (Eq. 24) ────────────────────────────────────
function theoreticalBound(V0, delta, sigma, T, delta0) {
  return V0 / delta + (sigma / delta) * Math.sqrt(2 * T * Math.log(1 / delta0));
}

// ─── Phase colors ──────────────────────────────────────────────────
const PHASE_COLORS = { C: "#10b981", R: "#f59e0b", F: "#ef4444" };
const PHASE_LABELS = { C: "Coherence", R: "Reorganization", F: "Fragmentation" };

// ─── Custom Tooltip ────────────────────────────────────────────────
function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: "rgba(15,23,42,0.95)", border: "1px solid rgba(148,163,184,0.2)",
      borderRadius: 8, padding: "10px 14px", fontSize: 12, color: "#e2e8f0",
      backdropFilter: "blur(8px)", lineHeight: 1.6,
    }}>
      <div style={{ fontWeight: 700, marginBottom: 4, color: "#94a3b8" }}>Step {label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color || "#e2e8f0" }}>
          {p.name}: <span style={{ fontWeight: 600 }}>{typeof p.value === "number" ? p.value.toFixed(3) : p.value}</span>
        </div>
      ))}
    </div>
  );
}

// ─── Phase Strip (color bar under charts) ──────────────────────────
function PhaseStrip({ data, width }) {
  const w = width || 600;
  const h = 16;
  const stepW = w / data.length;
  return (
    <svg width={w} height={h} style={{ marginLeft: 55, marginTop: -4 }}>
      {data.map((d, i) => (
        <rect key={i} x={i * stepW} y={0} width={stepW + 0.5} height={h}
          fill={PHASE_COLORS[d.z]} opacity={0.7} />
      ))}
    </svg>
  );
}

// ─── Phase Diagram (H vs S scatter) ────────────────────────────────
function PhaseDiagram({ controlled, uncontrolled, params }) {
  const p = { ...DEFAULT_PARAMS, ...params };
  return (
    <div style={{ position: "relative" }}>
      <ResponsiveContainer width="100%" height={340}>
        <ScatterChart margin={{ top: 20, right: 30, bottom: 20, left: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.12)" />
          <XAxis dataKey="H" type="number" name="Ĥ" domain={[0, 2.5]}
            label={{ value: "Ĥ (entropy)", position: "bottom", offset: 0, fill: "#94a3b8", fontSize: 12 }}
            tick={{ fill: "#64748b", fontSize: 11 }} stroke="rgba(148,163,184,0.2)" />
          <YAxis dataKey="S" type="number" name="Ŝ" domain={[0, 1]}
            label={{ value: "Ŝ (stability)", angle: -90, position: "insideLeft", fill: "#94a3b8", fontSize: 12 }}
            tick={{ fill: "#64748b", fontSize: 11 }} stroke="rgba(148,163,184,0.2)" />
          <ReferenceLine x={p.H_c} stroke="#ef4444" strokeDasharray="6 3" strokeWidth={1.5} />
          <ReferenceLine y={p.S_c} stroke="#3b82f6" strokeDasharray="6 3" strokeWidth={1.5} />
          <Tooltip content={<CustomTooltip />} />
          <Scatter name="Uncontrolled" data={uncontrolled} fill="#ef4444" opacity={0.35} r={3} />
          <Scatter name="CPL-Governor" data={controlled} fill="#10b981" opacity={0.6} r={3} />
          <Legend wrapperStyle={{ fontSize: 12, color: "#94a3b8" }} />
        </ScatterChart>
      </ResponsiveContainer>
      {/* Zone labels */}
      <div style={{ position: "absolute", top: 30, left: 70, color: "#10b981", fontSize: 11, fontWeight: 700, opacity: 0.6 }}>
        COHERENCE (C)
      </div>
      <div style={{ position: "absolute", top: 30, right: 50, color: "#ef4444", fontSize: 11, fontWeight: 700, opacity: 0.6 }}>
        FRAGMENTATION (F)
      </div>
    </div>
  );
}

// ─── Main Dashboard ────────────────────────────────────────────────
export default function CPLDashboard() {
  const [params, setParams] = useState(DEFAULT_PARAMS);
  const [activeTab, setActiveTab] = useState("context");
  const [showSettings, setShowSettings] = useState(false);

  const controlled = useMemo(() => computeNF(simulate(params, true)), [params]);
  const uncontrolled = useMemo(() => computeNF(simulate(params, false)), [params]);

  // Theoretical NF bound
  // σ_eff = √(σ_H² + σ_S²) — composite sub-Gaussian parameter of ξ_{k+1}
  // per Lemma 5.3: ξ = −ε^H·𝟙{Ĥ>Hc} + ε^S·𝟙{Ŝ<Sc}, independent components
  const V0 = uncontrolled[0]?.V || 0.1;
  const delta_drift = 0.05;
  const sigma_eff = Math.sqrt(params.sigma_H ** 2 + params.sigma_S ** 2);
  const boundData = useMemo(() => {
    return controlled.map((pt) => ({
      k: pt.k,
      bound: theoreticalBound(
        Math.max(V0, 0.1), delta_drift, sigma_eff, Math.max(pt.k, 1), 0.05
      ),
    }));
  }, [controlled, V0, sigma_eff]);

  const updateParam = (key, val) => {
    setParams((p) => ({ ...p, [key]: val }));
  };

  const tabs = [
    { id: "context", label: "L_k Context" },
    { id: "entropy", label: "Ĥ_k Entropy" },
    { id: "stability", label: "Ŝ_k Stability" },
    { id: "lyapunov", label: "V_k Lyapunov" },
    { id: "fragmentation", label: "N_F(T)" },
    { id: "phase", label: "(Ĥ, Ŝ) Phase" },
  ];

  const chartStyle = {
    background: "rgba(15,23,42,0.6)",
    borderRadius: 12,
    border: "1px solid rgba(148,163,184,0.1)",
    padding: "16px 8px 8px 8px",
    marginBottom: 16,
  };

  const refLineStyle = { strokeDasharray: "8 4", strokeWidth: 1.5 };

  return (
    <div style={{
      minHeight: "100vh",
      background: "linear-gradient(145deg, #0a0f1c 0%, #0f172a 40%, #111827 100%)",
      color: "#e2e8f0",
      fontFamily: "'JetBrains Mono', 'SF Mono', 'Fira Code', monospace",
      padding: "20px 24px",
    }}>
      {/* Header */}
      <div style={{ marginBottom: 24 }}>
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
            PHASE-AWARE CONTEXT WINDOW GOVERNANCE
          </span>
        </div>
        <p style={{ fontSize: 12, color: "#64748b", margin: "6px 0 0 0", maxWidth: 700 }}>
          Discrete simulation of Eqs. (9), (11), (14) with policy §4.
          Baseline (uncontrolled) vs CPL-Governor. All thresholds from the paper.
        </p>
      </div>

      {/* Controls row */}
      <div style={{ display: "flex", gap: 8, marginBottom: 16, flexWrap: "wrap", alignItems: "center" }}>
        <button
          onClick={() => setShowSettings(!showSettings)}
          style={{
            background: showSettings ? "rgba(59,130,246,0.2)" : "rgba(148,163,184,0.08)",
            border: "1px solid rgba(148,163,184,0.2)", borderRadius: 8,
            color: "#e2e8f0", padding: "6px 14px", fontSize: 12, cursor: "pointer",
            fontFamily: "inherit",
          }}
        >
          ⚙ Parameters
        </button>
        <button
          onClick={() => updateParam("seed", params.seed + 1)}
          style={{
            background: "rgba(148,163,184,0.08)",
            border: "1px solid rgba(148,163,184,0.2)", borderRadius: 8,
            color: "#e2e8f0", padding: "6px 14px", fontSize: 12, cursor: "pointer",
            fontFamily: "inherit",
          }}
        >
          ↻ New Seed ({params.seed})
        </button>
        <div style={{ fontSize: 11, color: "#64748b", marginLeft: 8 }}>
          T = {params.T} steps
        </div>
      </div>

      {/* Parameter sliders */}
      {showSettings && (
        <div style={{
          ...chartStyle, padding: 16,
          display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))", gap: 12,
        }}>
          {[
            { key: "sigma_H", label: "σ_H (entropy noise)", min: 0, max: 0.2, step: 0.005 },
            { key: "sigma_S", label: "σ_S (stability noise)", min: 0, max: 0.15, step: 0.005 },
            { key: "rho", label: "ρ (stability decay)", min: 0.5, max: 0.99, step: 0.01 },
            { key: "alpha_tight", label: "α_tight", min: 0.05, max: 0.8, step: 0.01 },
            { key: "b_F", label: "b_F (frag. penalty)", min: 0, max: 0.2, step: 0.005 },
            { key: "kappa_L", label: "κ_L (length penalty)", min: 0, max: 0.0002, step: 0.000005 },
            { key: "gamma", label: "γ (reorg. threshold)", min: 0.01, max: 0.5, step: 0.01 },
            { key: "T", label: "Horizon T", min: 50, max: 500, step: 10 },
          ].map(({ key, label, min, max, step }) => (
            <div key={key}>
              <div style={{ fontSize: 11, color: "#94a3b8", marginBottom: 4 }}>
                {label}: <span style={{ color: "#e2e8f0", fontWeight: 600 }}>{params[key]}</span>
              </div>
              <input type="range" min={min} max={max} step={step} value={params[key]}
                onChange={(e) => updateParam(key, parseFloat(e.target.value))}
                style={{ width: "100%", accentColor: "#3b82f6" }} />
            </div>
          ))}
        </div>
      )}

      {/* Tab navigation */}
      <div style={{
        display: "flex", gap: 4, marginBottom: 16, flexWrap: "wrap",
        borderBottom: "1px solid rgba(148,163,184,0.1)", paddingBottom: 8,
      }}>
        {tabs.map((tab) => (
          <button key={tab.id} onClick={() => setActiveTab(tab.id)}
            style={{
              background: activeTab === tab.id ? "rgba(59,130,246,0.15)" : "transparent",
              border: activeTab === tab.id ? "1px solid rgba(59,130,246,0.3)" : "1px solid transparent",
              borderRadius: 6, color: activeTab === tab.id ? "#60a5fa" : "#64748b",
              padding: "5px 12px", fontSize: 11, cursor: "pointer", fontFamily: "inherit",
              fontWeight: activeTab === tab.id ? 700 : 400, transition: "all 0.15s",
            }}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Charts */}
      <div>
        {/* (A) Context Length */}
        {activeTab === "context" && (
          <div style={chartStyle}>
            <div style={{ fontSize: 13, fontWeight: 700, color: "#e2e8f0", marginBottom: 4, paddingLeft: 8 }}>
              Context Length L_k — Invariant Cap Demonstration (Prop. 5.1)
            </div>
            <div style={{ fontSize: 11, color: "#64748b", marginBottom: 12, paddingLeft: 8 }}>
              Baseline explodes past L_practical → catastrophic latency. Governor keeps L_k ≤ L_cap.
            </div>
            <ResponsiveContainer width="100%" height={320}>
              <ComposedChart margin={{ top: 5, right: 30, bottom: 5, left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.08)" />
                <XAxis dataKey="k" data={uncontrolled} tick={{ fill: "#64748b", fontSize: 10 }} stroke="rgba(148,163,184,0.15)" />
                <YAxis domain={[0, params.L_max + 200]} tick={{ fill: "#64748b", fontSize: 10 }} stroke="rgba(148,163,184,0.15)" />
                <Tooltip content={<CustomTooltip />} />
                <ReferenceLine y={params.L_warn} stroke="#f59e0b" {...refLineStyle} label={{ value: "L_warn", fill: "#f59e0b", fontSize: 10, position: "right" }} />
                <ReferenceLine y={params.L_cap} stroke="#ef4444" {...refLineStyle} label={{ value: "L_cap", fill: "#ef4444", fontSize: 10, position: "right" }} />
                <ReferenceLine y={params.L_practical} stroke="#dc2626" strokeWidth={2} strokeDasharray="4 2"
                  label={{ value: "L_practical", fill: "#dc2626", fontSize: 10, position: "right" }} />
                <Line data={uncontrolled} dataKey="L" name="Baseline" stroke="#ef4444" strokeWidth={2}
                  dot={false} opacity={0.8} />
                <Line data={controlled} dataKey="L" name="CPL-Governor" stroke="#10b981" strokeWidth={2}
                  dot={false} />
                <Legend wrapperStyle={{ fontSize: 11, color: "#94a3b8" }} />
              </ComposedChart>
            </ResponsiveContainer>
            <div style={{ display: "flex", gap: 16, fontSize: 11, color: "#64748b", padding: "4px 8px" }}>
              <span>Phase strip (Governor):</span>
              {Object.entries(PHASE_LABELS).map(([k, v]) => (
                <span key={k} style={{ color: PHASE_COLORS[k] }}>■ {v}</span>
              ))}
            </div>
            <PhaseStrip data={controlled} />
          </div>
        )}

        {/* (B) Entropy */}
        {activeTab === "entropy" && (
          <div style={chartStyle}>
            <div style={{ fontSize: 13, fontWeight: 700, color: "#e2e8f0", marginBottom: 4, paddingLeft: 8 }}>
              Behavioral Entropy Ĥ_k — Contraction under Tight Decoding (Prop. 5.2)
            </div>
            <div style={{ fontSize: 11, color: "#64748b", marginBottom: 12, paddingLeft: 8 }}>
              Governor's tight regime (α_tight = {params.alpha_tight}) contracts entropy toward H_tgt &lt; H_c.
            </div>
            <ResponsiveContainer width="100%" height={320}>
              <ComposedChart margin={{ top: 5, right: 30, bottom: 5, left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.08)" />
                <XAxis dataKey="k" data={uncontrolled} tick={{ fill: "#64748b", fontSize: 10 }} stroke="rgba(148,163,184,0.15)" />
                <YAxis domain={[0, 3]} tick={{ fill: "#64748b", fontSize: 10 }} stroke="rgba(148,163,184,0.15)" />
                <Tooltip content={<CustomTooltip />} />
                <ReferenceLine y={params.H_c} stroke="#ef4444" {...refLineStyle}
                  label={{ value: "H_c = ln(3)", fill: "#ef4444", fontSize: 10, position: "right" }} />
                <Line data={uncontrolled} dataKey="H" name="Baseline" stroke="#ef4444" strokeWidth={1.5} dot={false} opacity={0.7} />
                <Line data={controlled} dataKey="H" name="CPL-Governor" stroke="#10b981" strokeWidth={2} dot={false} />
                <Legend wrapperStyle={{ fontSize: 11, color: "#94a3b8" }} />
              </ComposedChart>
            </ResponsiveContainer>
            <PhaseStrip data={controlled} />
          </div>
        )}

        {/* (C) Stability */}
        {activeTab === "stability" && (
          <div style={chartStyle}>
            <div style={{ fontSize: 13, fontWeight: 700, color: "#e2e8f0", marginBottom: 4, paddingLeft: 8 }}>
              Semantic Stability Ŝ_k — Degradation vs Recovery
            </div>
            <ResponsiveContainer width="100%" height={320}>
              <ComposedChart margin={{ top: 5, right: 30, bottom: 5, left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.08)" />
                <XAxis dataKey="k" data={uncontrolled} tick={{ fill: "#64748b", fontSize: 10 }} stroke="rgba(148,163,184,0.15)" />
                <YAxis domain={[0, 1]} tick={{ fill: "#64748b", fontSize: 10 }} stroke="rgba(148,163,184,0.15)" />
                <Tooltip content={<CustomTooltip />} />
                <ReferenceLine y={params.S_c} stroke="#3b82f6" {...refLineStyle}
                  label={{ value: "S_c = 0.7", fill: "#3b82f6", fontSize: 10, position: "right" }} />
                <Line data={uncontrolled} dataKey="S" name="Baseline" stroke="#ef4444" strokeWidth={1.5} dot={false} opacity={0.7} />
                <Line data={controlled} dataKey="S" name="CPL-Governor" stroke="#10b981" strokeWidth={2} dot={false} />
                <Legend wrapperStyle={{ fontSize: 11, color: "#94a3b8" }} />
              </ComposedChart>
            </ResponsiveContainer>
            <PhaseStrip data={controlled} />
          </div>
        )}

        {/* (D) Lyapunov Potential */}
        {activeTab === "lyapunov" && (
          <div style={chartStyle}>
            <div style={{ fontSize: 13, fontWeight: 700, color: "#e2e8f0", marginBottom: 4, paddingLeft: 8 }}>
              Lyapunov Potential V_k = (Ĥ_k − H_c)₊ + (S_c − Ŝ_k)₊ — Drift Visualization
            </div>
            <div style={{ fontSize: 11, color: "#64748b", marginBottom: 12, paddingLeft: 8 }}>
              Governor induces negative drift (Assumption D). Baseline: random walk without return.
            </div>
            <ResponsiveContainer width="100%" height={320}>
              <ComposedChart margin={{ top: 5, right: 30, bottom: 5, left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.08)" />
                <XAxis dataKey="k" data={uncontrolled} tick={{ fill: "#64748b", fontSize: 10 }} stroke="rgba(148,163,184,0.15)" />
                <YAxis tick={{ fill: "#64748b", fontSize: 10 }} stroke="rgba(148,163,184,0.15)" />
                <Tooltip content={<CustomTooltip />} />
                <ReferenceLine y={0} stroke="#10b981" strokeWidth={1} opacity={0.4} />
                <Line data={uncontrolled} dataKey="V" name="Baseline V_k" stroke="#ef4444" strokeWidth={1.5} dot={false} opacity={0.7} />
                <Line data={controlled} dataKey="V" name="Governor V_k" stroke="#10b981" strokeWidth={2} dot={false} />
                <Legend wrapperStyle={{ fontSize: 11, color: "#94a3b8" }} />
              </ComposedChart>
            </ResponsiveContainer>
            <PhaseStrip data={controlled} />
          </div>
        )}

        {/* (E) Fragmentation Occupancy N_F(T) */}
        {activeTab === "fragmentation" && (
          <div style={chartStyle}>
            <div style={{ fontSize: 13, fontWeight: 700, color: "#e2e8f0", marginBottom: 4, paddingLeft: 8 }}>
              Fragmentation Occupancy N_F(T) vs Theoretical Bound (Thm. 5.4)
            </div>
            <div style={{ fontSize: 11, color: "#64748b", marginBottom: 12, paddingLeft: 8 }}>
              Empirical N_F stays below the √T bound at δ₀ = 0.05. σ_eff = √(σ_H² + σ_S²) = {sigma_eff.toFixed(4)} (Lemma 5.3).
            </div>
            <ResponsiveContainer width="100%" height={320}>
              <ComposedChart margin={{ top: 5, right: 30, bottom: 5, left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.08)" />
                <XAxis dataKey="k" data={controlled} tick={{ fill: "#64748b", fontSize: 10 }} stroke="rgba(148,163,184,0.15)" />
                <YAxis tick={{ fill: "#64748b", fontSize: 10 }} stroke="rgba(148,163,184,0.15)" />
                <Tooltip content={<CustomTooltip />} />
                <Line data={uncontrolled} dataKey="NF" name="Baseline N_F" stroke="#ef4444" strokeWidth={1.5} dot={false} opacity={0.7} />
                <Line data={controlled} dataKey="NF" name="Governor N_F" stroke="#10b981" strokeWidth={2} dot={false} />
                <Line data={boundData} dataKey="bound" name="Bound (δ₀=0.05)" stroke="#8b5cf6" strokeWidth={1.5}
                  strokeDasharray="6 3" dot={false} />
                <Legend wrapperStyle={{ fontSize: 11, color: "#94a3b8" }} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* (F) Phase Diagram */}
        {activeTab === "phase" && (
          <div style={chartStyle}>
            <div style={{ fontSize: 13, fontWeight: 700, color: "#e2e8f0", marginBottom: 4, paddingLeft: 8 }}>
              Phase Space (Ĥ, Ŝ) — Trajectory Comparison
            </div>
            <div style={{ fontSize: 11, color: "#64748b", marginBottom: 12, paddingLeft: 8 }}>
              H_c and S_c divide the plane. Uncontrolled drifts into F zone; Governor returns to C.
            </div>
            <PhaseDiagram controlled={controlled} uncontrolled={uncontrolled} params={params} />
          </div>
        )}
      </div>

      {/* Summary stats */}
      <div style={{
        display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
        gap: 8, marginTop: 8,
      }}>
        {[
          {
            label: "Max L (Baseline)",
            value: Math.max(...uncontrolled.map((d) => d.L)).toFixed(0),
            color: "#ef4444",
          },
          {
            label: "Max L (Governor)",
            value: Math.max(...controlled.map((d) => d.L)).toFixed(0),
            color: "#10b981",
          },
          {
            label: "N_F Baseline",
            value: uncontrolled[uncontrolled.length - 1]?.NF || 0,
            color: "#ef4444",
          },
          {
            label: "N_F Governor",
            value: controlled[controlled.length - 1]?.NF || 0,
            color: "#10b981",
          },
          {
            label: "Final V (Baseline)",
            value: uncontrolled[uncontrolled.length - 1]?.V.toFixed(3) || "–",
            color: "#ef4444",
          },
          {
            label: "Final V (Governor)",
            value: controlled[controlled.length - 1]?.V.toFixed(3) || "–",
            color: "#10b981",
          },
        ].map((stat, i) => (
          <div key={i} style={{
            background: "rgba(15,23,42,0.6)", border: "1px solid rgba(148,163,184,0.1)",
            borderRadius: 8, padding: "10px 14px",
          }}>
            <div style={{ fontSize: 10, color: "#64748b", textTransform: "uppercase", letterSpacing: 0.5 }}>
              {stat.label}
            </div>
            <div style={{ fontSize: 20, fontWeight: 800, color: stat.color, marginTop: 2 }}>
              {stat.value}
            </div>
          </div>
        ))}
      </div>

      {/* Footer */}
      <div style={{ marginTop: 24, fontSize: 10, color: "#475569", lineHeight: 1.6, borderTop: "1px solid rgba(148,163,184,0.08)", paddingTop: 12 }}>
        <div>CPL 4.0 · Khomyakov (2025) · doi:10.5281/zenodo.17788635</div>
        <div>Phase-Aware Context Window Governance · doi:10.5281/zenodo.18784361</div>
        <div style={{ marginTop: 4 }}>
          Simulation implements discrete Eqs. (9), (11), (14) with policy (18)–(19).
          Seeded PRNG ensures reproducibility. All thresholds calibrable.
        </div>
      </div>
    </div>
  );
}
