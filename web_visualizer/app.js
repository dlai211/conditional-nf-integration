/* Conditional NF Visual Debugger — static client for telemetry_data.json. */

"use strict";

const COLORS = {
  ink: "#12151a",
  ink2: "#4d535c",
  muted: "#7b818a",
  line: "#ddd9d0",
  grid: "#e7e3da",
  blue: "#2a78d6",
  blueDark: "#1c5cab",
  orange: "#eb6834",
  aqua: "#1baf7a",
  red: "#c0392b",
};

const PLOT_CONFIG = {
  displayModeBar: false,
  responsive: true,
  scrollZoom: false,
};

const BLUE_SCALE = [
  [0.0, "#f7fbff"],
  [0.18, "#dcebfb"],
  [0.42, "#9ec5f4"],
  [0.68, "#3987e5"],
  [0.86, "#1c5cab"],
  [1.0, "#0d366b"],
];

const STAGE_META = {
  1: {
    label: "step 1 · base space",
    note: "independent uniform base · u ~ U([0,1]²)",
    callout: {
      file: "run_adaptive_integrand_structure_search.py",
      fn: "sample_from_proposal()",
      devices: ["CUDA"],
      note: "the base draw starts on the device",
    },
  },
  2: {
    label: "step 2 · marginal space",
    note: "marginals only · boundary accumulation at x → 0, 1",
    callout: {
      file: "nf/conditional_cubic_1d.py",
      fn: "ConditionalCubicFlow1D.forward()",
      devices: ["CUDA"],
      note: "one conditional spline per dimension",
    },
  },
  3: {
    label: "step 3 · copula space",
    note: "full proposal · the learned 1-edge copula is weak here (|ρ| < 0.04)",
    callout: {
      file: "run_bayesian_selected_copula_benchmark.py",
      fn: "StructuredCopula.sample_u() → marginal.forward()",
      devices: ["CUDA"],
      note: "the copula conditionals reuse the same spline class",
    },
  },
  4: {
    label: "step 4 · kinematic space",
    note: "phase-space map · for 2→2, p_T and η form a one-dimensional curve",
    callout: {
      file: "madgraph_process_integrand.py",
      fn: "TwoBodyPhaseSpace.map() → py_m0_smatrix",
      devices: ["CUDA", "CPU"],
      note: "unit.detach().cpu().numpy() before the Fortran call",
    },
  },
};

const ARCH = {
  driver: {
    file: "experiments/run_madgraph_process_conditional_copula.py",
    fn: "main()",
    devices: ["CPU"],
    note: "orchestration and process configuration",
  },
  engine: {
    file: "experiments/run_adaptive_integrand_structure_search.py",
    fn: "sample_from_proposal() · evaluate_ess()",
    devices: ["CUDA", "CPU"],
    note: "the model runs on CUDA; each chunk returns to CPU for the matrix element",
  },
  marginals: {
    file: "nf/conditional_cubic_1d.py",
    fn: "ConditionalCubicFlow1D.forward() / inverse()",
    devices: ["CUDA"],
    note: "the copula conditionals reuse this same class",
  },
  phasespace: {
    file: "madgraph_process_integrand.py",
    fn: "ConfiguredMadGraphIntegrand.log_integrand()",
    devices: ["CPU"],
    note: "NumPy phase-space map and Jacobian",
  },
  fortran: {
    file: "SubProcesses/P1_gg_ttx/matrix2py.pyd",
    fn: "py_m0_smatrix",
    devices: ["CPU"],
    note: "single-threaded Fortran matrix element",
  },
};

const STAGE_NODE = { 1: "engine", 2: "marginals", 3: "marginals", 4: "phasespace" };

const state = {
  data: null,
  conditions: [],
  betas: [],
  energy: 1000,
  requestedEnergy: 1000,
  betaIndex: 4,
  step: 1,
  density: 2000,
  playing: false,
  archNode: "engine",
  displayed: null,
  animToken: 0,
  playTimer: null,
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

function keyFor(energy) {
  return String(energy);
}

function nearestCondition(value) {
  return state.conditions.reduce(
    (best, candidate) =>
      Math.abs(candidate - value) < Math.abs(best - value) ? candidate : best,
    state.conditions[0]
  );
}

function indicesFor(total, wanted) {
  if (wanted >= total) return Array.from({ length: total }, (_, i) => i);
  const stride = total / wanted;
  return Array.from({ length: wanted }, (_, i) => Math.floor(i * stride));
}

function quantile(sorted, q) {
  if (!sorted.length) return 0;
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

function robustRange(values) {
  const sorted = values.slice().sort((a, b) => a - b);
  const lo = quantile(sorted, 0.005);
  const hi = quantile(sorted, 0.995);
  const span = hi - lo;
  const pad = Math.max(span * 0.08, span === 0 ? 1 : 0.05);
  return [lo - pad, hi + pad];
}

function axisStyle(title) {
  return {
    title: {
      text: title || "",
      font: { size: 10.5, color: COLORS.muted },
      standoff: 10,
    },
    gridcolor: COLORS.grid,
    zeroline: false,
    linecolor: COLORS.line,
    tickfont: { size: 9.5, color: COLORS.ink2 },
    ticks: "outside",
    ticklen: 3,
    tickcolor: COLORS.line,
    automargin: true,
  };
}

function baseLayout(extra = {}) {
  return {
    margin: { l: 48, r: 16, t: 14, b: 42 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: {
      family: "Inter, Segoe UI, system-ui, -apple-system, sans-serif",
      size: 10.5,
      color: COLORS.ink2,
    },
    hoverlabel: {
      bgcolor: "#fffdf8",
      bordercolor: COLORS.line,
      font: { color: COLORS.ink, size: 11 },
    },
    xaxis: axisStyle(),
    yaxis: axisStyle(),
    showlegend: false,
    ...extra,
  };
}

function deviceChips(devices) {
  return devices
    .map(
      (device, index) =>
        `${index ? '<span class="callout-arrow">→</span>' : ""}` +
        `<span class="device device-${device.toLowerCase()}">${device}</span>`
    )
    .join("");
}

function renderCallout(element, callout) {
  if (!element) return;
  element.innerHTML = `
    <div class="callout-row">
      <span class="callout-label">Active code</span>
      <code>${callout.file}::${callout.fn}</code>
    </div>
    <div class="callout-row">
      <span class="callout-label">Device</span>
      ${deviceChips(callout.devices)}
      <span class="callout-note">${callout.note}</span>
    </div>`;
}

/* Data loading ------------------------------------------------------------ */

function loadTelemetryWrapper() {
  return new Promise((resolve, reject) => {
    if (window.TELEMETRY_DATA) {
      resolve(window.TELEMETRY_DATA);
      return;
    }
    const script = document.createElement("script");
    script.src = "telemetry_data.js";
    script.onload = () => {
      if (window.TELEMETRY_DATA) resolve(window.TELEMETRY_DATA);
      else reject(new Error("telemetry_data.js loaded but set no data"));
    };
    script.onerror = () => reject(new Error("telemetry_data.js could not be loaded"));
    document.head.appendChild(script);
  });
}

async function loadTelemetry() {
  // When served over HTTP, the JSON is canonical and the 3.8 MB .js wrapper is
  // not downloaded. Opened from disk, fetch is blocked; load the wrapper then.
  if (window.location.protocol !== "file:") {
    try {
      const response = await fetch("telemetry_data.json", { cache: "no-store" });
      if (response.ok) return response.json();
    } catch (error) {
      // fall through to the wrapper
    }
  }
  return loadTelemetryWrapper();
}

function showFatal(message) {
  const fatal = $("#fatal");
  fatal.hidden = false;
  fatal.textContent = message;
}

/* Panel 1 — architecture -------------------------------------------------- */

function setActiveArchNode(nodeId) {
  $$(".arch-node").forEach((button) => {
    button.classList.toggle("is-active", button.dataset.node === nodeId);
  });
  state.archNode = nodeId;
  const boundary = $("#memory-boundary");
  if (boundary) {
    boundary.classList.toggle(
      "is-active",
      nodeId === "phasespace" || nodeId === "fortran" || state.step === 4
    );
  }
}

function renderArchCallout(nodeId) {
  const base = ARCH[nodeId];
  if (!base) return;
  let note = base.note;
  if (nodeId === "marginals" && state.step === 3) {
    note = "StructuredCopula.sample_u() calls the same ConditionalCubicFlow1D forward()";
  }
  renderCallout($("#arch-callout"), { ...base, note });
}

function initArchitecture() {
  $$(".arch-node").forEach((button) => {
    button.addEventListener("click", () => {
      setActiveArchNode(button.dataset.node);
      renderArchCallout(button.dataset.node);
    });
  });
  setActiveArchNode(STAGE_NODE[state.step]);
  renderArchCallout(STAGE_NODE[state.step]);
}

/* Panel 2 — coordinate space --------------------------------------------- */

function stageCoordinates(step, energy, density) {
  const stage = state.data.stages[keyFor(energy)];
  const indices = indicesFor(stage.u.length, density);
  const pair = (array, axis) => indices.map((index) => array[index][axis]);

  if (step === 1) {
    return {
      step,
      x: pair(stage.u, 0),
      y: pair(stage.u, 1),
      xLabel: "u₁  (base sample)",
      yLabel: "u₂  (base sample)",
      xRange: [0, 1],
      yRange: [0, 1],
      hover: "u₁ = %{x:.3f}<br>u₂ = %{y:.3f}<extra></extra>",
      custom: null,
      contour: false,
    };
  }
  if (step === 2) {
    return {
      step,
      x: pair(stage.x_marg, 0),
      y: pair(stage.x_marg, 1),
      xLabel: "x₁  (marginal)",
      yLabel: "x₂  (marginal)",
      xRange: [0, 1],
      yRange: [0, 1],
      hover: "x₁ = %{x:.3f}<br>x₂ = %{y:.3f}<extra></extra>",
      custom: null,
      contour: density >= 1000,
    };
  }
  if (step === 3) {
    return {
      step,
      x: pair(stage.x_copula, 0),
      y: pair(stage.x_copula, 1),
      xLabel: "x₁  (copula proposal)",
      yLabel: "x₂  (copula proposal)",
      xRange: [0, 1],
      yRange: [0, 1],
      hover: "x₁ = %{x:.3f}<br>x₂ = %{y:.3f}<extra></extra>",
      custom: null,
      contour: density >= 1000,
    };
  }

  const pt = indices.map((index) => stage.kinematic.pt[index]);
  const eta = indices.map((index) => stage.kinematic.eta[index]);
  return {
    step,
    x: pt,
    y: eta,
    xLabel: "p_T  [GeV]",
    yLabel: "η",
    xRange: robustRange(pt),
    yRange: robustRange(eta),
    hover:
      "p_T = %{x:.1f} GeV<br>η = %{y:.2f}" +
      "<br>m_tt = %{customdata:.1f} GeV<extra></extra>",
    custom: indices.map((index) => stage.kinematic.mtt[index]),
    contour: density >= 1000,
  };
}

function densityTrace(coords) {
  return {
    type: "histogram2d",
    x: coords.x,
    y: coords.y,
    nbinsx: 30,
    nbinsy: 30,
    colorscale: [
      [0, "#fbfaf6"],
      [0.28, "#e8f1fc"],
      [0.62, "#86b6ef"],
      [1, "#2a78d6"],
    ],
    opacity: 0.58,
    showscale: false,
    hoverinfo: "skip",
    visible: coords.contour,
  };
}

function scatterTrace(coords, density) {
  const trace = {
    type: "scattergl",
    mode: "markers",
    x: coords.x,
    y: coords.y,
    name: "samples",
    marker: {
      size: density >= 5000 ? 2.3 : density >= 2000 ? 3.0 : 4.2,
      color: COLORS.orange,
      opacity: density >= 5000 ? 0.36 : 0.48,
      line: { width: 0 },
    },
    hovertemplate: coords.hover,
  };
  if (coords.custom) trace.customdata = coords.custom;
  return trace;
}

function spaceLayout(coords, density) {
  const annotations = [
    {
      text: `${STAGE_META[coords.step].note} · N = ${density.toLocaleString("en-US")}`,
      xref: "paper",
      yref: "paper",
      x: 0.012,
      y: 0.985,
      xanchor: "left",
      yanchor: "top",
      showarrow: false,
      font: { size: 9.5, color: COLORS.muted },
      align: "left",
      bgcolor: "rgba(251,250,246,0.82)",
      borderpad: 2,
    },
  ];
  const layout = baseLayout({
    margin: { l: 50, r: 16, t: 14, b: 44 },
    xaxis: { ...axisStyle(coords.xLabel), range: coords.xRange },
    yaxis: { ...axisStyle(coords.yLabel), range: coords.yRange },
    annotations,
    hovermode: "closest",
  });
  return layout;
}

function updateStageIndicators() {
  const meta = STAGE_META[state.step];
  $("#space-stage-label").textContent = meta.label;
  $$("#stage-rail span").forEach((span) => {
    span.classList.toggle("is-active", Number(span.dataset.stage) === state.step);
  });
  $$("#step-buttons button").forEach((button) => {
    button.setAttribute(
      "aria-pressed",
      String(Number(button.dataset.step) === state.step)
    );
  });
  renderCallout($("#space-callout"), meta.callout);
  setActiveArchNode(STAGE_NODE[state.step]);
  renderArchCallout(STAGE_NODE[state.step]);
}

function updateEdgeReadout() {
  const meta = state.data.condition_meta.find((item) => item.energy === state.energy);
  const element = $("#edge-readout");
  if (!element) return;
  if (!meta) {
    element.textContent = "—";
    return;
  }
  element.textContent =
    `marginal 0 edge peak = ${meta.marginal_peaks[0].toFixed(2)} · ` +
    `marginal 1 = ${meta.marginal_peaks[1].toFixed(2)} (smoke checkpoint)`;
}

function renderSpace({ animate = true } = {}) {
  const plot = $("#space-plot");
  updateEdgeReadout();
  const coords = stageCoordinates(state.step, state.energy, state.density);
  const layout = spaceLayout(coords, state.density);
  const contour = densityTrace(coords);
  const scatter = scatterTrace(coords, state.density);
  const previous = state.displayed;
  const token = ++state.animToken;

  const canInterpolate =
    animate &&
    previous &&
    previous.step <= 3 &&
    coords.step <= 3 &&
    previous.x.length === coords.x.length;

  if (canInterpolate) {
    const oldX = previous.x.slice();
    const oldY = previous.y.slice();
    const newX = coords.x;
    const newY = coords.y;
    const start = performance.now();
    const duration = 520;

    const frame = (now) => {
      if (token !== state.animToken) return;
      const t = Math.min(1, (now - start) / duration);
      const eased = t * t * (3 - 2 * t);
      const x = oldX.map((value, index) => value + (newX[index] - value) * eased);
      const y = oldY.map((value, index) => value + (newY[index] - value) * eased);
      Plotly.restyle(plot, { x: [x], y: [y] }, [1]);
      if (t < 1) {
        requestAnimationFrame(frame);
      } else {
        Plotly.restyle(
          plot,
          {
            x: [coords.x],
            y: [coords.y],
            hovertemplate: coords.hover,
          },
          [1]
        );
        if (coords.custom) {
          Plotly.restyle(plot, { customdata: [coords.custom] }, [1]);
        }
        Plotly.restyle(
          plot,
          {
            x: [coords.x],
            y: [coords.y],
            visible: [coords.contour],
          },
          [0]
        );
        Plotly.relayout(plot, { annotations: layout.annotations });
        state.displayed = coords;
      }
    };
    requestAnimationFrame(frame);
    return;
  }

  const fade = animate && previous && previous.step !== coords.step;
  if (fade) {
    plot.classList.add("is-fading");
    window.setTimeout(() => {
      if (token !== state.animToken) return;
      Plotly.react(plot, [contour, scatter], layout, PLOT_CONFIG);
      state.displayed = coords;
      requestAnimationFrame(() => plot.classList.remove("is-fading"));
    }, 150);
  } else {
    Plotly.react(plot, [contour, scatter], layout, PLOT_CONFIG);
    state.displayed = coords;
    plot.classList.remove("is-fading");
  }
}

function setStep(step, animate = true) {
  state.step = Math.min(4, Math.max(1, Number(step)));
  updateStageIndicators();
  renderSpace({ animate });
}

/* Panel 3 — annealing bridge ---------------------------------------------- */

function projectionFrom(z) {
  const rows = z.length;
  const cols = z[0].length;
  const total = new Array(cols).fill(0);
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      total[col] += Math.pow(10, z[row][col]);
    }
  }
  const peak = Math.max(...total, 1e-12);
  return total.map((value) => value / peak);
}

function renderAnneal({ animate = false } = {}) {
  const plot = $("#anneal-plot");
  const projection = $("#anneal-projection");
  const data = state.data.annealing[keyFor(state.energy)];
  const beta = state.betas[state.betaIndex];
  const z = data.log10_density[state.betaIndex];
  const zTarget = data.log10_density[state.betas.length - 1];

  const flat = z.flat();
  const zMin = Math.min(...flat);
  const zMax = Math.max(...flat);
  const span = Math.max(zMax - zMin, 0.5);

  const contour = {
    type: "contour",
    x: data.x0,
    y: data.x1,
    z,
    colorscale: BLUE_SCALE,
    zmin: zMin - span * 0.03,
    zmax: zMax + span * 0.03,
    contours: {
      coloring: "heatmap",
      showlines: true,
      ncontours: 16,
    },
    line: { width: 0.4, color: "rgba(255,255,255,0.26)" },
    colorbar: {
      title: { text: "log₁₀ p", side: "right", font: { size: 9, color: COLORS.muted } },
      thickness: 8,
      len: 0.82,
      tickfont: { size: 8, color: COLORS.ink2 },
      outlinewidth: 0,
    },
    hovertemplate:
      "x₀ = %{x:.3f}<br>x₁ = %{y:.3f}<br>log₁₀ p = %{z:.2f}<extra></extra>",
  };

  const layout = baseLayout({
    margin: { l: 48, r: 58, t: 10, b: 40 },
    xaxis: { ...axisStyle("x₀"), range: [0, 1] },
    yaxis: { ...axisStyle("x₁"), range: [0, 1], scaleanchor: "x", scaleratio: 1 },
    annotations: [
      {
        text: `β = ${beta.toFixed(2)} · √ŝ = ${state.energy} GeV`,
        xref: "paper",
        yref: "paper",
        x: 0.02,
        y: 0.98,
        xanchor: "left",
        yanchor: "top",
        showarrow: false,
        font: { size: 9.5, color: COLORS.muted },
      },
    ],
  });

  const current = projectionFrom(z);
  const target = projectionFrom(zTarget);
  const projectionLayout = baseLayout({
    margin: { l: 42, r: 12, t: 10, b: 40 },
    xaxis: { ...axisStyle("x₀"), range: [0, 1] },
    yaxis: { ...axisStyle("p_β(x₀)"), range: [0, 1.05] },
    showlegend: true,
    legend: {
      orientation: "h",
      x: 0.02,
      y: 0.02,
      xanchor: "left",
      yanchor: "bottom",
      font: { size: 8.3, color: COLORS.ink2 },
      bgcolor: "rgba(251,250,246,0.88)",
      bordercolor: COLORS.line,
      borderwidth: 1,
    },
  });

  const traces = [];
  const atTarget = state.betaIndex === state.betas.length - 1;
  if (!atTarget) {
    traces.push({
      type: "scatter",
      mode: "lines",
      x: data.x0,
      y: target,
      name: "target β = 1.00",
      line: { color: COLORS.muted, width: 1.2, dash: "dash" },
      hovertemplate: "x₀ = %{x:.3f}<br>p = %{y:.3f}<extra>target</extra>",
    });
  }
  traces.push({
    type: "scatter",
    mode: "lines",
    x: data.x0,
    y: current,
    name: `β = ${beta.toFixed(2)}`,
    line: { color: COLORS.blue, width: 2.2 },
    hovertemplate: "x₀ = %{x:.3f}<br>p = %{y:.3f}<extra>current</extra>",
  });

  const apply = () => {
    Plotly.react(plot, [contour], layout, PLOT_CONFIG);
    Plotly.react(projection, traces, projectionLayout, PLOT_CONFIG);
  };

  $("#anneal-label").textContent = `β = ${beta.toFixed(2)}`;
  renderCallout($("#anneal-callout"), {
    file: "experiments/run_adaptive_integrand_structure_search.py",
    fn: "sample_from_proposal() · log_prob() · target_log_integrand()",
    devices: ["CUDA", "CPU"],
    note: "density grid evaluated on the host, matrix element per grid cell",
  });

  if (animate) {
    plot.classList.add("is-fading");
    projection.classList.add("is-fading");
    window.setTimeout(() => {
      apply();
      requestAnimationFrame(() => {
        plot.classList.remove("is-fading");
        projection.classList.remove("is-fading");
      });
    }, 140);
  } else {
    apply();
  }
}

/* Panel 4 — importance sampling ------------------------------------------- */

function statusForPull(pull) {
  const magnitude = Math.abs(pull);
  if (magnitude >= 3) return { className: "status-bad", label: "tension" };
  if (magnitude >= 2) return { className: "status-warn", label: "watch" };
  return { className: "status-good", label: "consistent" };
}

function renderWeights() {
  const plot = $("#weights-plot");
  const weights = state.data.weights[keyFor(state.energy)];
  const values = weights.log10_w;
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const range = robustRange(values);

  const histogram = {
    type: "histogram",
    x: values,
    nbinsx: 44,
    marker: {
      color: COLORS.blue,
      opacity: 0.82,
      line: { width: 0 },
    },
    hovertemplate: "log₁₀ W = %{x:.2f}<br>count = %{y}<extra></extra>",
  };

  const layout = baseLayout({
    margin: { l: 46, r: 14, t: 16, b: 40 },
    xaxis: { ...axisStyle("log₁₀ Wₙ"), range },
    yaxis: { ...axisStyle("count") },
    bargap: 0.035,
    shapes: [
      {
        type: "line",
        x0: mean,
        x1: mean,
        y0: 0,
        y1: 1,
        yref: "paper",
        line: { color: COLORS.orange, width: 1.2, dash: "dash" },
      },
    ],
    annotations: [
      {
        text: `mean = ${mean.toFixed(2)} · N = ${values.length.toLocaleString("en-US")}`,
        xref: "paper",
        yref: "paper",
        x: 0.985,
        y: 0.96,
        xanchor: "right",
        yanchor: "top",
        showarrow: false,
        font: { size: 9.5, color: COLORS.muted },
      },
    ],
  });

  Plotly.react(plot, [histogram], layout, PLOT_CONFIG);

  const ess = weights.ess_sample;
  const validationEss = weights.nf && weights.nf.ess != null ? weights.nf.ess : null;
  $("#ess-value").textContent = ess.toFixed(4);
  $("#ess-note").textContent =
    validationEss == null
      ? "sample-level diagnostic"
      : `sample-level · validation run ESS/N = ${validationEss.toFixed(4)}`;
  const fill = $("#ess-meter-fill");
  fill.style.width = `${Math.max(0, Math.min(1, ess)) * 100}%`;
  fill.style.background = ess >= 0.8 ? COLORS.blue : ess >= 0.5 ? COLORS.orange : COLORS.red;
  $("#ess-meter").setAttribute("aria-valuenow", ess.toFixed(4));

  const nf = weights.nf;
  const sobol = weights.sobol;
  if (nf && sobol) {
    const delta = ((nf.estimate_pb - sobol.estimate_pb) / sobol.estimate_pb) * 100;
    $("#integral-value").textContent =
      `${nf.estimate_pb.toPrecision(6)} / ${sobol.estimate_pb.toPrecision(6)} pb`;
    $("#integral-note").textContent =
      `NF / Sobol · Δ = ${delta >= 0 ? "+" : ""}${delta.toFixed(3)}%` +
      ` · σ_NF = ${nf.stderr_pb.toExponential(2)} pb`;
  } else {
    $("#integral-value").textContent = "unavailable";
    $("#integral-note").textContent = "validation summary not found";
  }

  const pull = weights.pull;
  if (pull != null) {
    const status = statusForPull(pull);
    $("#pull-value").textContent = `${pull >= 0 ? "+" : ""}${pull.toFixed(2)}σ`;
    $("#pull-value").className = `gauge-value ${status.className}`;
    $("#pull-note").innerHTML =
      `<span class="status-pill ${status.className}">${status.label}</span>` +
      ` · |pull| ${Math.abs(pull) < 2 ? "<" : "≥"} 2`;
  } else {
    $("#pull-value").textContent = "—";
    $("#pull-value").className = "gauge-value";
    $("#pull-note").textContent = "reference values unavailable";
  }

  $("#weights-label").textContent = `${state.energy} GeV`;
  renderCallout($("#weights-callout"), {
    file: "experiments/validate_madgraph_process_integral.py",
    fn: "nf_estimate()",
    devices: ["CUDA", "CPU"],
    note: "500,000-sample validation estimate; the histogram uses 2,000 trace samples",
  });
}

/* Controls ---------------------------------------------------------------- */

function bindControls() {
  const energyInput = $("#energy");
  const energyValue = $("#energy-value");
  const energyHint = $("#energy-hint");

  const applyEnergy = (requested, animatePlots = false) => {
    state.requestedEnergy = requested;
    const nearest = nearestCondition(requested);
    const changed = nearest !== state.energy;
    state.energy = nearest;
    energyValue.textContent = `${nearest} GeV`;
    energyHint.textContent =
      requested === nearest
        ? `Computed condition: ${nearest} GeV`
        : `${requested} GeV → nearest computed condition: ${nearest} GeV`;
    if (changed) {
      renderSpace({ animate: animatePlots });
      renderAnneal({ animate: animatePlots });
      renderWeights();
    }
  };

  energyInput.addEventListener("input", () => {
    applyEnergy(Number(energyInput.value), false);
  });
  applyEnergy(Number(energyInput.value), false);

  const betaInput = $("#beta");
  const betaValue = $("#beta-value");
  const applyBeta = () => {
    state.betaIndex = Number(betaInput.value);
    betaValue.textContent = state.betas[state.betaIndex].toFixed(2);
    renderAnneal({ animate: false });
  };
  betaInput.addEventListener("input", applyBeta);
  betaValue.textContent = state.betas[state.betaIndex].toFixed(2);

  $$("#step-buttons button").forEach((button) => {
    button.addEventListener("click", () => setStep(button.dataset.step));
  });
  $$("#stage-rail span").forEach((span) => {
    span.addEventListener("click", () => setStep(span.dataset.stage));
  });

  $$('input[name="density"]').forEach((radio) => {
    radio.addEventListener("change", () => {
      if (!radio.checked) return;
      state.density = Number(radio.value);
      renderSpace({ animate: false });
    });
  });

  const playButton = $("#play");
  const playLabel = $("#play-label");
  const playIcon = playButton.querySelector(".play-icon");

  const stopPlayback = () => {
    state.playing = false;
    if (state.playTimer) window.clearInterval(state.playTimer);
    state.playTimer = null;
    playButton.classList.remove("is-playing");
    playButton.setAttribute("aria-pressed", "false");
    playLabel.textContent = "Play";
    playIcon.textContent = "▶";
  };

  const startPlayback = () => {
    state.playing = true;
    playButton.classList.add("is-playing");
    playButton.setAttribute("aria-pressed", "true");
    playLabel.textContent = "Pause";
    playIcon.textContent = "❚❚";
    if (state.step === 4) {
      state.betaIndex = 0;
      betaInput.value = "0";
      betaValue.textContent = state.betas[0].toFixed(2);
      renderAnneal({ animate: false });
      setStep(1, false);
    }
    state.playTimer = window.setInterval(() => {
      if (state.step < 4) {
        setStep(state.step + 1, true);
      } else {
        state.betaIndex = (state.betaIndex + 1) % state.betas.length;
        betaInput.value = String(state.betaIndex);
        betaValue.textContent = state.betas[state.betaIndex].toFixed(2);
        renderAnneal({ animate: false });
        setStep(1, false);
      }
    }, 1400);
  };

  playButton.addEventListener("click", () => {
    if (state.playing) stopPlayback();
    else startPlayback();
  });

  window.addEventListener("keydown", (event) => {
    if (event.key === " " && event.target === document.body) {
      event.preventDefault();
      if (state.playing) stopPlayback();
      else startPlayback();
    }
  });
}

/* Init -------------------------------------------------------------------- */

function renderMath() {
  if (typeof window.renderMathInElement !== "function") return;
  window.renderMathInElement(document.body, {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "\\(", right: "\\)", display: false },
    ],
    throwOnError: false,
  });
}

function debounce(fn, wait) {
  let timer = null;
  return (...args) => {
    window.clearTimeout(timer);
    timer = window.setTimeout(() => fn(...args), wait);
  };
}

async function init() {
  if (!window.Plotly) {
    showFatal("Plotly failed to load. Check the CDN connection and reload.");
    return;
  }
  try {
    state.data = await loadTelemetry();
  } catch (error) {
    showFatal(
      `Could not load telemetry (${error.message}). Run the dump script, then ` +
        "serve this folder: python -m http.server 8000."
    );
    return;
  }

  state.conditions = state.data.conditions;
  state.betas = state.data.meta.betas;

  // Optional deep link: ?energy=1500&step=3&beta=2&density=5000
  const params = new URLSearchParams(window.location.search);
  const stepParam = params.has("step") ? Number(params.get("step")) : NaN;
  if ([1, 2, 3, 4].includes(stepParam)) state.step = stepParam;
  const betaParam = params.has("beta") ? Number(params.get("beta")) : NaN;
  if (Number.isInteger(betaParam) && betaParam >= 0 && betaParam < state.betas.length) {
    state.betaIndex = betaParam;
  }
  const densityParam = params.has("density") ? Number(params.get("density")) : NaN;
  if ([500, 2000, 5000].includes(densityParam)) state.density = densityParam;
  const requested = params.has("energy") ? Number(params.get("energy")) : NaN;
  state.requestedEnergy =
    Number.isFinite(requested) && requested >= 400 && requested <= 2000
      ? requested
      : 1000;
  state.energy = nearestCondition(state.requestedEnergy);

  $("#energy").value = String(state.requestedEnergy);
  $("#beta").value = String(state.betaIndex);
  $("#beta-value").textContent = state.betas[state.betaIndex].toFixed(2);
  $$('input[name="density"]').forEach((radio) => {
    radio.checked = Number(radio.value) === state.density;
  });

  const deviceChip = $("#device-chip");
  if (deviceChip) {
    const device = state.data.meta.device;
    deviceChip.textContent = `${device.toUpperCase()} model · CPU matrix element`;
  }

  bindControls();
  initArchitecture();
  updateStageIndicators();
  renderSpace({ animate: false });
  renderAnneal({ animate: false });
  renderWeights();
  renderMath();

  window.addEventListener(
    "resize",
    debounce(() => {
      ["space-plot", "anneal-plot", "anneal-projection", "weights-plot"].forEach(
        (id) => {
          const element = document.getElementById(id);
          if (element) Plotly.Plots.resize(element);
        }
      );
    }, 140)
  );
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
