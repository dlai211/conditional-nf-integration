# Specification: Conditional NF Visual Debugger Web App

## 1. Goal
Provide an interactive, browser-based debugger to visualize how data points transform through the conditional normalizing flow architecture and inspect where MadGraph evaluates physical matrix elements. The interface must dynamically connect active source code locations, exact mathematical transformations, and slider-controlled visual progression.

---

## 2. Interactive Controls
The application layout must provide accessible, synchronized global controls:
- **Collision Energy Slider ($\sqrt{\hat{s}}$):** Range $[400\text{ GeV}, 2000\text{ GeV}]$ with step size $50\text{ GeV}$ (default: $1000\text{ GeV}$).
- **Annealing Stage Slider ($\beta$):** Discrete checkpoints $[0.00, 0.08, 0.25, 0.55, 1.00]$.
- **Transformation Step Scrubber:** Discrete steps $[1, 2, 3, 4]$ representing the sampling pipeline.
- **Sample Density Selector:** Radios for $500$, $2000$, or $5000$ points to balance rendering frame rate with distribution resolution.
- **Animation Toggle:** Play/Pause controls to automatically step through parameter values.

---

## 3. Four Core Visual Panels

### Panel 1: Macro Execution & Data Flow Architecture
- **Interactive Architecture Graph:** Visual flowchart tracing the calling hierarchy:
  1. Driver: `experiments/run_madgraph_process_conditional_copula.py`
  2. Search Engine: `run_adaptive_integrand_structure_search.py`
  3. 1D Marginals: `nf/conditional_cubic_1d.py`
  4. Phase-Space Mapping: `madgraph_process_integrand.py`
  5. Fortran Subprocess: `SubProcesses/P1_gg_ttx/matrix2py.pyd` via `py_m0_smatrix`
- **Memory Boundary Annotation:** Explicitly highlight the host $\leftrightarrow$ device transfer bottleneck (`unit.detach().cpu().numpy()`) where tensors move from GPU VRAM to CPU RAM for single-threaded Fortran evaluation.

### Panel 2: Coordinate Space Evolution (2D Interactive Scatter / Density)
- **Step 1 (Base Space):** Standard uniform distribution $u \sim U([0, 1]^2)$.
- **Step 2 (Marginal Space):** Points mapped through conditional cubic-Hermite splines $x_{\text{marg}}$, showing boundary accumulation near $0$ and $1$.
- **Step 3 (Copula Space):** Points after applying autoregressive dependencies $q_\theta(x \mid \sqrt{\hat{s}})$, showing diagonal correlations.
- **Step 4 (Kinematic Space):** Points mapped to physical collider observables ($p_T$, $\eta$, $m_{t\bar{t}}$).

### Panel 3: Annealing Bridge & Target Peak Formation
- **Dynamic 2D Contour / 1D Projection:** Display the morphing density slice $p_{\beta_t}(x \mid c)$.
- **Evolutionary Steps:**
  - $\beta = 0.00$: Broad, flat uniform distribution across the hypercube.
  - $\beta \in [0.08, 0.25]$: Formation of early topological ridges along resonance domains.
  - $\beta \to 1.00$: Localized, sharp Breit-Wigner peaks corresponding to physical phase space.

### Panel 4: Importance Sampling Diagnostics & Convergence
- **Weight Distribution Histogram:** Live histogram of importance weights $W_n = \frac{f(X_n \mid c)}{r_\theta(X_n \mid c)}$.
- **Real-Time Statistical Gauges:**
  - Normalized Effective Sample Size ($\text{ESS} / N$).
  - Estimated Integral ($\hat{I}_{\text{NF}}$) vs. Scrambled Sobol Baseline ($\hat{I}_{\text{Sobol}}$).
  - Pull Metric: $\text{Pull} = \frac{\hat{I}_{\text{NF}} - \hat{I}_{\text{Sobol}}}{\sqrt{\sigma_{\text{NF}}^2 + \sigma_{\text{Sobol}}^2}}$.

---

## 4. Data Delivery Format
The web application must be statically hosted and decoupled from runtime Python environments:
- **Extraction Target:** A standalone telemetry file (`web_visualizer/telemetry_data.json`, size $< 10\text{ MB}$).
- **Contents:**
  - `conditions`: Pre-computed list of $\sqrt{\hat{s}}$ values $[400, 650, 1000, 1500, 1950]\text{ GeV}$.
  - `stages`: Coordinates $(u, x_{\text{marg}}, x_{\text{copula}})$ per condition at downsampled sizes ($N = 2000$).
  - `annealing`: Grid evaluations of $p_\beta(x)$ for $\beta \in [0.00, 0.08, 0.25, 0.55, 1.00]$.
  - `weights`: Evaluated weight arrays $W$ and corresponding reference values ($\hat{I}_{\text{Sobol}}$, $\sigma_{\text{Sobol}}$).

---

## 5. Strict Clarity Requirements (Code + Math + Progression)

Every view across the four panels must simultaneously present:

### 1. Active Code File & Function
- Directly display the active source file name, class, function name, and execution hardware (`cuda` vs `cpu`).
- Example Callout:
  ```text
  Active Code: nf/conditional_cubic_1d.py::forward()
  Device: CUDA (Device Tensors) -> Transferred to Host CPU for matrix2py.pyd
  ```

### 2. Governing Mathematical Formulation
- Render equations directly below active plots using KaTeX:
  - Base Generation:
    $$u \sim U([0, 1]^d)$$
  - Marginal Spline Density:
    $$q_i(x_i \mid c) = \left|\frac{\partial u_i}{\partial x_i}\right| = \frac{1}{h_k} \cdot \left(d_0 (1-t)^2 + 2 s t (1-t) + d_1 t^2\right)$$
  - Full Copula Proposal:
    $$q_\theta(x \mid c) = \prod_{i=1}^d q_i(x_i \mid c) \cdot c_\theta(u_1, \dots, u_d \mid c)$$
  - Annealing Bridge:
    $$p_{\beta_t}(x \mid c) \propto q_{t-1}(x \mid c)^{1 - \beta_t} \cdot f(x \mid c)^{\beta_t}$$
  - Unnormalized Matrix Element:
    $$f(x \mid c) = |\mathcal{M}(x \mid c)|^2 \cdot \Phi(x) \cdot \mathcal{J}(x) \cdot \Theta_{\text{cuts}}(x)$$
  - Importance Weight:
    $$W_n = \frac{f(X_n \mid c)}{r_\theta(X_n \mid c)}, \quad r_\theta = (1 - \epsilon) q_\theta + \epsilon U$$

### 3. Animated State Progression
- No static snapshots. Points must visibly translate across spaces when moving the scrubbers:
  - Scrubbing the Pipeline step ($1 \to 4$) smoothly moves points from $u \to x_{\text{marg}} \to x_{\text{copula}} \to \text{momenta}$.
  - Adjusting $\beta$ continuously morphs the density surface from flat to peaked.
  - Adjusting $\sqrt{\hat{s}}$ updates the marginal edge height ($1.49 \to 16.19$) and realigns the peak weights.
