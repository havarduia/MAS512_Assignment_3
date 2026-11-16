# Assignment 3 — Q1: PPO yaw control with a physics-based power equation and reward

| File / folder | Content |
|---|---|
| `rl-wind.ipynb` | original notebook of the course (the **baseline**), unchanged |
| `q1_wind_turbine_rl.ipynb` | our solution in a single cell: each section starts with a short explanation (comments) followed by its code. Already executed, outputs saved |

## What we did

1. **Power equation** — the original $P=\tfrac12\rho AC_pv^3\cos^3\gamma$ has no generator limit (4.4 MW at 15 m/s
   for an 80 m rotor). New model of a 2 MW pitch-regulated turbine: cut-in 3 m/s, rated 2 MW above 11.6 m/s,
   cut-out 25 m/s, yaw loss $\cos^{1.88}\gamma$ (Gebraad et al. 2016, NREL FLORIS). On the same winds the old
   equation counts 45 % more energy.
2. **Reward** — the original adds MW and degrees. New reward in per-unit of rated power:
   $r=-\big[(P_e(v,0)-P_e(v,\gamma))/P_\text{rated} + (P_\text{yaw,max}/P_\text{rated})\,|\omega|/\omega_\text{max} + \lambda_\gamma(1-\cos\gamma)\big]$,
   with $P_\text{yaw,max}=20$ kW and $\lambda_\gamma\approx4.1$ (curvature of the aerodynamic loss at 15 m/s).
3. **Observation fix** — the raw angles jump at 0/360° and make about 1 training run in 4 fail, for both rewards;
   encoding them as (sin, cos) removes the failures (0/40).
4. **Final comparison** (both with sin/cos, 20 seeds × 10 one-hour winds each): mean yaw error 1.61° → 1.42°,
   worst seed 2.92° → 1.52°, time within ±5° 97.3 % → 99.0 %, yaw motion −9.8 %, same energy (98.3 % of ideal).

## Environment

From the `Assignment3` folder (Python 3.10):

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements_modified.txt
.venv/bin/python -m ipykernel install --user --name assignment3-venv --display-name "Assignment3 (.venv, Python 3.10)"
```

Open the notebook with the kernel **Assignment3 (.venv, Python 3.10)**.

## Running

`models/` (trained PPO models), `logs/` (training curves, evaluation tables) and `figures/` (every
figure of the notebook as PNG: power curve, baseline reward terms, typical-seed comparison, 20-seed
distribution, tracking window) are **not** in the repository (see `.gitignore`); the notebook creates them. The
figures are also embedded in the executed notebook.

* First run: all models are trained in parallel processes (90 runs of 300 000 steps, about 1.5 h on a 16-thread
  CPU), then evaluated.
* Later runs (`RETRAIN = False`, default): the saved models and logs are loaded; the whole notebook runs in about
  5 minutes.
* `RETRAIN = True` trains everything again. Training is deterministic (CPU, fixed seeds): same numbers.

## Variants trained in the notebook

| Variant | Power equation | Reward | Observation |
|---|---|---|---|
| `baseline` | original | original | raw angles |
| `physics` | new | new (energy-lost form) | raw angles |
| `physics_absolute` | new | $+P/P_\text{rated}$ form (ablation) | raw angles |
| `physics_lambda1` | new | $\lambda_\gamma=1$ (ablation) | raw angles |
| `baseline_circ`, `physics_circ` | as above | as above | (sin, cos) angles |
