# Q2: Lite3 navigation with PPO (PyBullet)

## Run

```bash
pip install -r requirements.txt
python run_tests.py
```

This runs the 5 tests and writes `results/`, `plots/final/` and `gifs/final/`. It takes about 5–10 minutes.

Other commands:

```bash
python plots/make_plots.py                                              # redraw the plots only
python gifs/make_gifs.py                                                # remake the GIFs only
python train_navigation.py 8000000 8 --from models/navigation_policy.zip   # train navigation
python train_walking.py 2000000 8 --omni                                # train walking
```

## Files

| File | What it does |
|---|---|
| `run_tests.py` | runs the 5 tests on the fixed map, saves the results, makes the plots and GIFs |
| `train_navigation.py` | trains the navigation policy |
| `train_walking.py` | trains the walking policy |
| `q2_navigation_env.py` | Q2 navigation task: tall goal, long-range camera, fixed map |
| `navigation_env.py` | navigation task: arena, obstacles, camera, rewards |
| `walking_env.py` | Lite3 robot in PyBullet, walking task |
| `plots/make_plots.py` | makes the plots |
| `plots/plot_style.py` | plot colours and map drawing |
| `gifs/make_gifs.py` | makes the GIFs |
| `models/` | trained policies: `navigation_policy.zip`, `walking_policy.zip` |
| `results/` | test results table and recorded runs |
| `logs/` | navigation training logs (used by the training plot) |
| `Lite3/` | robot model |
