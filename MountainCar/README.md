# MountainCar

This folder contains the notebook and runnable script for the **MountainCar** RL project.

## Contents
- `MountainCar.ipynb` — demo notebook (with outputs).
- `MountainCar_clean.ipynb` — cleaned notebook (outputs cleared) for version control.
- `mountaincar.py` — runnable script with `argparse` (auto-exported from notebook with light fixes).
- `requirements.txt` — Python dependencies.
- `.gitignore`

## Run
Create a virtual env and install requirements:
```bash
pip install -r requirements.txt
```

Run the script:
```bash
python mountaincar.py
```

Notes:
- I applied heuristic fixes to match `gymnasium`'s API where I detected older Gym usages (e.g. `env.step()` / `env.reset()` differences). Please review the script if your notebook uses custom wrappers or a different Gym version.
- If you'd like, I can further adapt the script to include `--seed`, `--train` flags and model saving/loading.
