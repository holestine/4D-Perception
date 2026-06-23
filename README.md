# 4D-Perception
Multi-Object Tracking in 3D

## Setup

### Conda Environment

This project uses the `4D` conda environment. Select it as the kernel when running notebooks in VSCode.

### Rerun Notebook Widget

The Rerun viewer widget loads its JavaScript from a CDN by default, which may time out in offline or restricted network environments. Set the `RERUN_NOTEBOOK_ASSET` conda env var to `serve-local` so rerun serves the bundled widget from disk instead of fetching from the CDN:

```bash
conda env config vars set RERUN_NOTEBOOK_ASSET=serve-local -n 4D
```

Reactivate the environment (`conda activate 4D`) for the variable to take effect. The notebooks also set this explicitly via `os.environ` as a fallback for stale kernel sessions.
