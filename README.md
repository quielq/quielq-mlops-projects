# AI 231 — ML Operations: Machine Exercise Submissions

This repository holds my machine exercise (ME) submissions for **AI 231 (ML Operations)**.
Each exercise lives in its own top-level folder, numbered in submission order.

## For the professor: where to look

| Exercise | Folder | What to check |
|---|---|---|
| ME1 — CNN from scratch with einops/einsum | [`machine-exercise-1/`](machine-exercise-1/) | Open [`machine-exercise-1/notebooks/me1_cnn_einops_mnist.ipynb`](machine-exercise-1/notebooks/me1_cnn_einops_mnist.ipynb) directly — it contains the model, training run, logs, test-set accuracy, and the 4x4 prediction grid, all in one place. |

Supporting material for ME1:
- [`machine-exercise-1/logs/`](machine-exercise-1/logs/) — raw training/epoch logs saved outside the notebook, for traceability.
- [`machine-exercise-1/figures/`](machine-exercise-1/figures/) — exported prediction grid image.
- [`docs/gpu_environment_setup.md`](docs/gpu_environment_setup.md) — every terminal command run to set up the GPU/cluster environment for this exercise, with comments on why and when each was run.
- [`docs/agent_audit_trail.md`](docs/agent_audit_trail.md) — an audit trail of how the coding agent (Claude Code) worked through this assignment, including the prompts used and notes on how to improve them.

## Repo conventions

- One folder per machine exercise: `machine-exercise-N/`.
- Each exercise folder contains its own `notebooks/`, `logs/`, and `figures/` subfolders.
- Notebooks are meant to be read top-to-bottom and are self-contained (imports, data loading, model, training, evaluation, visualization all included).
- This repo was initialized and committed by an AI coding agent (Claude Code) working directly with the author on the University of the Philippines DGX HPC cluster. See the audit trail linked above for details.

## Environment

- Python virtual environment (`.venv/`, not committed) with `torch`, `einops`, `jupyter`, `matplotlib`.
- Trained on an NVIDIA A100-40GB GPU on the UP COE HPC cluster.
