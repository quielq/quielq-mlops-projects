# GPU / Cluster Environment Setup Log: AI 231 ME1

This file records every terminal command the coding agent (Claude Code) ran to set up
the environment on the UP COE HPC cluster (DGX-class node `ai-n002.hpc.coe.upd.edu.ph`,
8x NVIDIA A100-SXM4-40GB) before writing any exercise code. Commands are grouped by
purpose, in the order they were actually run, with a comment on **why** and **when**
(relative to the session) each was executed.

## 1. Check GPU availability on the cluster

```bash
nvidia-smi
```
**Why:** First step required by the assignment. Confirm A100s are visible and see how
many are free before committing to training on this node.
**When:** Very first command of the session, before anything else.
**Result:** 8x A100-SXM4-40GB. GPUs 0-5 were completely idle (0% util, 0 MiB used).
GPUs 6-7 were ~37-40% utilized by another user's job (~10.4 GB each), so those two were
avoided.

## 2. Orient in the filesystem

```bash
pwd && ls -la && hostname && whoami
```
**Why:** Confirm the working directory (`/mnt/jfs_hpc/home/quiel.andrew.quiwa/quielq-mlops-projects`)
was empty and safe to initialize as a fresh repo, and confirm which host/user we're
operating as.
**When:** Immediately after the GPU check.

## 3. Check tooling already available (git, gh, python, torch, einops, jupyter)

```bash
git --version
gh auth status
gh config list
which python python3 conda
python3 --version
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python3 -c "import einops; print(einops.__version__)"
python3 -c "import jupyter; print('jupyter ok')"
```
**Why:** Avoid assuming tools exist. `gh` turned out to be missing entirely, and the
system Python had none of `torch`/`einops`/`jupyter` installed.
**When:** Right after orienting in the filesystem, before any installs.
**Result:** `gh` not found; system `python3` had no ML packages; `git` was present.

## 4. Look for a conda installation and SLURM

```bash
conda env list
which conda mamba
ls ~/.conda/envs ~/miniconda3 ~/anaconda3
module avail
which sbatch srun squeue
```
**Why:** The other user's GPU processes were running under a `.conda/envs/rmr` path,
which suggested conda might be available. It wasn't, for this user, so a plain `venv`
was used instead. Also checked whether the cluster requires SLURM job submission
instead of running directly on the node.
**When:** After discovering no ML packages were installed, before deciding on an
environment strategy.
**Result:** No conda for this user. SLURM (`sbatch`/`srun`/`squeue`) is available, but
since the agent already has a live shell on a GPU compute node with free A100s, training
was run directly on the node rather than queued. This was confirmed with the user first.

## 5. Check internet access and disk space for package installs

```bash
python3 -m pip --version
curl -sI https://pypi.org
df -h /mnt/jfs_hpc/home/quiel.andrew.quiwa
```
**Why:** Before attempting `pip install torch` (a large download), confirm outbound
internet works from the compute node and that there's enough disk space.
**When:** Right before installing anything.
**Result:** pip present, PyPI reachable, ~962 TB free on the shared JuiceFS mount.

## 6. Install the GitHub CLI (`gh`) into the user's local bin, no root required

```bash
mkdir -p ~/.local/bin
GH_VER=$(curl -s https://api.github.com/repos/cli/cli/releases/latest | grep -oP '"tag_name": "v\K[^"]+')
curl -sL "https://github.com/cli/cli/releases/download/v${GH_VER}/gh_${GH_VER}_linux_amd64.tar.gz" -o gh.tar.gz
tar -xzf gh.tar.gz
cp gh_${GH_VER}_linux_amd64/bin/gh ~/.local/bin/gh
chmod +x ~/.local/bin/gh
gh --version
```
**Why:** The user asked to publish the repo to GitHub "agent-initialized and
agent-committed," which requires `gh` for authenticated repo operations. This HPC node
has no root access, so the binary was downloaded directly into `~/.local/bin` rather
than installed via a system package manager.
**When:** After the user chose "install gh so we can publish agentically" in response
to a clarifying question about GitHub auth.

## 7. Create a project-local Python virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip -q
```
**Why:** Isolate ME1's Python dependencies (`torch`, `einops`, `jupyter`, `matplotlib`)
from the system Python, following the same `venv` workflow taught in the course's
"Deep Learning Toolkit" lecture (`0_Toolkit.pdf`, slides 23-26), since conda was not
available for this user.
**When:** In parallel with the `gh` install, since neither depends on the other.

## 8. Authenticate `gh` via OAuth device flow

```bash
gh auth login --hostname github.com --git-protocol https --scopes repo
```
**Why:** Non-interactive session, so a normal `gh auth login` menu wouldn't work. The
device-flow mode (prints a one-time code + URL) does work without a TTY, as long as a
human completes the code entry in their own browser. Run in the background so the user
could complete the browser step while other setup continued.
**When:** After confirming with the user they wanted `gh` installed and authenticated
directly, rather than supplying a Personal Access Token manually.
**Result:** Authenticated as GitHub user `quielq` with `repo` scope.

## 9. Initialize the git repository and scaffold folders

```bash
git init -q
git config --local user.name "Quiel Andrew Quiwa"
git config --local user.email "quielquiwa@gmail.com"
git branch -M main
mkdir -p machine-exercise-1 docs
cat > .gitignore <<'EOF'
.venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
data/
*.pt
*.pth
.DS_Store
EOF
```
**Why:** Turn the folder into a git repo per the user's request, using **local** (not
global) git config so this doesn't alter shared HPC account defaults for other
projects. `.gitignore` excludes the virtual environment, downloaded MNIST data, and
model checkpoints so the repo stays lightweight.
**When:** After `gh` auth succeeded, before writing any exercise code.

## 10. Install ML packages into the venv

```bash
source .venv/bin/activate
pip install -q torch einops jupyter matplotlib numpy ipykernel
python -c "import torch, einops; print(torch.__version__, torch.cuda.is_available(), einops.__version__)"
```
**Why:** These are exactly the libraries the exercise permits: PyTorch tensors, plus
`einops`/`einsum` for the layer math, and `jupyter`/`matplotlib` to author and run the
required notebook with visualizations.
**When:** Run in the background immediately after git init, since it's a long download
(PyTorch's CUDA-enabled wheel) that doesn't block other setup work.

---

*This log will be appended to as training and evaluation commands are run for ME1.*
