# Agent Audit Trail: AI 231 ME1

This is a running record of how the coding agent (Claude Code, using Claude Sonnet 5)
worked through this assignment: what was asked, what the agent did in response, and a
note on how each prompt could be made tighter next time. It exists so the process is
reviewable later, separate from the deliverable itself.

Convention: each entry quotes the user's prompt (trimmed only for length where noted),
summarizes what the agent actually did, and ends with a **Prompt improvement note**.

---

## Prompt 1: Initial assignment brief

> Use of the UPD DGX Cluster:
> 1. run nvidia-smi to see available A100 GPUs in the cluster. If the work and training
>    below requires it, connect and use one of the available clusters. If there are no
>    available GPUs, let me know before proceeding.
> 2. Any code executed in the terminal must be listed down in an output markdown file.
>    Add comments for explainability on why each part of the code was executed and when
>    they were executed.
>
> GitHub Repository Setup:
> 1. Turn this folder into a github repo that we can publish in my github account.
> 2. All machine exercise submissions for my AI 231 - ML Operations class should be
>    published here. Add a README file that will help my professor navigate where he
>    should check my machine exercises. For now, I am only doing my first machine
>    exercise
> 3. We must submit this machine exercise agent-initialized and agent-committed.
>
> Machine Exercise:
> 1. Build a 3-layer CNN model for MNIST classification by implementing all
>    layers/operations using einops/einsum. Note that we are allowed to use pytorch
>    tensors, but NOT neural nets or MLPs
> 2. Train the model for 5 epochs and report the test split accuracy.
> 3. Sample 16 images and put them in 4x4 grid together with gt and prediction. Display
>    all in a jupyter notebook.
> 4. Show the model, training, logs, and visualizations. Make sure to visualize and
>    report on the test set. Make it clear in the code that we're reporting on the test
>    set.
> 5. Include line-by-line comments on what each part of the code does, making sure that
>    the context from our MLops classes are explained as seen in the class lecture
>    files attached. We will be tested on these later on, so we must make sure that the
>    code is explainable and easy to understand for further review and reporting.
> 6. Deliver the output first so I can review, then seek confirmation from me
>
> Overall explainability:
> 1. Create a markdown file that explains how the agent worked on the project. This
>    should be like an audit trail that I can review later on. Include in the markdown
>    file my prompts, and add a comment below each prompt on how we can further improve
>    our prompts moving forward.
> 2. For any clarifications to deliver the best output, ask me so that we can clarify.

**What the agent did:** Ran `nvidia-smi` (found 8x A100-40GB, 6 free), surveyed the
environment (no conda, no `gh`, no git identity, SLURM available but unused since the
agent already had a live shell on a GPU node), then asked four clarifying questions
before touching anything risky: (1) run directly on the node vs. submit via SLURM, (2)
how to authenticate GitHub publishing, (3) where the referenced "class lecture files"
actually were, (4) desired repo name/visibility. See Prompt 2 below for the answers.

**Prompt improvement note:** This prompt bundled four largely independent workstreams
(cluster check, GitHub setup, the ML exercise itself, and a meta-request for an audit
trail) into one message. That's fine for an agent that asks clarifying questions first.
For a *faster* turnaround it would help to either (a) split these into separate
messages so each can be confirmed and closed out independently, or (b) front-load the
answers to the predictable ambiguities up front: where the "attached" lecture files
actually live (a path, since nothing was literally attached to the chat), what GitHub
auth method is preferred, and desired repo name/visibility. The phrase "class lecture
files attached" assumed the files were part of the chat context when they were actually
a path on disk (`~/ai231-lectures`) the agent had to be told about explicitly.

---

## Prompt 2: Answers to clarifying questions

> "How should I run the training on the cluster?" **Run directly on this node**
> "How do you want me to publish the repo to your GitHub account?" **install gh so we
> can publish agentically**
> "The task mentions 'class lecture files attached' ... how should I proceed?" **I'll
> provide the files/path now**
> "What should the GitHub repo be named, and should it be public or private?"
> **quielq-mlops-projects, public**

**What the agent did:** Installed `gh` CLI locally (no root needed, since this is a
shared HPC node), authenticated it via OAuth device flow (background process, user
completed the browser step), created the local `.venv`, and initialized the git repo
with local (not global) commit identity. All logged in
[`gpu_environment_setup.md`](gpu_environment_setup.md).

**Prompt improvement note:** These answers were efficient. Each was a single decisive
choice with no hedging, which let the agent proceed without a second round of
questions. Choosing "install gh so we can publish agentically" over supplying a token
directly was a good call for reusability (the agent can now use `gh` for future
machine exercises in this same repo without asking again).

---

## Prompt 3: Lecture file location

> "Here's the path to the lecture slides: /ai231-lectures"

**What the agent did:** Checked `/ai231-lectures`. It did not exist. Reported this
clearly rather than guessing or silently falling back to generic explanations.

**Prompt improvement note:** The path given was an absolute path from filesystem root,
but the files actually lived under the user's home directory
(`~/ai231-lectures`). A `~`-relative or fully-qualified home path avoids this kind of
back-and-forth. Minor, but worth noting since it cost a round trip.

---

## Prompt 4: Corrected path

> "Check this directory: ~/ai231-lectures"
> *(This followed the agent attempting `ls /` to hunt for the directory itself, which
> the user interrupted/rejected. That was reasonable, since broad filesystem exploration
> on a shared HPC login node is the kind of thing worth steering rather than letting an
> agent free-search.)*

**What the agent did:** Listed `~/ai231-lectures`, found five PDFs: `0_Toolkit.pdf`,
`1_Toolkit_Python.pdf`, `2_Toolkit_Numpy.pdf`, `3_Toolkit_Einsum.pdf`,
`4_Toolkit_Einops.pdf`, Dr. Rowel Atienza's UP "Deep Learning Toolkit" lecture series.
Read the Einsum and Einops decks in full (most directly relevant to the CNN-via-einops
exercise) to ground the notebook's comments in the exact notation and framing taught in
class (Einstein summation convention, `rearrange`/`repeat`/`reduce`/`pack`/`unpack`,
the "2D conv output must reshape to match MLP input" motivation for einops, etc.).

**Prompt improvement note:** Correcting a wrong path with the minimal fix ("Check this
directory: ~/ai231-lectures") rather than re-explaining the whole request was the
right move. It let the agent resume immediately without re-deriving context. This is
the pattern worth repeating: when redirecting a stuck agent, give the smallest correction
that unblocks it rather than restating the goal.

---

## Notebook build, training, and results (no new user prompt, continuation of Prompt 1's task list)

**What the agent did:** Installed `torch`, `torchvision`, `einops`, `jupyter`,
`matplotlib`, `nbformat`/`nbconvert` into the project `.venv`. Designed and generated
`machine-exercise-1/notebooks/me1_cnn_einops_mnist.ipynb`: a 3-layer CNN (2 conv layers
implemented via `Tensor.unfold` + `einops.rearrange` + `einops.einsum`, max-pooling via
`einops.reduce`, a linear classifier head via `einops.einsum`) with a manually
implemented cross-entropy loss, trained for 5 epochs with `torch.optim.Adam` on GPU 0.
Executed the notebook end-to-end with `jupyter nbconvert --execute` on the cluster GPU
and verified zero errors across all 9 code cells before reporting results.

**Result:** Test-set accuracy **98.42%** (9,842/10,000), about 9.5 seconds per epoch.
All 16 sampled test images in the 4x4 visualization grid were correctly classified. Full
per-epoch log in [`../machine-exercise-1/logs/train_log.txt`](../machine-exercise-1/logs/train_log.txt),
grid image in [`../machine-exercise-1/figures/prediction_grid.png`](../machine-exercise-1/figures/prediction_grid.png).

**Per the user's explicit instruction 6 under "Machine Exercise"** ("Deliver the output
first so I can review, then seek confirmation from me"), the agent stopped here, before
any `git add`/`git commit`/`git push`, to present these results for review rather than
proceeding straight to publishing.

---

## Prompt 5: Commit

> "Commit the working tree changes with a sensible message."

**What the agent did:** Found an untracked `.claude/settings.local.json` (local Claude
Code permission config, not part of the coursework) and added `.claude/` to
`.gitignore` rather than committing it. Staged the seven deliverable files by name
(README, docs, and everything under `machine-exercise-1/`) and committed as
`21b714f` with a message summarizing the implementation approach and the 98.42% test
result, co-authored per the repo's convention.

**Prompt improvement note:** Short and unambiguous. "Commit ... with a sensible
message" correctly left the message wording to the agent's judgment while being clear
about scope (working tree changes, not "everything including local config").

---

## Prompt 6: Push

> "push quielq-mlops-projects"

**What the agent did:** Confirmed the repo didn't already exist on GitHub, created it
as public via `gh repo create ... --source=. --remote=origin`, then hit a plain
`git push` credential error (`git` wasn't wired to use `gh`'s stored OAuth token).
Fixed with `gh auth setup-git`, then pushed `main` successfully. Verified the repo is
live and public at https://github.com/quielq/quielq-mlops-projects.

**Prompt improvement note:** Terse and correct. The repo name alone was enough context
given it was already agreed in Prompt 2 (public, this exact name). Nothing to improve
here.

---

## Prompt 7: Dash sweep

> "Can you do a clean sweep of the repository and remove en and em dashes from the
> codebase, comments, and etc. unless absolutely necessary? I prefer simple and direct
> sentences for readability and easier understanding."

**What the agent did:** Found and rewrote every em/en dash across `README.md`,
`docs/gpu_environment_setup.md`, `docs/agent_audit_trail.md` (this file), and the
ME1 notebook's markdown cells and code comments, replacing them with periods, commas,
or colons, and plain hyphens for numeric ranges (e.g. "GPU 0-5"). Regenerated and
re-executed the notebook afterward so its committed outputs stayed in sync with the
edited cell source (results unchanged: 98.42% test accuracy, same fixed seed). Opened
as its own pull request rather than amending the already-open commit/push PR, since it
was a separate, repo-wide concern.

**Prompt improvement note:** Clear and specific about both the mechanical rule (remove
em/en dashes) and the underlying preference (simple, direct sentences), which let the
agent make good judgment calls on rephrasing rather than just deleting characters. The
"unless absolutely necessary" qualifier was useful: it's why numeric ranges kept a
plain hyphen instead of being spelled out as "0 to 5".

---

*(This trail is appended to as further machine exercises are added to this repo.)*
