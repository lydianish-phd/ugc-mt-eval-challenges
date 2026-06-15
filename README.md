# Evaluation Challenges

## Part I: Inference on Jean Zay (H100 / gpu_p6)

This repository contains the code and data for the evaluation challenge experiments. The immediate goal is to run **inference** for the two newly added models:

- `Qwen/Qwen2.5-7B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`

These model IDs are defined in `src/constants.py`.

---

### 1. Load the Jean Zay environment

On Jean Zay (H100 / gpu_p6), load the required modules:

```bash
module purge
module load arch/h100
module load pytorch-gpu/py3/2.5.0
source ~/.bashrc
````

---

### 2. Check whether `vllm` is already available

```bash
python -c "import vllm; print(vllm.__version__)"
```

If this fails, install `vllm`.

---

### 3. Install `vllm` (if needed)

Derive the CUDA tag from the current PyTorch environment:

```bash
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
CUDA_TAG=$(python -c "import torch; print('cu' + torch.version.cuda.replace('.', ''))")

echo "Torch version: $TORCH_VERSION"
echo "CUDA tag: $CUDA_TAG"
```

Install:

```bash
pip install --user --no-cache-dir vllm --extra-index-url https://download.pytorch.org/whl/${CUDA_TAG}
```

Verify:

```bash
python -c "import vllm; print(vllm.__version__)"
```

---

### 4. Download Hugging Face models (on login node only)

Compute nodes do **not** have internet access. You must pre-download models.

```bash
huggingface-cli login
```

Then:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download("Qwen/Qwen2.5-7B-Instruct")
snapshot_download("mistralai/Mistral-7B-Instruct-v0.3")
PY
```

Optional check:

```bash
python - <<'PY'
from transformers import AutoTokenizer

AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
print("Models available in cache.")
PY
```

---

### 5. Clone the repository

```bash
git clone https://github.com/lydianish-phd/evaluation-challenges.git
cd evaluation-challenges
```

---

### 6. Extract the datasets

```bash
bash slurm/extract_data.sh
```

This creates:

```
data_extracted/
```

---

### 7. Create a working branch

```bash
git checkout -b add-mistral-qwen
```

---

### 8. Update the SLURM log path and account

Edit the SLURM script:

```bash
slurm/generate.slurm
```

You need to update **both the log path and the account name**, as they are currently hard-coded for a specific user/project.

#### 8.1 Update the log path

Replace the existing line:

```bash
#SBATCH --output=/lustre/fsn1/projects/rech/ncm/udc54vm/evaluation-challenges/logs/%x/%x_%j.log
```

with your own project and user path, for example:

```bash
#SBATCH --output=/lustre/fsn1/projects/rech/<project>/<user>/evaluation-challenges/logs/%x/%x_%j.log
```

Make sure the directory exists:

```bash
mkdir -p /lustre/fsn1/projects/rech/<project>/<user>/evaluation-challenges/logs/generate
```

#### 8.2 Update the account

The script currently uses an account like:

```bash
#SBATCH --account=ncm@h100
```

Here:

* `ncm` is the project/account name
* `@h100` specifies the GPU partition

You must replace `ncm` with your own project/account if different. 

Make sure the account matches both:

* your allocation on Jean Zay
* the GPU partition you are using

⚠️ If you do not update these correctly, the job may fail to submit or logs may not be written.

---

### 9. Verify model selection in SLURM script

`slurm/generate.slurm` already targets the correct models:

```bash
for i in {3..4}
do
    ...
done
```

These correspond to:

* `Qwen/Qwen2.5-7B-Instruct`
* `mistralai/Mistral-7B-Instruct-v0.3`

---

### 10. Launch inference

```bash
sbatch slurm/generate.slurm
```

Outputs will be written to:

```
experiments/eamt2026/outputs/<model_name>/<corpus>/
```

---

### Notes

* The script is configured for **gpu_p6 (H100)**. Do not change partition or architecture.
* Ensure Hugging Face cache is visible from compute nodes.
* Do not run on `main`; use your branch.
* Commit any fixes (paths, environment tweaks) before launching large jobs.


## Part II: Evaluation on Jean Zay

After inference has completed, switch to the **evaluation** environment and run the metric computation and aggregation pipeline for the two newly added models:

- `Qwen/Qwen2.5-7B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`

The current evaluation script is:

```bash
slurm/evaluate.slurm
```

The metric models used by `src/evaluate.py` are:

* `Unbabel/wmt22-comet-da`
* `Unbabel/wmt22-cometkiwi-da`

Do **not** download or use `Unbabel/XCOMET-XL` for now.

---

### 1. Load the evaluation environment

Use the Jean Zay PyTorch 2.3.0 module for evaluation:

```bash
module purge
module load pytorch-gpu/py3/2.3.0
source ~/.bashrc
```

---

### 2. Check whether the required packages are available

First check whether `unbabel-comet` and `sacrebleu` are already installed:

```bash
python -c "import comet; print('comet ok')"
python -c "import sacrebleu; print('sacrebleu ok')"
```

If one of these fails, install both locally:

```bash
pip install --user --no-cache-dir unbabel-comet sacrebleu
```

Optional verification:

```bash
python -c "import comet, sacrebleu; print('Packages available.')"
```

---

### 3. Download the COMET models on the login node

As with inference, compute nodes do **not** have internet access, so download the evaluation models on the login/home node first.

The models to cache are:

* `Unbabel/wmt22-comet-da`
* `Unbabel/wmt22-cometkiwi-da`

If needed:

```bash
huggingface-cli login
```

Then download them:

```bash
python - <<'PY'
from comet import download_model

download_model("Unbabel/wmt22-comet-da")
download_model("Unbabel/wmt22-cometkiwi-da")
print("COMET models downloaded.")
PY
```

Do **not** download:

```text
Unbabel/XCOMET-XL
```

---

### 4. Make sure the repository branch is up to date

If you are continuing from the inference step:

```bash
cd evaluation-challenges
git checkout add-mistral-qwen
```

If needed, pull the latest changes on that branch before running evaluation.

---

### 5. Update the SLURM log path and account

Edit the SLURM script:

```bash
slurm/evaluate.slurm
```

You need to update **both the log path and the account name**, as they are currently hard-coded for a specific user/project.

#### 5.1 Update the log path

Replace the existing line:

```bash
#SBATCH --output=/lustre/fsn1/projects/rech/ncm/udc54vm/evaluation-challenges/logs/%x/%x_%j.log
```

with your own project and user path, for example:

```bash
#SBATCH --output=/lustre/fsn1/projects/rech/<project>/<user>/evaluation-challenges/logs/%x/%x_%j.log
```

Make sure the directory exists:

```bash
mkdir -p /lustre/fsn1/projects/rech/<project>/<user>/evaluation-challenges/logs/evaluate
```

#### 5.2 Update the account

The script currently uses an account like:

```bash
#SBATCH --account=ncm@v100
```

Here:

* `ncm` is the project/account name
* `@v100` specifies the GPU partition

You must replace `ncm` with your own project/account if different. 

Make sure the account matches both:

* your allocation on Jean Zay
* the GPU partition you are using

⚠️ If you do not update these correctly, the job may fail to submit or logs may not be written.

---

### 6. Check the environment and paths in `slurm/evaluate.slurm`

The current script already uses:

* module: `pytorch-gpu/py3/2.3.0`
* models: `Qwen/Qwen2.5-7B-Instruct` and `mistralai/Mistral-7B-Instruct-v0.3`
* corpora: `rocsmt footweets mmtc pfsmb`
* metrics: `bleu comet cometkiwi`

It expects:

* generated outputs in:

```bash
experiments/eamt2026/outputs
```

* extracted data in:

```bash
data_extracted
```

So before running evaluation, make sure inference has finished and the generated outputs are present under:

```bash
experiments/eamt2026/outputs/<model>/<corpus>/
```

---

### 7. Launch evaluation

Submit the evaluation job:

```bash
sbatch slurm/evaluate.slurm
```

This runs:

* `src/postprocess.py`
* `src/evaluate.py`
* `src/significance.py`
* `src/aggregate.py`
* `src/analyze.py`
* `src/plot_delta.py`
* `src/make_score_tables.py`

The outputs are written under:

```bash
experiments/eamt2026/
```

including score files, plots, and tables.

---

### 8. Finalise the branch

Once both inference and evaluation have completed successfully and all expected outputs have been written to `experiments/eamt2026/`, commit and push the branch:

```bash
git status
git add .
git commit -m "Add Qwen and Mistral inference and evaluation outputs"
git push origin add-mistral-qwen
```

Then open a merge request for review.

---

### Notes

* Keep this step in the **PyTorch 2.3.0** environment; do not reuse the inference environment.
* Do not enable xCOMET for now.
* If the COMET cache is not visible from compute nodes, you may need to redirect/cache models in a shared location before launching the job.
* The branch to use for all changes is `add-mistral-qwen`.

