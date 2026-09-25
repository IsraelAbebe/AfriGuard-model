#!/bin/bash -l
# Usage: sbatch train_plain.sbatch examples/train_full/afriguard_plain_afriqueqwen3.5-4b-50langs-instruct_full_sft.yaml
# One GPU per run to match the original AfriGuard-inst runs (same effective batch size).
#SBATCH --job-name=afriguard-plain
#SBATCH --account=kempner_kdbrantley_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
CONFIG=${1:?usage: sbatch train_plain.sbatch <config.yaml>}

module purge
module load cuda/13.3.1-fasrc01
source .venv/bin/activate

export HF_HOME=/n/netscratch/kdbrantley_lab/Lab/adzcai/.cache/huggingface
export HF_TOKEN=${HF_TOKEN:-$(cat .hf_token)}
# Per-job datasets cache: runs sharing a tokenizer otherwise rewrite each other's mmapped arrow files (SIGBUS).
export HF_DATASETS_CACHE=/tmp/hf-datasets-$SLURM_JOB_ID
export TRITON_CACHE_DIR=/tmp/triton-$SLURM_JOB_ID
export WANDB_DISABLED=true

echo "config=$CONFIG node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"
FORCE_TORCHRUN=1 llamafactory-cli train "$CONFIG"
