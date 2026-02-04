#!/bin/bash
#SBATCH --job-name=Bhuvanesh
#SBATCH --partition=small
#SBATCH --gres=gpu:1g.24gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs_3d_hn.out
echo "Running on node: $(hostname)"
echo "SLURM_JOB_GPUS=$SLURM_JOB_GPUS"

# --- HACK: assume GPU 0, MIG index = SLURM_JOB_GPUS ---
TARGET_MIG=$SLURM_JOB_GPUS

UUID=$(nvidia-smi -L | \
  awk -v m="Device  $TARGET_MIG" '
    /^GPU 0:/ {f=1; next}
    /^GPU 1:/ {f=0}
    f && $0 ~ m {print $NF}
  ' | tr -d '()')

if [ -z "$UUID" ]; then
  echo "ERROR: Could not find MIG UUID for GPU0 Device $TARGET_MIG"
  exit 1
fi

export CUDA_VISIBLE_DEVICES=$UUID
echo "Mapped MIG UUID: $UUID"


echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
##nvidia-smi -L
python3 model_test.py
