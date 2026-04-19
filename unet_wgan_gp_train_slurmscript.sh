#!/bin/bash
#SBATCH --job-name=Bhuvanesh
#SBATCH --partition=small
#SBATCH --gres=gpu:1g.24gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=runs/wgan_gp/fullrun1/stdout.log

set -euo pipefail

echo "Running on node: $(hostname)"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"

# Map SLURM logical GPU id (0-7) to MIG UUID.
declare -A MIG_UUID_BY_SLURM_ID=(
  [0]="MIG-a95be726-a3ac-5445-8427-a9fba33402a4"
  [1]="MIG-418e4605-20dd-5066-8ba8-ecaa0dcd9e2b"
  [2]="MIG-093f7e1b-6f21-5e12-bc9e-435e7f019600"
  [3]="MIG-d92daec9-5afd-5a16-adac-861b9216d4f9"
  [4]="MIG-cc67418b-8d94-51c4-88d2-ec45ff0e92c0"
  [5]="MIG-b1ebaa2d-6256-57d7-a497-6c7fc5dd254b"
  [6]="MIG-283697f2-709b-5149-bf8e-30a0abe75dc4"
  [7]="MIG-83d74367-8c40-525a-9aab-fb561a0a3daa"
)

SLURM_GPU_RAW="${SLURM_JOB_GPUS:-${SLURM_STEP_GPUS:-}}"
if [[ -z "$SLURM_GPU_RAW" ]]; then
  echo "SLURM did not provide GPU id; defaulting to id=0"
  SLURM_GPU_ID=0
  SELECTED_MIG_UUID="${MIG_UUID_BY_SLURM_ID[$SLURM_GPU_ID]:-}"
else
  FIRST_TOKEN="${SLURM_GPU_RAW%%,*}"
  if [[ "$FIRST_TOKEN" == MIG-* ]]; then
    SELECTED_MIG_UUID="$FIRST_TOKEN"
    SLURM_GPU_ID="from_mig_uuid"
  elif [[ "$FIRST_TOKEN" =~ ([0-9]+) ]]; then
    SLURM_GPU_ID="${BASH_REMATCH[1]}"
    SELECTED_MIG_UUID="${MIG_UUID_BY_SLURM_ID[$SLURM_GPU_ID]:-}"
  else
    echo "ERROR: Could not parse GPU id from SLURM value: '$SLURM_GPU_RAW'"
    exit 1
  fi
fi

if [[ -z "$SELECTED_MIG_UUID" ]]; then
  echo "ERROR: No MIG UUID mapping found for SLURM GPU id '$SLURM_GPU_ID'"
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$SELECTED_MIG_UUID"
echo "SLURM GPU (raw): $SLURM_GPU_RAW"
echo "SLURM GPU id parsed: $SLURM_GPU_ID"
echo "Allotted CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

export CUDA_VISIBLE_DEVICES="MIG-d92daec9-5afd-5a16-adac-861b9216d4f9"

python3 UNET_WGAN_GP_Train.py \
  --run_dir /data/home/sk1022/MajorProject_sCTGen/awesome-synthetic-CT-generation/bhuvanesh/runs/wgan_gp/fullrun1 \
  --patch_size 32,128,128 \
  --num_adjacent_slices 3d \
  --use_pruned_data false \
  --use_preproc_files true \
  --epochs 30 \
  --batch_size 8 \
  --wandb_mode online \
  --wandb_run_group "WGAN-GP UNET3D" \
  --wandb_run_name "fullrun1" \
  --log_file train_wgan_gp.log
