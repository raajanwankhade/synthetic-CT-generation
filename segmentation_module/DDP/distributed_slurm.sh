#!/bin/bash
#SBATCH --job-name=Bhuvanesh
#SBATCH --partition=small
#SBATCH --gres=gpu:1g.24gb:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=segmentation.log

# MIG Multi-GPU Training for UNet3D Segmentation
set -e

echo "Job $SLURM_JOB_ID on $(hostname) started at $(date)"

# --------------------------------------------------
# MIG UUIDs to use (discover via: nvidia-smi -L)
# --------------------------------------------------
MIG_UUIDS=(
    "MIG-a95be726-a3ac-5445-8427-a9fba33402a4"  # GPU0 Device 0
    "MIG-418e4605-20dd-5066-8ba8-ecaa0dcd9e2b"  # GPU0 Device 1
    "MIG-093f7e1b-6f21-5e12-bc9e-435e7f019600"  # GPU0 Device 2
    "MIG-d92daec9-5afd-5a16-adac-861b9216d4f9"  # GPU0 Device 3
    "MIG-cc67418b-8d94-51c4-88d2-ec45ff0e92c0"  # GPU1 Device 0
    "MIG-b1ebaa2d-6256-57d7-a497-6c7fc5dd254b"  # GPU1 Device 1
    "MIG-283697f2-709b-5149-bf8e-30a0abe75dc4"  # GPU1 Device 2
    "MIG-83d74367-8c40-525a-9aab-fb561a0a3daa"  # GPU1 Device 3
)

NUM_GPUS=${#MIG_UUIDS[@]}
[[ $NUM_GPUS -eq 0 ]] && { echo "No MIG UUIDs configured"; exit 1; }

# For >1 MIG slice on same bus, use gloo (NCCL unsupported)
BACKEND=$([[ $NUM_GPUS -le 1 ]] && echo nccl || echo gloo)

# Export CUDA devices
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${MIG_UUIDS[*]}")
export MIG_UUIDS_CSV=$(IFS=,; echo "${MIG_UUIDS[*]}")  # Add this line
export DIST_BACKEND="$BACKEND"
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_DEBUG=WARN

echo "GPUs: $NUM_GPUS | Backend: $BACKEND | CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

INPUT_FOLDER="/data/home/sk1022/MajorProject_sCTGen/awesome-synthetic-CT-generation/bhuvanesh/datasets/segmentation_dataset/content/drive/MyDrive/synthrad2025_train_nii_dataset/synthRAD2025_Task1_Train/synthRAD2025_Task1_Train/Task1/HN/"
OUTPUT_FOLDER="/data/home/sk1022/MajorProject_sCTGen/awesome-synthetic-CT-generation/bhuvanesh/datasets/segmentation_dataset/Task1/HN_segmentations"
ANATOMY="HN"
# --------------------------------------------------
# Launch
# --------------------------------------------------
if [[ $NUM_GPUS -gt 1 ]]; then
    torchrun --standalone --nproc_per_node=$NUM_GPUS \
        --master_port=$((29500 + RANDOM % 1000)) generate_TS_segmentations.py \
        --input_folder "$INPUT_FOLDER" \
        --output_folder "$OUTPUT_FOLDER" \
        --anatomy "$ANATOMY" \
        --dist_backend "$BACKEND"
    # torchrun --standalone --nproc_per_node=$NUM_GPUS \
    #     --master_port=$((29500 + RANDOM % 1000)) train_segmentator.py 
else
    python generate_TS_segmentations.py \
        --input_folder "$INPUT_FOLDER" \
        --output_folder "$OUTPUT_FOLDER" \
        --anatomy "$ANATOMY" \
        --dist_backend "$BACKEND"
    # python train_segmentator.py
fi

echo "Completed at $(date)"
