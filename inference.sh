#!/bin/bash
#SBATCH --job-name=Bhuvanesh
#SBATCH --partition=small
#SBATCH --gres=gpu:1g.24gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=inference_logs.out
echo "Running on node: $(hostname)"
echo "SLURM_JOB_GPUS=$SLURM_JOB_GPUS"
export CUDA_VISIBLE_DEVICES="MIG-cc67418b-8d94-51c4-88d2-ec45ff0e92c0"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

python -m inference.model_upload \
--model_type 3d \
--checkpoint /data/home/sk1022/MajorProject_sCTGen/awesome-synthetic-CT-generation/bhuvanesh/runs/Unet3dHN/full_run2/checkpoints/best.pth \
--export_path /data/home/sk1022/MajorProject_sCTGen/awesome-synthetic-CT-generation/bhuvanesh/runs/Unet3dHN/full_run2/checkpoints/scripted.pth

echo "MODEL UPLOADED"

python -m inference.model_test

echo "INFERENCE MADE"

python -m inference.postproc \
--org_ct /data/home/sk1022/MajorProject_sCTGen/awesome-synthetic-CT-generation/bhuvanesh/datasets/bhuvaneshsingla/synthrad2025-hn/versions/3/Task1 \
--sct /data/home/sk1022/MajorProject_sCTGen/awesome-synthetic-CT-generation/bhuvanesh/inference/inference_outputs

echo "POST PROCESSING DONE"

python -m inference.calc_metrics

echo "METRICS CALCULATED"
