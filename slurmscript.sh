#!/bin/bash
#SBATCH --job-name=Bhuvanesh
#SBATCH --partition=small
#SBATCH --gres=gpu:1g.24gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=runs/Unet3dHN/full_run2/logs.out
echo "Running on node: $(hostname)"
echo "SLURM_JOB_GPUS=$SLURM_JOB_GPUS"
export CUDA_VISIBLE_DEVICES="MIG-418e4605-20dd-5066-8ba8-ecaa0dcd9e2b"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
python3 UNET_Train.py
