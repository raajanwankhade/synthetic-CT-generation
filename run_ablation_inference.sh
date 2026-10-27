#!/bin/bash
#SBATCH --job-name=Bhuvanesh
#SBATCH --partition=small
#SBATCH --gres=gpu:1g.24gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=inference_3d_ppd_logs.log
echo "Running on node: $(hostname)"
echo "SLURM_JOB_GPUS=$SLURM_JOB_GPUS"
export CUDA_VISIBLE_DEVICES="MIG-cc67418b-8d94-51c4-88d2-ec45ff0e92c0"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
python -u ablation_inferences.py --outputs-root inference/ablation_outputs_3d_ppd --stop-on-error