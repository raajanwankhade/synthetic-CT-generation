#!/bin/bash
#SBATCH --job-name=Bhuvanesh
#SBATCH --partition=small
#SBATCH --gres=gpu:1g.24gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=ts_test2.log
echo "Running on node: $(hostname)"
echo "SLURM_JOB_GPUS=$SLURM_JOB_GPUS"
export CUDA_VISIBLE_DEVICES="MIG-a95be726-a3ac-5445-8427-a9fba33402a4"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

python3 -u UNET_Train_TS.py \
    --run_dir /data/home/sk1022/MajorProject_sCTGen/awesome-synthetic-CT-generation/bhuvanesh/runs/TS_alternate/test2 \
    --root /data/home/sk1022/MajorProject_sCTGen/awesome-synthetic-CT-generation/bhuvanesh/datasets/bhuvaneshsingla/synthrad2025-hn/versions/Experiment \
    --log_file testing.log \
    --binary_teacher_path /data/home/sk1022/MajorProject_sCTGen/awesome-synthetic-CT-generation/bhuvanesh/ablations/Unet3D/HN/preproc_l1_ssim_dice/run1/checkpoints/frozen_teacher.pth \
    --wandb_mode disabled \
    --wandb_run_name "test2" \
    --TS_epochs 3 \
    --regular_epochs 2 \
    --ts_fraction 0.5 \
    --ts_batch_size 8 \
    --ts_seg_classes all \
    --surrogate_weight 0.5 \
    --surrogate_lr 1e-3 \
    --ts_min_dice_for_cache 0.0 \
    --persist_ts_cache true