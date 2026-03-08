#!/bin/bash
#SBATCH --job-name=Bhuvanesh
#SBATCH --partition=small
#SBATCH --gres=gpu:1g.24gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=teacher_train_logs.log
echo "Running on node: $(hostname)"
echo "SLURM_JOB_GPUS=$SLURM_JOB_GPUS"
export CUDA_VISIBLE_DEVICES="MIG-cc67418b-8d94-51c4-88d2-ec45ff0e92c0"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# python train_teacher_segmentor.py \
#   --run_dir /data/home/sk1022/MajorProject_sCTGen/awesome-synthetic-CT-generation/bhuvanesh/ablations/Unet2D/HN/preproc_l1_ssim_dice/run1 \
#   --patch_size 192,192 \
#   --num_adjacent_slices 0 \
#   --use_pruned_data false \
#   --use_preproc_files true \
#   --epochs 100 \
#   --log_file teacher_training.log \
#   &

# python train_teacher_segmentor.py \
#   --run_dir /data/home/sk1022/MajorProject_sCTGen/awesome-synthetic-CT-generation/bhuvanesh/ablations/Unet2D/HN/pruned_preproc_l1_ssim_dice/run1 \
#   --patch_size 192,192 \
#   --num_adjacent_slices 0 \
#   --use_pruned_data true \
#   --use_preproc_files true \
#   --epochs 100 \
#   --log_file teacher_training.log \
#   &

python train_teacher_segmentor.py \
  --run_dir /data/home/sk1022/MajorProject_sCTGen/awesome-synthetic-CT-generation/bhuvanesh/ablations/Unet3D/HN/preproc_l1_ssim_dice/run1 \
  --patch_size 32,192,192 \
  --num_adjacent_slices 3d \
  --use_pruned_data false \
  --use_preproc_files true \
  --epochs 100 \
  --log_file teacher_training.log \
  &

python train_teacher_segmentor.py \
  --run_dir /data/home/sk1022/MajorProject_sCTGen/awesome-synthetic-CT-generation/bhuvanesh/ablations/Unet3D/HN/pruned_preproc_l1_ssim_dice/run1 \
  --patch_size 32,192,192 \
  --num_adjacent_slices 3d \
  --use_pruned_data true \
  --use_preproc_files true \
  --epochs 100 \
  --log_file teacher_training.log \
  &
wait