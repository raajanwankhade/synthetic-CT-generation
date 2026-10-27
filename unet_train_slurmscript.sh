#!/bin/bash
#SBATCH --job-name=Bhuvanesh
#SBATCH --partition=small
#SBATCH --gres=gpu:1g.24gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=testing/test.log
echo "Running on node: $(hostname)"
echo "SLURM_JOB_GPUS=$SLURM_JOB_GPUS"
export CUDA_VISIBLE_DEVICES="MIG-cc67418b-8d94-51c4-88d2-ec45ff0e92c0"
# export CONFIG_PATH="ablations/Unet3D/HN/preproc_l1_ssim/run2/config.py"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
# python3 UNET_Train.py
# python3 AFP_train.py
python UNET_Train.py \
  --run_dir /data/home/sk1022/MajorProject_sCTGen/awesome-synthetic-CT-generation/bhuvanesh/testing \
  --patch_size 32,192,192 \
  --num_adjacent_slices 3d \
  --use_pruned_data false \
  --use_preproc_files true \
  --epochs 1 \
  --log_file training.log \
  # --wandb_mode online \
  # --wandb_run_name "[abl]_UNet3D_HN_preproc_L1_SSIM_DICE_run1" \
  &
# export CUDA_VISIBLE_DEVICES="MIG-b1ebaa2d-6256-57d7-a497-6c7fc5dd254b"
# python UNET_Train.py \
#   --run_dir /data/home/sk1022/MajorProject_sCTGen/awesome-synthetic-CT-generation/bhuvanesh/ablations/Unet3D/HN/pruned_preproc_l1_ssim_dice/run1 \
#   --patch_size 32,192,192 \
#   --num_adjacent_slices 3d \
#   --use_pruned_data true \
#   --use_preproc_files true \
#   --epochs 57 \
#   --log_file resumed_training.log \
#   --wandb_mode online \
#   --wandb_run_name "[abl]_UNet3D_HN_pruned_preproc_L1_SSIM_DICE_run1" \
#   --resume_path /data/home/sk1022/MajorProject_sCTGen/awesome-synthetic-CT-generation/bhuvanesh/ablations/Unet3D/HN/pruned_preproc_l1_ssim_dice/run1/checkpoints/best.pth \
#   --id_wandb ubsmnjyi \
#   --resume_wandb allow
#   &
wait
