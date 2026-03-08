#!/bin/bash
#SBATCH --job-name=Bhuvanesh
#SBATCH --partition=small
#SBATCH --gres=gpu:1g.24gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=training2d_logs.log
echo "Running on node: $(hostname)"
echo "SLURM_JOB_GPUS=$SLURM_JOB_GPUS"
export CUDA_VISIBLE_DEVICES="MIG-cc67418b-8d94-51c4-88d2-ec45ff0e92c0"
# export CONFIG_PATH="ablations/Unet3D/HN/preproc_l1_ssim/run2/config.py"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
# python3 UNET_Train.py
# python3 AFP_train.py
python UNET_Train.py \
  --run_dir /data/home/sk1022/MajorProject_sCTGen/awesome-synthetic-CT-generation/bhuvanesh/ablations/Unet2D/HN/preproc_l1_ssim_dice/run1 \
  --patch_size 192,192 \
  --num_adjacent_slices 0 \
  --use_pruned_data false \
  --use_preproc_files true \
  --epochs 80 \
  --log_file training.log \
  --wandb_mode online \
  --wandb_run_name "[abl]_UNet2D_HN_preproc_L1_SSIM_DICE_run1" \
  &

python UNET_Train.py \
  --run_dir /data/home/sk1022/MajorProject_sCTGen/awesome-synthetic-CT-generation/bhuvanesh/ablations/Unet2D/HN/pruned_preproc_l1_ssim_dice/run1 \
  --patch_size 192,192 \
  --num_adjacent_slices 0 \
  --use_pruned_data true \
  --use_preproc_files true \
  --epochs 80 \
  --log_file training.log \
  --wandb_mode online \
  --wandb_run_name "[abl]_UNet2D_HN_pruned_preproc_L1_SSIM_DICE_run1" \
  &
wait