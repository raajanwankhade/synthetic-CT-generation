#!/bin/bash
#SBATCH --job-name=Bhuvanesh
#SBATCH --partition=small
#SBATCH --gres=gpu:1g.24gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=gradcam_pp_smalldataset.log
echo "Running on node: $(hostname)"
echo "SLURM_JOB_GPUS=$SLURM_JOB_GPUS"
export CUDA_VISIBLE_DEVICES="MIG-cc67418b-8d94-51c4-88d2-ec45ff0e92c0"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# GradCAM
# python3 gradcam_patient_3d.py \
#     --patient_dir /data/home/sk1022/MajorProject_sCTGen/awesome-synthetic-CT-generation/bhuvanesh/datasets/bhuvaneshsingla/synthrad2025-hn/versions/Experiment \
#     --use_preproc_files \
#     --save_gradcam \
#     --save_overlay \
#     --output_dir gradcam_outputs \


# GradCAM++
python3 gradcam_pp_patient_3d.py \
    --patient_dir /data/home/sk1022/MajorProject_sCTGen/awesome-synthetic-CT-generation/bhuvanesh/datasets/bhuvaneshsingla/synthrad2025-hn/versions/Experiment \
    --use_preproc_files \
    --save_gradcam \
    --save_overlay \
    --output_dir gradcam_pp_outputs \