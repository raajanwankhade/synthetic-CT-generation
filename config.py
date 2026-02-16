import os

cfg = {
    "run_dir": "/data/home/sk1022/MajorProject_sCTGen/awesome-synthetic-CT-generation/bhuvanesh/runs/Unet3dAB/test_run",
    # Data
    "root": '/data/home/sk1022/MajorProject_sCTGen/awesome-synthetic-CT-generation/bhuvanesh/datasets/bhuvaneshsingla/synthrad2025-ab/versions/2/versions/1/Task1',
    "modality": "mr",
    "patch_size": (32, 192, 224),
    "anatomy": "AB",  # "AB", "HN", or "TH" for 3D
    "num_adjacent_slices": "3d",  # 3D patches
    "cache_size": 2,
    "foreground_threshold": 0.3,
    "val_split": 0.2,

    # # Segmentation data
    # "Segmentation_Dir": "/data/home/sk1022/MajorProject_sCTGen/awesome-synthetic-CT-generation/bhuvanesh/datasets/segmentation_dataset/Task1/AB_segmentations",
    # "CT_DIR": "/data/home/sk1022/MajorProject_sCTGen/awesome-synthetic-CT-generation/bhuvanesh/datasets/bhuvaneshsingla/synthrad2025-ab/versions/2/versions/1/Task1",
    # "seg_ct_filename": "ct_preproc.mha",
    # "seg_label_filename": "segmentation.mha",
    # "output_dir": "/data/home/sk1022/MajorProject_sCTGen/awesome-synthetic-CT-generation/bhuvanesh/runs/segmentationAB_unet3d",

    # "teacher_filepath": "/data/home/sk1022/MajorProject_sCTGen/awesome-synthetic-CT-generation/bhuvanesh/runs/segmentationAB_unet3d/fold_5/seg_best.pth",
    # Dataloaders
    "batch_size": 4,
    "num_workers_train": 4,
    "num_workers_val": 4,
    "prefetch_factor": 4,

    # Perf / determinism (HPC training typically prefers speed)
    "deterministic": False,
    "cudnn_benchmark": True,
    "allow_tf32": True,

    # AMP on H100: prefer BF16
    "amp_dtype": "bf16",  # "bf16" or "fp16"
    "channels_last": True,
    "torch_compile": False,
    "torch_compile_mode": "reduce-overhead",  # or "max-autotune"

    # SSIM cost control
    "ssim_downsample": 2,  # 1 = full-res, 2 = avg_pool2d by 2

    # Logging/I/O control
    "log_images_every": 5,  # epochs; set 0 to disable W&B image logging
    "save_vis_every": 5,    # epochs; set 0 to disable PNG writes

    # Training
    "start":1,
    "epochs": 1,
    "lr": 1e-2,
    "lambda_ssim": 100.0,
    "lambda_seg": 0.1,
    "scheduler_patience": 10,
    "grad_clip": None,
    "sanity_steps": 1,
    "resume_path": None,

    # # Segmentation training
    # "seg_num_classes": 9,
    # "seg_epochs": 200,
    # "seg_class_weights": None,
    # "seg_foreground_fraction": 0.6,
    # "seg_max_retries": 20,
    # "seg_steps_per_epoch": None,
    # "seg_rotation_prob": 0.5,
    # "seg_scale_prob": 0.5,
    # "seg_scale_range": (0.9, 1.1),
    # "seg_noise_prob": 0.3,
    # "seg_noise_std": 0.02,
    # "seg_early_stop_patience": 10,
    # "seg_k_folds": 5,
    # "seg_kfold_seed": 42,
    # "seg_learning_rate": 1e-3,

    "ANATOMY_CLASSES": { ##Anatomy based classes to use for segmentation training
        "AB": [
                2, # kidney right
                3, # kidney left
                5, # liver
                6, # stomach
                *range(10, 14+1), #lungs
                *range(26, 50+1), #vertebrae
                51, #heart
                79, # spinal cord
                *range(92, 115+1), # ribs
                116 #sternum
            ],
            "HN": [
                15, # esophagus
                16, # trachea
                17, # thyroid
                *range(26, 50+1), #vertebrae
                79, #spinal cord
                90, # brain
                91, # skull
            ],
            "TH": [
                2, # kidney right
                3, # kidney left
                5, # liver
                6, # stomach
                *range(10, 14+1), #lungs
                *range(26, 50+1), #vertebrae
                51, #heart
                79, # spinal cord
                *range(92, 115+1), # ribs
                116 #sternum
            ]
       
    },
    # Checkpoint
    "ckpt_dir": "checkpoints",
    "ckpt_filename": "best.pth",
    # "teacher_filename" : "frozen_teacher.pth",

    # W&B
    "wandb_mode": "online",  # "online", "offline", or "disabled"
    "wandb_project": "test_runs",
    "wandb_run_name": "Segmentation_Student_UNET3D_AB",
    "id_wandb":None,
    "resume_wandb":None,

    #Figure save locations
    "Loss_Plot_name": "Loss Plot.png",
    "Visualisation_Fig": "Batch_Output.png",

    # Model (3D)
    "model3d_cfg": {
        "in_ch": 1,
        "out_ch": 1,
        "base_ch": 32,
        "stage_channels": [32, 64, 128, 256],
        "blocks_per_stage": [2, 2, 2, 2],
        "convs_per_block": 2,
        "negative_slope": 0.01,
    },

    # "seg_model3d_cfg": {
    #     "in_ch": 1,
    #     "out_ch": 1,  # This will be overridden in code based on anatomy
    #     "base_ch": 32,
    #     "stage_channels": [32, 64, 128, 256],
    #     "blocks_per_stage": [2, 2, 2, 2],
    #     "convs_per_block": 2,
    #     "negative_slope": 0.01,
    # },
}
