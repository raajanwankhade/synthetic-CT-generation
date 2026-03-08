cfg = {
    "run_dir": "/data/home/sk1022/MajorProject_sCTGen/awesome-synthetic-CT-generation/bhuvanesh/ablations/Unet3D/HN/preproc_l1_ssim_dice/run1",
    # Data
    "root": '/data/home/sk1022/MajorProject_sCTGen/awesome-synthetic-CT-generation/bhuvanesh/datasets/bhuvaneshsingla/synthrad2025-hn/versions/3/Task1',
    "modality": "mr",

    "patch_size": (32, 192, 192),
    "anatomy": "HN",  # "AB", "HN", or "TH" for 3D
    "num_adjacent_slices": "3d",  # 3D patches
    "cache_size": 2,
    "foreground_threshold": 0.3,
    "val_split": 0.2,

    "use_preproc_files": True,
    "use_pruned_data": False,

    # Dataloaders
    "batch_size": 8,
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
    
    # torch.compile (PyTorch 2.0+ only)
    # Disabled: Triton 3.1.0 (required by PyTorch 2.5.1) can't handle CUDA 12.8+ driver
    "torch_compile": False,
    "torch_compile_mode": "default",  # "default", "reduce-overhead", or "max-autotune"
    "torch_compile_backend": "inductor",  # "inductor" (optimizing) | "aot_eager" (debug only)

    # SSIM cost control
    "ssim_downsample": 2,  # 1 = full-res, 2 = avg_pool2d by 2

    # Logging/I/O control
    "log_images_every": 5,  # epochs; set 0 to disable W&B image logging
    "save_vis_every": 5,    # epochs; set 0 to disable PNG writes

    # Training
    "start":1,
    "epochs": 2,
    "lr": 1e-3,

    "lambda_seg": 100,
    "lambda_ssim": 1000,
    "lambda_l1": 1.0,
    "lambda_afp": 0.1,

    "scheduler_patience": 10,
    "grad_clip": None,
    "sanity_steps": 1,
    "resume_path": None,

    # Teacher Training
    "teacher_filename": "frozen_teacher.pth",
    "teacher_patience": 10,  # early stopping patience for teacher validation loss
    "teacher_cfg": {
        "in_ch": 1,
        "out_ch": 1,
        "base_ch": 32,
        "stage_channels": [32, 64, 128, 256],
        "num_res_units": 2,
        "norm": "instance",
    },

    
    # Checkpoint
    "ckpt_dir": "checkpoints",
    "ckpt_filename": "best.pth",

    # W&B
    "wandb_mode": "disabled",  # "online", "offline", or "disabled"
    "wandb_project": "MR_CT_Translation_UNET",
    "wandb_run_group": "ablations",
    "wandb_run_name": "[abl]_UNet3D_HN_pruned_preproc_L1_SSIM_run2",
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

    # Model (2D)
    "model2d_cfg": {
        "in_ch": 1,
        "out_ch": 1,
        "base_ch": 32,
        "stage_channels": [32, 64, 128, 256],
        "blocks_per_stage": [2, 2, 2, 2],
        "convs_per_block": 2,
        "negative_slope": 0.01,
    },

    # AFP Config
    "nnunet_results_dir": "/data/home/sk1022/.totalsegmentator/nnunet/results",
    "val_overlap": 0.5,
    "val_sliding_window_batch": 4,
    "afp_batch_train": 4,
    "afp_batch_val": 1,

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


}
