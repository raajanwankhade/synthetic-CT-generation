cfg = {
    "run_dir": "/workspace/runs/wgan_gp/run1",
    # Data
    "root": '/workspace/awesome-synthetic-CT-generation/bhuvanesh/datasets/bhuvaneshsingla/synthrad2025-hn/versions/3/Task1',
    "modality": "mr",

    "patch_size": (32, 128, 128),
    "anatomy": "HN",  # "AB", "HN", or "TH" for 3D
    "num_adjacent_slices": "3d",  # 3D patches
    "cache_size": 8,
    "max_retries": 4,
    "foreground_threshold": 0.3,
    "val_split": 0.2,

    "use_preproc_files": True,
    "use_pruned_data": False,

    # Dataloaders
    "batch_size": 32,
    "num_workers_train": 20,
    "num_workers_val": 4,
    "prefetch_factor": 6,
    "drop_last_train": True,
    "in_order_train": False,
    "in_order_val": True,

    # Perf / determinism (HPC training typically prefers speed)
    "deterministic": False,
    "cudnn_benchmark": True,
    "allow_tf32": True,

    # AMP on H100: prefer BF16
    "amp_dtype": "bf16",  # "bf16" or "fp16"
    "channels_last": True,
    
    # torch.compile (PyTorch 2.0+ only)
    "torch_compile": True,
    "torch_compile_mode": "default",  # "default", "reduce-overhead", or "max-autotune"
    "torch_compile_backend": "inductor",  # "inductor" (optimizing) | "aot_eager" (debug only)
    "compile_generator": True,
    "compile_critic": False,

    # SSIM cost control
    "ssim_downsample": 2,  # 1 = full-res, 2 = avg_pool2d by 2

    # Logging/I/O control
    "log_images_every": 5,  # epochs; set 0 to disable W&B image logging
    "save_vis_every": 5,    # epochs; set 0 to disable PNG writes

    # Training
    "start":1,
    "epochs": 6,
    "lr": 1e-3,

    "lambda_seg": 100,
    "lambda_ssim": 1000,
    "lambda_l1": 1.0,
    "lambda_adv": 0.4,
    "lambda_afp": 0.1,

    # WGAN-GP training config (UNET3D generator + Discriminator3D critic)
    "wgan_gp_cfg": {
        "lambda_adv": 0.4,  # Generator adversarial-loss weight
        "lambda_gp": 10.0,   # Gradient penalty coefficient
        "n_critic": 5,       # Critic updates per generator update
        "gen_lr": 1e-4,
        "disc_lr": 1e-4,
        "disc_betas": (0.0, 0.9),
        "gen_scheduler_patience": 10,
        "disc_scheduler_patience": 10,
        "critic_grad_clip": None,
        "recompute_fake_each_critic_step": False,
        "best_ckpt_filename": "best_wgan_gp.pth",
    },

    "scheduler_patience": 10,
    "grad_clip": None,
    "sanity_steps": 1,
    "resume_path": None,

    # Alternating TS training
    "TS_epochs": 1,
    "regular_epochs": 2,
    "ts_seg_classes": "all",  # Mode selector: "binary", "all", or list indices.
                                # Examples: "binary", "all", or [1, 3, 5]
                                # Class indices are TotalSegmentator labels in [1, 104].
                                # For "all", class-count is inferred automatically.
    "surrogate_weight": 0.5,  # Weight on surrogate Dice in total loss.
                               # total_loss += surrogate_weight * s_loss
    "surrogate_lr": 1e-3,  # Learning rate for surrogate head optimizer.
                            # Independent of main model LR schedule.
                            # Fixed — not stepped by the scheduler.
    "persist_ts_cache": True,  # Save pseudo-label volumes to disk.
                                # Enables resume across training interruptions.
                                # Files saved to run_dir/ts_cache/

    # TS dataloader and I/O controls
    "ts_fraction": 0.2,
    "ts_batch_size": 8,
    "ts_patch_size": (32, 192, 192),
    "ts_stride": (32, 192, 192),
    "ts_num_workers": 0,
    "ts_cache_size": 2,
    "ts_use_preproc_files": False,  # False => TS loader uses mr.mha/ct.mha (non-preproc files)
    "ts_randomize_each_cycle": True,  # Re-sample TS patient subset each TS cycle (seed + cycle_idx)
    "ts_min_dice_for_cache": 0.0,  # Minimum TS Dice needed to cache pseudo labels.
                                # 0.0 disables gating so cache populates from early cycles.
    "ts_use_ramdisk": True,
    "ts_tmp_dir": None,  # If set, overrides ramdisk/default temp path for TS NIfTI files
    "keep_ts_inputs": False,
    "binary_teacher_path": None,

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
    "wandb_mode": "online",  # "online", "offline", or "disabled"
    "wandb_project": "MR_CT_Translation_UNET",
    "wandb_run_group": "WGAN-GP UNET3D",  # e.g. "Ablation_Study_1"
    "wandb_run_name": "run1",  # e.g. "Unet3D_HN_ssim_l1"
    "id_wandb":None,
    "resume_wandb":None, #allow


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

    # Discriminator (3D)
    "discriminator3d_cfg": {
        "in_ch": 1,
        "base_ch": 16,
        "stage_channels": [16, 32, 64, 128],
        "blocks_per_stage": [2, 2, 2, 2],
        "convs_per_block": 2,
        "use_residual": True,
        "negative_slope": 0.2,
        "norm": "instance",  # "instance" or "batch"
        "pool_logits": False,
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
