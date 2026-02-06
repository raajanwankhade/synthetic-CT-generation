from __future__ import annotations

from typing import Optional, Tuple, List, Union

from pydantic import BaseModel, Field, ConfigDict


class PathsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_dir: str
    root: str


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    modality: str
    patch_size: Optional[Tuple[int, ...]] = None
    anatomy: Optional[str] = None
    num_adjacent_slices: Union[int, str] = 0
    cache_size: int = 1
    foreground_threshold: float = 0.3
    val_split: float = 0.2


class DataloaderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    batch_size: int
    num_workers_train: int
    num_workers_val: int
    prefetch_factor: int


class PerfConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    deterministic: bool = False
    cudnn_benchmark: bool = True
    allow_tf32: bool = True
    amp_dtype: str = "bf16"
    channels_last: bool = True
    torch_compile: bool = False
    torch_compile_mode: str = "reduce-overhead"
    ssim_downsample: int = 2


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start: int = 1
    epochs: int
    lr: float
    lambda_ssim: float
    lambda_sem: float = 0.1
    scheduler_patience: int
    grad_clip: Optional[float] = None
    sanity_steps: int = 1
    resume_path: Optional[str] = None


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    log_images_every: int = 5
    save_vis_every: int = 5
    wandb_mode: str = "online"
    wandb_project: str
    wandb_run_name: str
    id_wandb: Optional[str] = None
    resume_wandb: Optional[str] = None
    loss_plot_name: str = "Loss Plot.png"
    visualisation_fig: str = "Batch_Output.png"


class CheckpointConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ckpt_dir: str = "checkpoints"
    ckpt_filename: str = "best.pth"
    teacher_filename: str = "frozen_teacher.pth"


class Model3DConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    in_ch: int
    out_ch: int
    base_ch: int
    stage_channels: List[int]
    blocks_per_stage: List[int]
    convs_per_block: int
    negative_slope: float = 0.01


class TeacherModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    spatial_dims: int = 3
    in_channels: int = 1
    out_channels: int = 1
    channels: Tuple[int, ...] = (32, 64, 128, 256, 512)
    strides: Tuple[int, ...] = (2, 2, 2, 2)
    num_res_units: int = 2
    norm: str = "instance"


class InferenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_path: str
    input_path: str
    modality: str
    patch_size: Tuple[int, ...]
    stride: Optional[Tuple[int, ...]] = None
    output_dir: str
    device: str = "cuda"
    num_adjacent_slices: Union[int, str] = 0
    channels_last: bool = True


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    paths: PathsConfig
    data: DataConfig
    dataloader: DataloaderConfig
    perf: PerfConfig
    training: TrainingConfig
    logging: LoggingConfig
    checkpoint: CheckpointConfig
    model3d: Model3DConfig
    teacher_model: TeacherModelConfig
    inference: InferenceConfig
