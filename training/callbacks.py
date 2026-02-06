import torch

SHUTDOWN_REQUESTED = False


def enforce_channels_last(x: torch.Tensor, y: torch.Tensor, m: torch.Tensor, device: torch.device):
    if device.type != "cuda":
        return x, y, m
    if x.dim() == 5:  # 3D
        return (
            x.contiguous(memory_format=torch.channels_last_3d),
            y.contiguous(memory_format=torch.channels_last_3d),
            m.contiguous(memory_format=torch.channels_last_3d),
        )
    return (
        x.contiguous(memory_format=torch.channels_last),
        y.contiguous(memory_format=torch.channels_last),
        m.contiguous(memory_format=torch.channels_last),
    )
