import torch


def encode_poisson(
    data: torch.Tensor, num_timesteps: int, scale: float = 0.25
) -> torch.Tensor:
    num_batches, frame_x, frame_y = data.shape
    data = scale * data.view((num_batches, 1, -1)).repeat((1, num_timesteps, 1))
    return (torch.rand(data.shape) < (data * scale)).float()


def encode_class(
    class_idx: torch.Tensor, num_classes: int, num_timesteps: int
) -> torch.Tensor:
    num_batches = class_idx.numel()
    target = torch.nn.functional.one_hot(class_idx, num_classes=num_classes)
    return target.view((num_batches, 1, -1)).repeat((1, num_timesteps, 1)).float()
