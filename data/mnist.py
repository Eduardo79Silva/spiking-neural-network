import os
import torch
import torchvision
from torch.utils.data import DataLoader

# Windows: ~/AppData/Local/torch/datasets
# Linux/Mac: ~/.cache/torch/datasets
_DEFAULT_CACHE_DIR = os.path.join(
    os.path.expanduser("~"),
    "AppData",
    "Local",
    "torch",
    "datasets" if os.name == "nt" else os.path.join(".cache", "torch", "datasets"),
)

_MNIST_VALIDATION_FILE = os.path.join("MNIST", "raw", "train-images-idx3-ubyte")


def _resolve_cache_dir(cache_dir: str | None) -> str:
    """Return cache_dir if provided, else the platform default."""
    return cache_dir if cache_dir is not None else _DEFAULT_CACHE_DIR


def _dataset_cached(cache_dir: str) -> bool:
    """Return True only if the raw training file is present on disk."""
    return os.path.exists(os.path.join(cache_dir, _MNIST_VALIDATION_FILE))


def _make_transform() -> torchvision.transforms.Compose:
    return torchvision.transforms.Compose([torchvision.transforms.ToTensor()])


def load_mnist(
    batch_size: int = 64,
    cache_dir: str | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """
    Load MNIST train and test DataLoaders.

    Checks the platform cache directory before attempting a download.
    Falls back to the current working directory if the cache path is
    not writable.

    Args:
        batch_size:   Samples per batch for both loaders.
        cache_dir:    Override the default cache location.
        num_workers:  DataLoader worker processes (0 = main process only).
        pin_memory:   Pin memory for faster GPU transfers.

    Returns:
        (train_loader, test_loader)
    """
    root = _resolve_cache_dir(cache_dir)
    cached = _dataset_cached(root)

    if not cached:
        os.makedirs(root, exist_ok=True)
        if not os.access(root, os.W_OK):
            root = "."

    transform = _make_transform()

    train_data = torchvision.datasets.MNIST(
        root=root,
        train=True,
        download=not cached,
        transform=transform,
    )

    test_data = torchvision.datasets.MNIST(
        root=root,
        train=False,
        download=not cached,
        transform=transform,
    )

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    train_loader = DataLoader(train_data, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_data, shuffle=False, **loader_kwargs)

    return train_loader, test_loader
