import torch


def resolve_torch_device(requested_device: str | None) -> torch.device:
    """Resolve an optional configured CPU or CUDA device with availability checks."""
    if requested_device in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        device = torch.device(requested_device)
    except RuntimeError as exc:
        raise ValueError(f"Invalid device value: {requested_device!r}") from exc
    if device.type == "cpu":
        return device
    if device.type != "cuda":
        raise ValueError(f"Device must be 'auto', 'cpu', 'cuda', or 'cuda:<index>', got {requested_device!r}")
    if not torch.cuda.is_available():
        raise ValueError(f"CUDA requested with device={requested_device!r}, but CUDA is unavailable")
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise ValueError(
            f"CUDA device index is unavailable: requested={requested_device!r}, "
            f"device_count={torch.cuda.device_count()}"
        )
    return device
