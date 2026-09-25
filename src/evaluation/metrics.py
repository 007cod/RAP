import torch


def valid_mask(
    labels: torch.Tensor,
    null_val=0.0,
    preds: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build the shared label mask used by all evaluation metrics.

    This intentionally follows the project's original masked metric definition:
    validity is determined only by ``labels`` and ``null_val``. NaN losses are
    converted to zero after masking.
    """
    null_tensor = torch.as_tensor(null_val, dtype=labels.dtype, device=labels.device)
    if torch.isnan(null_tensor):
        valid = ~torch.isnan(labels)
    else:
        valid = labels != null_tensor
    if mask is not None:
        valid = valid & mask.to(device=labels.device, dtype=torch.bool)
    return valid


def _masked_loss(loss: torch.Tensor, labels: torch.Tensor, null_val, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    valid = valid_mask(labels, null_val=null_val, mask=mask)
    normalized_mask = valid.to(dtype=loss.dtype)
    normalized_mask = normalized_mask / normalized_mask.mean()
    normalized_mask = torch.where(
        torch.isnan(normalized_mask),
        torch.zeros_like(normalized_mask),
        normalized_mask,
    )
    weighted_loss = loss * normalized_mask
    weighted_loss = torch.where(
        torch.isnan(weighted_loss),
        torch.zeros_like(weighted_loss),
        weighted_loss,
    )
    return weighted_loss, valid


def masked_error_sums(
    preds: torch.Tensor,
    labels: torch.Tensor,
    null_val=0.0,
    dim=None,
    mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Aggregate losses using the same mask semantics as the scalar metrics."""
    valid = valid_mask(labels, null_val=null_val, mask=mask)
    binary_mask = valid.to(dtype=preds.dtype)
    error = preds - labels

    def masked_sum(loss: torch.Tensor) -> torch.Tensor:
        value = loss * binary_mask
        value = torch.where(torch.isnan(value), torch.zeros_like(value), value)
        return value.sum(dim=dim)

    return {
        "absolute_error": masked_sum(error.abs()),
        "squared_error": masked_sum(error.square()),
        "absolute_percentage_error": masked_sum(error.abs() / labels),
        "error": masked_sum(error),
        "count": valid.sum(dim=dim),
    }


def metric_tensors_from_sums(sums: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Calculate all metric values from sums produced by ``masked_error_sums``."""
    denominator = sums["count"].to(dtype=sums["absolute_error"].dtype).clamp_min(1.0)
    mse = sums["squared_error"] / denominator
    return {
        "mae": sums["absolute_error"] / denominator,
        "mape": sums["absolute_percentage_error"] / denominator,
        "rmse": torch.sqrt(mse),
        "bias": sums["error"] / denominator,
        "count": sums["count"],
    }


def _metric_tensors(
    preds: torch.Tensor,
    labels: torch.Tensor,
    null_val,
    mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    return metric_tensors_from_sums(masked_error_sums(preds, labels, null_val, mask=mask))


def masked_mse(
    preds: torch.Tensor,
    labels: torch.Tensor,
    null_val=0.0,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    loss, _ = _masked_loss((preds - labels) ** 2, labels, null_val, mask)
    return torch.mean(loss)


def masked_rmse(
    preds: torch.Tensor,
    labels: torch.Tensor,
    null_val=0.0,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val, mask=mask))


def masked_mae(
    preds: torch.Tensor,
    labels: torch.Tensor,
    null_val=0.0,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    loss, _ = _masked_loss(torch.abs(preds - labels), labels, null_val, mask)
    return torch.mean(loss)


def masked_mape(
    preds: torch.Tensor,
    labels: torch.Tensor,
    null_val=0.0,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    loss, _ = _masked_loss(torch.abs(preds - labels) / labels, labels, null_val, mask)
    return torch.mean(loss)


def compute_all_metrics(
    preds: torch.Tensor,
    labels: torch.Tensor,
    null_val=0.0,
    mask: torch.Tensor | None = None,
) -> tuple[float, float, float]:
    mae = masked_mae(preds, labels, null_val, mask).item()
    mape = masked_mape(preds, labels, null_val, mask).item()
    rmse = masked_rmse(preds, labels, null_val, mask).item()
    return mae, mape, rmse


def compute_metric_dict(
    preds: torch.Tensor,
    labels: torch.Tensor,
    null_val=0.0,
    mask: torch.Tensor | None = None,
) -> dict:
    metrics = _metric_tensors(preds, labels, null_val, mask)
    mape = metrics["mape"].item()
    return {
        "mae": metrics["mae"].item(),
        "mape": mape,
        "mape_percent": mape * 100.0,
        "rmse": metrics["rmse"].item(),
        "bias": metrics["bias"].item(),
        "count": int(metrics["count"].item()),
    }
