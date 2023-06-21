import torch


def right_minus_left(
        x: torch.Tensor,
        reduce_boundary: int = 1) -> torch.Tensor:
    """
    Compute differences (right - left) and stores to left.
    """
    grad = torch.zeros_like(x)
    grad[..., :-1] = x[..., 1:] - x[..., :-1]
    if reduce_boundary > 0:
        grad[..., -reduce_boundary:] = 0
    return grad


def left_minus_right(
        x: torch.Tensor,
        reduce_boundary: int = 1) -> torch.Tensor:
    """
    Compute differences (left - right) and stores to right.
    """
    grad = torch.zeros_like(x)
    grad[..., 1:] = x[..., :-1] - x[..., 1:]
    if reduce_boundary > 0:
        grad[..., :reduce_boundary] = 0
    return grad
