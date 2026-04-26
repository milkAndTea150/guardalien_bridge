"""GuardAlign-style entropic optimal transport on synthetic embeddings."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


Tensor = torch.Tensor


def _validate_embeddings(image_embeds: Tensor, text_embeds: Tensor) -> None:
    if image_embeds.ndim != 2 or text_embeds.ndim != 2:
        raise ValueError("image_embeds and text_embeds must both be 2D tensors")
    if image_embeds.shape[1] != text_embeds.shape[1]:
        raise ValueError(
            "image_embeds and text_embeds must have the same feature dimension"
        )
    if image_embeds.shape[0] == 0 or text_embeds.shape[0] == 0:
        raise ValueError("image_embeds and text_embeds must be non-empty")


def compute_cost_matrix(
    image_embeds: Tensor,
    text_embeds: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """Compute cosine distance cost matrix with values in approximately [0, 2]."""
    _validate_embeddings(image_embeds, text_embeds)

    image_unit = F.normalize(image_embeds.float(), p=2, dim=1, eps=eps)
    text_unit = F.normalize(text_embeds.float(), p=2, dim=1, eps=eps)
    cosine_sim = image_unit @ text_unit.T
    cost_matrix = 1.0 - cosine_sim
    return cost_matrix.clamp(min=0.0, max=2.0)


def sinkhorn_transport(
    cost_matrix: Tensor,
    epsilon: float = 0.05,
    num_iters: int = 100,
    tol: float = 1e-6,
) -> Tensor:
    """Compute a dense entropic OT plan under uniform marginals."""
    if cost_matrix.ndim != 2:
        raise ValueError("cost_matrix must be a 2D tensor")
    if cost_matrix.numel() == 0:
        raise ValueError("cost_matrix must be non-empty")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if num_iters <= 0:
        raise ValueError("num_iters must be positive")

    m, n = cost_matrix.shape
    device = cost_matrix.device
    dtype = cost_matrix.dtype

    a = torch.full((m,), 1.0 / m, dtype=dtype, device=device)
    b = torch.full((n,), 1.0 / n, dtype=dtype, device=device)

    kernel = torch.exp(-cost_matrix / epsilon).clamp_min(1e-12)
    u = torch.ones_like(a)
    v = torch.ones_like(b)

    for _ in range(num_iters):
        u_prev = u
        kv = kernel @ v
        u = a / kv.clamp_min(1e-12)
        ktu = kernel.T @ u
        v = b / ktu.clamp_min(1e-12)

        if torch.max(torch.abs(u - u_prev)).item() < tol:
            break

    transport_plan = u[:, None] * kernel * v[None, :]
    return transport_plan.clamp_min(0.0)


def compute_patch_scores(transport_plan: Tensor, cost_matrix: Tensor) -> Tensor:
    """Aggregate per-patch suspiciousness scores from transport-weighted costs."""
    if transport_plan.shape != cost_matrix.shape:
        raise ValueError("transport_plan and cost_matrix must have the same shape")
    return (transport_plan * cost_matrix).sum(dim=1)


def guardalign_op_score(
    image_embeds: Tensor,
    text_embeds: Tensor,
    epsilon: float = 0.05,
    num_iters: int = 100,
) -> Dict[str, Tensor]:
    """End-to-end GuardAlign-style OT scoring."""
    cost_matrix = compute_cost_matrix(image_embeds, text_embeds)
    transport_plan = sinkhorn_transport(
        cost_matrix,
        epsilon=epsilon,
        num_iters=num_iters,
    )
    patch_scores = compute_patch_scores(transport_plan, cost_matrix)
    return {
        "cost_matrix": cost_matrix,
        "transport_plan": transport_plan,
        "patch_scores": patch_scores,
    }
