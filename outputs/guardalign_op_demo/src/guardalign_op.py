from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


_INTENTIONALLY_ROW_NORMALIZE = False


def _validate_embeddings(image_embeds: torch.Tensor, text_embeds: torch.Tensor) -> None:
    if not isinstance(image_embeds, torch.Tensor) or not isinstance(text_embeds, torch.Tensor):
        raise TypeError("image_embeds and text_embeds must be torch.Tensor")
    if image_embeds.ndim != 2 or text_embeds.ndim != 2:
        raise ValueError("image_embeds and text_embeds must have shape [num_items, hidden_dim]")
    if image_embeds.shape[1] != text_embeds.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: {image_embeds.shape[1]} vs {text_embeds.shape[1]}"
        )
    if image_embeds.shape[0] == 0 or text_embeds.shape[0] == 0:
        raise ValueError("image_embeds and text_embeds must contain at least one item")


def compute_cost_matrix(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute C(m,n)=1-cos(x_m,z_n).

    Args:
        image_embeds: Tensor[M, D]
        text_embeds: Tensor[N, D]
        eps: numerical stability epsilon for normalization

    Returns:
        Tensor[M, N] cosine-distance cost matrix.
    """
    _validate_embeddings(image_embeds, text_embeds)
    image_norm = F.normalize(image_embeds.float(), p=2, dim=-1, eps=eps)
    text_norm = F.normalize(text_embeds.float(), p=2, dim=-1, eps=eps)
    cost = 1.0 - image_norm @ text_norm.t()
    return cost.clamp(min=0.0, max=2.0)


def sinkhorn_transport(
    cost_matrix: torch.Tensor,
    epsilon: float = 0.05,
    num_iters: int = 100,
    tol: float = 1e-6,
) -> torch.Tensor:
    """Compute an entropic OT plan with uniform marginals.

    Args:
        cost_matrix: Tensor[M, N]
        epsilon: entropic regularization strength
        num_iters: maximum Sinkhorn iterations
        tol: early-stop tolerance on scaling vectors

    Returns:
        Tensor[M, N] transport plan whose row sums are approximately 1/M
        and column sums are approximately 1/N.
    """
    if cost_matrix.ndim != 2:
        raise ValueError("cost_matrix must have shape [M, N]")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if num_iters <= 0:
        raise ValueError("num_iters must be positive")

    cost = cost_matrix.float()
    M, N = cost.shape
    device = cost.device
    dtype = cost.dtype

    a = torch.full((M,), 1.0 / M, device=device, dtype=dtype)
    b = torch.full((N,), 1.0 / N, device=device, dtype=dtype)

    # Kernel for entropic OT. Clamping avoids exact zeros in extreme cases.
    K = torch.exp(-cost / epsilon).clamp_min(1e-12)
    u = torch.ones_like(a)
    v = torch.ones_like(b)

    for _ in range(num_iters):
        u_prev = u
        u = a / (K @ v).clamp_min(1e-12)
        v = b / (K.t() @ u).clamp_min(1e-12)
        if torch.max(torch.abs(u - u_prev)) < tol:
            break

    plan = u[:, None] * K * v[None, :]

    # This intentionally buggy branch is useful for demonstrating repair.
    # Correct OT with uniform marginals should NOT row-normalize to 1.
    if _INTENTIONALLY_ROW_NORMALIZE:
        plan = plan / plan.sum(dim=1, keepdim=True).clamp_min(1e-12)

    return plan


def compute_patch_scores(transport_plan: torch.Tensor, cost_matrix: torch.Tensor) -> torch.Tensor:
    """Compute patch_score(m)=sum_n T(m,n)*C(m,n)."""
    if transport_plan.shape != cost_matrix.shape:
        raise ValueError("transport_plan and cost_matrix must have the same shape")
    return (transport_plan * cost_matrix).sum(dim=1)


def guardalign_op_score(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    epsilon: float = 0.05,
    num_iters: int = 100,
    tol: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """End-to-end paper-specific OP scoring module.

    This is not intended to replace mature OT libraries. It is a compact,
    inspectable implementation for algorithm-to-code verification.
    """
    cost = compute_cost_matrix(image_embeds, text_embeds)
    plan = sinkhorn_transport(cost, epsilon=epsilon, num_iters=num_iters, tol=tol)
    patch_scores = compute_patch_scores(plan, cost)
    global_score = patch_scores.mean()
    return {
        "cost_matrix": cost,
        "transport_plan": plan,
        "patch_scores": patch_scores,
        "global_score": global_score,
    }
