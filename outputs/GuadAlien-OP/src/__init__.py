"""Minimal CPU-only GuardAlign-style optimal transport package."""

from .guardalign_op import (
    compute_cost_matrix,
    compute_patch_scores,
    guardalign_op_score,
    sinkhorn_transport,
)

__all__ = [
    "compute_cost_matrix",
    "sinkhorn_transport",
    "compute_patch_scores",
    "guardalign_op_score",
]
