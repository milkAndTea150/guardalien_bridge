from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torch.nn.functional as F


Tensor = torch.Tensor


def _validate_embeddings(image_embeds: torch.Tensor, text_embeds: torch.Tensor) -> None:
    if image_embeds.ndim != 2 or text_embeds.ndim != 2:
        raise ValueError("image_embeds and text_embeds must both be 2D tensors")
    if image_embeds.shape[0] == 0 or text_embeds.shape[0] == 0:
        raise ValueError("image_embeds and text_embeds must be non-empty")
    if image_embeds.shape[1] != text_embeds.shape[1]:
        raise ValueError("image_embeds and text_embeds must have the same feature dimension")


def compute_cost_matrix(
    image_embeds: Tensor,
    text_embeds: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """Compute cosine-distance costs C(i,j)=1-cos(x_i, z_j)."""
    _validate_embeddings(image_embeds, text_embeds)
    image_norm = F.normalize(image_embeds.float(), p=2, dim=1, eps=eps)
    text_norm = F.normalize(text_embeds.float(), p=2, dim=1, eps=eps)
    cosine_sim = image_norm @ text_norm.T
    return (1.0 - cosine_sim).clamp(min=0.0, max=2.0)


def sinkhorn_transport(
    cost_matrix: Tensor,
    epsilon: float = 0.05,
    num_iters: int = 100,
    tol: float = 1e-6,
) -> Tensor:
    """Compute an entropic OT plan with uniform marginals."""
    if cost_matrix.ndim != 2:
        raise ValueError("cost_matrix must be a 2D tensor")
    if cost_matrix.numel() == 0:
        raise ValueError("cost_matrix must be non-empty")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if num_iters <= 0:
        raise ValueError("num_iters must be positive")

    cost = cost_matrix.float()
    m, n = cost.shape
    device = cost.device
    dtype = cost.dtype

    a = torch.full((m,), 1.0 / m, dtype=dtype, device=device)
    b = torch.full((n,), 1.0 / n, dtype=dtype, device=device)

    kernel = torch.exp(-cost / epsilon).clamp_min(1e-12)
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
    """Aggregate row-wise transported cost into one score per image patch."""
    if transport_plan.shape != cost_matrix.shape:
        raise ValueError("transport_plan and cost_matrix must have the same shape")
    return (transport_plan * cost_matrix).sum(dim=1)


def guardalign_op_score(
    image_embeds: Tensor,
    text_embeds: Tensor,
    epsilon: float = 0.05,
    num_iters: int = 100,
) -> Dict[str, Tensor]:
    """Run the full OP scoring pipeline on precomputed embeddings."""
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


def extract_clip_embeddings(
    image,
    texts: Sequence[str],
    model_name: str = "openai/clip-vit-base-patch32",
    device: str | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Extract CLIP image-patch and text embeddings in the same feature space."""
    try:
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:
        raise ImportError(
            "transformers is required for extract_clip_embeddings. "
            "Install dependencies from requirements.txt."
        ) from exc

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()

    with torch.no_grad():
        image_inputs = processor(images=image, return_tensors="pt")
        text_inputs = processor(text=list(texts), return_tensors="pt", padding=True, truncation=True)
        image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

        vision_outputs = model.vision_model(**image_inputs)
        text_outputs = model.text_model(**text_inputs)

        patch_states = vision_outputs.last_hidden_state[:, 1:, :]
        image_embeds = model.visual_projection(patch_states.squeeze(0))
        image_embeds = F.normalize(image_embeds.float(), p=2, dim=-1)

        text_embeds = model.text_projection(text_outputs.pooler_output)
        text_embeds = F.normalize(text_embeds.float(), p=2, dim=-1)

    patch_count = image_embeds.shape[0]
    grid_size = int(patch_count**0.5)
    if grid_size * grid_size != patch_count:
        raise ValueError(f"Patch count {patch_count} does not form a square grid")

    return image_embeds, text_embeds, grid_size
