from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.guardalign_ot import (
    compute_cost_matrix,
    compute_patch_scores,
    extract_clip_embeddings,
    sinkhorn_transport,
)

def save_overlay(image: Image.Image, patch_scores: torch.Tensor, grid_size: int, output_path: Path) -> None:
    image_np = np.array(image)
    heat = patch_scores.detach().cpu().numpy().reshape(grid_size, grid_size)
    heat = heat - heat.min()
    if heat.max() > 0:
        heat = heat / heat.max()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image_np)
    overlay = ax.imshow(
        heat,
        cmap="jet",
        alpha=0.45,
        extent=(0, image_np.shape[1], image_np.shape[0], 0),
        interpolation="bilinear",
    )
    ax.set_title("Sinkhorn Patch Score Heatmap")
    ax.axis("off")
    cbar = fig.colorbar(overlay, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized patch score")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract CLIP embeddings and test Sinkhorn OT")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--text", action="append", required=True)
    parser.add_argument("--model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--num-iters", type=int, default=300)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/clip_demo"))
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    image_embeds, text_embeds, grid_size = extract_clip_embeddings(
        image=image,
        texts=args.text,
        model_name=args.model,
        device=args.device,
    )

    cost = compute_cost_matrix(image_embeds, text_embeds)
    plan = sinkhorn_transport(cost, epsilon=args.epsilon, num_iters=args.num_iters)
    patch_scores = compute_patch_scores(plan, cost)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "patch_scores_heatmap.png"
    save_overlay(image, patch_scores, grid_size, output_path)

    topk = min(5, patch_scores.numel())
    top_scores, top_indices = torch.topk(patch_scores, k=topk)
    prompt_summary = " | ".join(args.text)

    print("device:", args.device)
    print("image_embeds shape:", tuple(image_embeds.shape))
    print("text_embeds shape:", tuple(text_embeds.shape))
    print("cost_matrix shape:", tuple(cost.shape))
    print("transport_plan shape:", tuple(plan.shape))
    print("texts:", prompt_summary)
    print("epsilon:", args.epsilon)
    print("num_iters:", args.num_iters)
    print("top patch indices:", top_indices.tolist())
    print("top patch scores:", top_scores.tolist())
    print("saved to:", output_path)


if __name__ == "__main__":
    main()
