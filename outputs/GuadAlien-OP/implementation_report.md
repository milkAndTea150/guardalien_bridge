# Implementation Report

## Scope
This project implements a minimal GuardAlign-style optimal transport operator in PyTorch for CPU execution only. It uses cosine distance between synthetic image patch embeddings and synthetic text embeddings, then applies entropic Sinkhorn iterations with uniform marginals.

## What Was Implemented
- `compute_cost_matrix(image_embeds, text_embeds, eps=1e-8)` normalizes both inputs and computes cosine distance.
- `sinkhorn_transport(cost_matrix, epsilon=0.05, num_iters=100, tol=1e-6)` computes a dense entropic transport plan under uniform marginals.
- `compute_patch_scores(transport_plan, cost_matrix)` aggregates transport-weighted costs into per-patch suspiciousness scores.
- `guardalign_op_score(image_embeds, text_embeds, epsilon=0.05, num_iters=100)` provides the end-to-end API.

## Testing
The default pytest suite uses only synthetic CPU tensors and checks:
- importability
- cost matrix shape
- finite and bounded cost values
- non-negative transport entries
- approximate row and column marginal constraints
- identity toy matching
- permutation toy matching
- patch score shape and non-negativity
- end-to-end output keys
- invalid feature dimension handling

## Assumptions
- Uniform source and target weights are used because no learned or external marginals were specified.
- Entropic regularization is exposed as a tunable hyperparameter.
- Input embeddings are provided directly; no real encoder, CLIP dependency, or dataset loader is included.

## Non-Goals
This implementation is a compact research prototype and does not claim to replace POT or GeomLoss. Those libraries are treated only as optional external reference oracles if a user later wants comparison checks.

## Limits
- The Sinkhorn solver is intentionally simple and dense.
- No batching, GPU path, or large-scale optimization is included.
- No real-world unsafe text mining or image embedding extraction is included.
