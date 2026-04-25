# Implementation Report

## Algorithm
GuardAlign-style Sinkhorn Optimal Transport module.

## Goal
Generate a paper-specific OP module from algorithm description and verify it with toy-level mathematical tests.

## Implemented Components
- [x] Embedding shape validation
- [x] Cosine-distance cost matrix: `C(m,n)=1-cos(x_m,z_n)`
- [x] Entropic Sinkhorn transport with uniform marginals
- [x] Patch-level suspiciousness score: `patch_score(m)=sum_n T(m,n) * C(m,n)`
- [x] End-to-end OP score wrapper
- [x] Pytest unit tests
- [x] Toy example script

## Mapping to Algorithm Description
| Paper / Spec Item | Code Location | Status |
|---|---|---|
| `C(m,n)=1-cos(x_m,z_n)` | `src/guardalign_op.py::compute_cost_matrix` | implemented |
| Sinkhorn OT transport plan | `src/guardalign_op.py::sinkhorn_transport` | implemented |
| `patch_score(m)=sum_n T(m,n)C(m,n)` | `src/guardalign_op.py::compute_patch_scores` | implemented |

## Engineering Assumptions
1. Entropic Sinkhorn OT is used because the task asks for Sinkhorn.
2. Uniform marginals are used unless otherwise specified.
3. Tests use synthetic embeddings to isolate algorithm correctness from model dependencies.
4. POT and GeomLoss are optional reference oracles, not required dependencies.

## Current Limitations
1. This verifies toy-level mathematical behavior, not full paper reproduction.
2. This does not yet use CLIP or VLM embeddings.
3. This does not compare against POT by default.
4. Hyperparameters `epsilon` and `num_iters` may need tuning for real data.

## Repair History
- Root cause: transport plan was incorrectly row-normalized to sum to 1 instead of satisfying uniform OT marginals.
- Modified files: `src/guardalign_op.py`
- Fix summary: removed row-normalization branch so Sinkhorn plan preserves row sums 1/M and column sums 1/N.
