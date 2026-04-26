import pytest
import torch

from src.guardalign_ot import (
    compute_cost_matrix,
    compute_patch_scores,
    guardalign_op_score,
    sinkhorn_transport,
)


def test_cost_matrix_shape():
    image = torch.randn(4, 8)
    text = torch.randn(3, 8)
    cost = compute_cost_matrix(image, text)
    assert cost.shape == (4, 3)


def test_cost_matrix_in_valid_range():
    image = torch.randn(5, 6)
    text = torch.randn(2, 6)
    cost = compute_cost_matrix(image, text)
    assert torch.isfinite(cost).all()
    assert torch.all(cost >= 0.0)
    assert torch.all(cost <= 2.0)


def test_transport_plan_non_negative():
    cost = torch.rand(4, 3)
    plan = sinkhorn_transport(cost, epsilon=0.05, num_iters=300)
    assert torch.all(plan >= 0.0)


def test_transport_plan_matches_uniform_marginals():
    m, n = 4, 3
    cost = torch.rand(m, n)
    plan = sinkhorn_transport(cost, epsilon=0.05, num_iters=500, tol=1e-8)
    expected_rows = torch.full((m,), 1.0 / m, dtype=plan.dtype, device=plan.device)
    expected_cols = torch.full((n,), 1.0 / n, dtype=plan.dtype, device=plan.device)
    assert torch.allclose(plan.sum(dim=1), expected_rows, atol=1e-4)
    assert torch.allclose(plan.sum(dim=0), expected_cols, atol=1e-4)


def test_identity_case_prefers_diagonal():
    emb = torch.eye(4)
    cost = compute_cost_matrix(emb, emb)
    plan = sinkhorn_transport(cost, epsilon=0.01, num_iters=500, tol=1e-9)
    assert torch.all(torch.diag(plan) > plan.mean(dim=1))


def test_permutation_case_recovers_matching():
    image = torch.eye(4)
    perm = torch.tensor([2, 0, 3, 1])
    text = image[perm]
    cost = compute_cost_matrix(image, text)
    plan = sinkhorn_transport(cost, epsilon=0.01, num_iters=500, tol=1e-9)
    expected = torch.argsort(perm)
    assert torch.equal(plan.argmax(dim=1).cpu(), expected.cpu())


def test_patch_scores_shape():
    cost = torch.rand(6, 5)
    plan = sinkhorn_transport(cost, epsilon=0.05, num_iters=200)
    scores = compute_patch_scores(plan, cost)
    assert scores.shape == (6,)
    assert torch.isfinite(scores).all()


def test_end_to_end_output_keys():
    image = torch.randn(3, 4)
    text = torch.randn(2, 4)
    result = guardalign_op_score(image, text)
    assert set(result.keys()) == {"cost_matrix", "transport_plan", "patch_scores", "ot_cost"}
    assert result["cost_matrix"].shape == (3, 2)
    assert result["transport_plan"].shape == (3, 2)
    assert result["patch_scores"].shape == (3,)


def test_feature_dim_mismatch_raises():
    image = torch.randn(3, 4)
    text = torch.randn(2, 5)
    with pytest.raises(ValueError, match="same feature dimension"):
        compute_cost_matrix(image, text)


def test_empty_cost_matrix_raises():
    with pytest.raises(ValueError, match="non-empty"):
        sinkhorn_transport(torch.empty(0, 3))
