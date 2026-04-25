import pytest
import torch

from src.guardalign_op import (
    compute_cost_matrix,
    sinkhorn_transport,
    compute_patch_scores,
    guardalign_op_score,
)


def test_cost_matrix_shape():
    image = torch.randn(4, 8)
    text = torch.randn(3, 8)
    cost = compute_cost_matrix(image, text)
    assert cost.shape == (4, 3)


def test_cost_matrix_is_finite_and_in_range():
    image = torch.randn(4, 8)
    text = torch.randn(3, 8)
    cost = compute_cost_matrix(image, text)
    assert torch.isfinite(cost).all()
    assert torch.all(cost >= 0)
    assert torch.all(cost <= 2.0 + 1e-6)


def test_transport_plan_non_negative():
    cost = torch.rand(4, 3)
    plan = sinkhorn_transport(cost, epsilon=0.05, num_iters=200)
    assert torch.all(plan >= 0)


def test_transport_plan_marginals():
    M, N = 4, 3
    cost = torch.rand(M, N)
    plan = sinkhorn_transport(cost, epsilon=0.05, num_iters=300)
    expected_rows = torch.full((M,), 1.0 / M)
    expected_cols = torch.full((N,), 1.0 / N)
    assert torch.allclose(plan.sum(dim=1), expected_rows, atol=1e-2)
    assert torch.allclose(plan.sum(dim=0), expected_cols, atol=1e-2)


def test_identity_matching_diagonal_mass():
    emb = torch.eye(4)
    cost = compute_cost_matrix(emb, emb)
    plan = sinkhorn_transport(cost, epsilon=0.01, num_iters=300)
    diagonal_mass = torch.diag(plan).sum()
    total_mass = plan.sum()
    assert diagonal_mass / total_mass > 0.7


def test_permutation_matching():
    image = torch.eye(4)
    perm = torch.tensor([2, 0, 3, 1])
    text = image[perm]
    cost = compute_cost_matrix(image, text)
    plan = sinkhorn_transport(cost, epsilon=0.01, num_iters=300)
    matched_text_index = plan.argmax(dim=1)
    expected = torch.argsort(perm)
    assert torch.equal(matched_text_index.cpu(), expected.cpu())


def test_patch_scores_shape():
    M, N = 4, 3
    cost = torch.rand(M, N)
    plan = sinkhorn_transport(cost, epsilon=0.05, num_iters=200)
    patch_scores = compute_patch_scores(plan, cost)
    assert patch_scores.shape == (M,)
    assert torch.isfinite(patch_scores).all()


def test_end_to_end_output_keys():
    image = torch.randn(4, 8)
    text = torch.randn(3, 8)
    result = guardalign_op_score(image, text)
    assert "cost_matrix" in result
    assert "transport_plan" in result
    assert "patch_scores" in result
    assert "global_score" in result


def test_invalid_feature_dim_raises():
    image = torch.randn(4, 8)
    text = torch.randn(3, 7)
    with pytest.raises(ValueError):
        compute_cost_matrix(image, text)
