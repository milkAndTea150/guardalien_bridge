import torch
import pytest

from src.guardalign_op import (
    compute_cost_matrix,
    compute_patch_scores,
    guardalign_op_score,
    sinkhorn_transport,
)


def test_import():
    import src.guardalign_op as module

    assert hasattr(module, "compute_cost_matrix")
    assert hasattr(module, "sinkhorn_transport")
    assert hasattr(module, "compute_patch_scores")
    assert hasattr(module, "guardalign_op_score")


def test_cost_matrix_shape():
    image = torch.randn(4, 8)
    text = torch.randn(3, 8)

    cost = compute_cost_matrix(image, text)

    assert cost.shape == (4, 3)


def test_cost_matrix_finite_values():
    image = torch.randn(5, 6)
    text = torch.randn(2, 6)

    cost = compute_cost_matrix(image, text)

    assert torch.isfinite(cost).all()
    assert torch.all(cost >= 0.0)
    assert torch.all(cost <= 2.0)


def test_transport_plan_non_negative():
    cost = compute_cost_matrix(torch.randn(4, 5), torch.randn(3, 5))
    plan = sinkhorn_transport(cost, epsilon=0.1, num_iters=200)

    assert torch.all(plan >= 0.0)


def test_transport_plan_marginals_match_uniform_constraints():
    cost = compute_cost_matrix(torch.randn(4, 5), torch.randn(3, 5))
    plan = sinkhorn_transport(cost, epsilon=0.1, num_iters=300)

    expected_rows = torch.full((4,), 1.0 / 4, dtype=plan.dtype)
    expected_cols = torch.full((3,), 1.0 / 3, dtype=plan.dtype)

    assert torch.allclose(plan.sum(dim=1), expected_rows, atol=1e-4)
    assert torch.allclose(plan.sum(dim=0), expected_cols, atol=1e-4)


def test_identity_matching_toy_case():
    image = torch.eye(3)
    text = torch.eye(3)

    result = guardalign_op_score(image, text, epsilon=0.01, num_iters=400)
    plan = result["transport_plan"]
    scores = result["patch_scores"]

    assert torch.equal(plan.argmax(dim=1), torch.arange(3))
    assert torch.all(scores < 1e-3)


def test_permutation_matching_toy_case():
    image = torch.eye(3)
    text = torch.eye(3)[torch.tensor([2, 0, 1])]

    result = guardalign_op_score(image, text, epsilon=0.01, num_iters=400)
    plan = result["transport_plan"]

    assert torch.equal(plan.argmax(dim=1), torch.tensor([1, 2, 0]))


def test_patch_score_shape():
    cost = compute_cost_matrix(torch.randn(6, 4), torch.randn(5, 4))
    plan = sinkhorn_transport(cost, epsilon=0.1, num_iters=200)
    scores = compute_patch_scores(plan, cost)

    assert scores.shape == (6,)
    assert torch.all(scores >= 0.0)


def test_end_to_end_output_keys():
    result = guardalign_op_score(torch.randn(4, 7), torch.randn(3, 7))

    assert set(result.keys()) == {"cost_matrix", "transport_plan", "patch_scores"}


def test_invalid_feature_dimension_raises():
    image = torch.randn(4, 5)
    text = torch.randn(3, 6)

    with pytest.raises(ValueError, match="same feature dimension"):
        compute_cost_matrix(image, text)
