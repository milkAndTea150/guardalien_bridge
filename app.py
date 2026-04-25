
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from codex_client import call_codex_json

app = FastAPI(title="GuardAlien Research Code Bridge", version="0.1.0")

BASE_DIR = Path(os.getenv("GUARDALIEN_OUTPUT_DIR", "./outputs")).resolve()


class GenerateProjectRequest(BaseModel):
    task_name: str = "guardalign_op"
    algorithm_spec: Dict[str, Any] = Field(default_factory=dict)
    paper_excerpt: str = ""
    equations: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    codex_instruction: str = ""
    project_name: Optional[str] = None


class SaveProjectRequest(BaseModel):
    project_name: str
    files: Dict[str, str]


class RunTestsRequest(BaseModel):
    project_name: str


class RepairProjectRequest(BaseModel):
    project_name: str
    algorithm_spec: Dict[str, Any] = Field(default_factory=dict)
    pytest_result: Dict[str, Any] = Field(default_factory=dict)
    max_repair_round: int = 1


def _safe_project_dir(project_name: str) -> Path:
    safe_name = project_name.replace("/", "_").replace("..", "_").strip() or "guardalign_op"
    project_dir = (BASE_DIR / safe_name).resolve()
    if not str(project_dir).startswith(str(BASE_DIR)):
        raise ValueError("Invalid project path")
    return project_dir


def _read_project_files(project_dir: Path) -> Dict[str, str]:
    files: Dict[str, str] = {}
    for path in project_dir.rglob("*"):
        if path.is_file() and path.suffix in {".py", ".json", ".md", ".txt"}:
            rel_path = path.relative_to(project_dir).as_posix()
            files[rel_path] = path.read_text(encoding="utf-8")
    return files


def _write_files(project_dir: Path, files: Dict[str, str]) -> None:
    project_dir.mkdir(parents=True, exist_ok=True)
    for rel_path, content in files.items():
        file_path = (project_dir / rel_path).resolve()
        if not str(file_path).startswith(str(project_dir)):
            raise ValueError(f"Invalid file path: {rel_path}")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")


def _default_algorithm_spec(req: GenerateProjectRequest) -> Dict[str, Any]:
    if req.algorithm_spec:
        return req.algorithm_spec
    return {
        "algorithm_name": "guardalign_op",
        "research_context": "GuardAlign-style OP module for multimodal safety scoring",
        "task_type": "paper_specific_optimal_transport_module",
        "goal": "Compute cosine-distance cost matrix, Sinkhorn transport plan, patch-level suspiciousness scores, and a global score.",
        "inputs": {
            "image_embeds": "Tensor[M, D], image patch embeddings",
            "text_embeds": "Tensor[N, D], unsafe text/concept embeddings",
        },
        "outputs": {
            "cost_matrix": "Tensor[M, N]",
            "transport_plan": "Tensor[M, N]",
            "patch_scores": "Tensor[M]",
            "global_score": "scalar tensor",
        },
        "core_equations": req.equations or [
            "C(m,n)=1-cos(x_m,z_n)",
            "patch_score(m)=sum_n T(m,n)*C(m,n)",
        ],
        "core_steps": [
            "validate input shapes",
            "L2-normalize image and text embeddings",
            "compute cosine-distance cost matrix",
            "run entropic Sinkhorn with uniform marginals",
            "compute patch-level scores",
            "aggregate global score",
        ],
        "engineering_assumptions": [
            "Uses entropic Sinkhorn because the prompt asks for Sinkhorn OT.",
            "Uses uniform marginals unless otherwise specified.",
            "Uses synthetic embeddings for default unit tests.",
            "POT/GeomLoss are optional reference oracles, not required dependencies.",
        ],
        "required_tests": [
            "cost matrix shape and finite values",
            "transport plan non-negativity",
            "transport marginal constraints",
            "identity matching toy case",
            "permutation matching toy case",
            "patch score shape",
            "invalid shape raises ValueError",
        ],
    }


def _mock_project_files(project_name: str, algorithm_spec: Dict[str, Any], intentionally_buggy: bool = False) -> Dict[str, str]:
    """Return a deterministic runnable GuardAlign-style OP project.

    Set intentionally_buggy=True only if you want to demo repair behavior by
    creating a marginal-normalization bug. Normal generation defaults to correct.
    """
    row_norm_bug = "True" if intentionally_buggy else "False"

    op_py = f'''
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


_INTENTIONALLY_ROW_NORMALIZE = {row_norm_bug}


def _validate_embeddings(image_embeds: torch.Tensor, text_embeds: torch.Tensor) -> None:
    if not isinstance(image_embeds, torch.Tensor) or not isinstance(text_embeds, torch.Tensor):
        raise TypeError("image_embeds and text_embeds must be torch.Tensor")
    if image_embeds.ndim != 2 or text_embeds.ndim != 2:
        raise ValueError("image_embeds and text_embeds must have shape [num_items, hidden_dim]")
    if image_embeds.shape[1] != text_embeds.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: {{image_embeds.shape[1]}} vs {{text_embeds.shape[1]}}"
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
    return {{
        "cost_matrix": cost,
        "transport_plan": plan,
        "patch_scores": patch_scores,
        "global_score": global_score,
    }}
'''.lstrip()

    tests_py = '''
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
'''.lstrip()

    example_py = '''
import torch

from src.guardalign_op import guardalign_op_score


def main():
    torch.manual_seed(0)
    image_embeds = torch.randn(4, 8)
    unsafe_text_embeds = torch.randn(3, 8)
    result = guardalign_op_score(image_embeds, unsafe_text_embeds)

    print("cost_matrix shape:", tuple(result["cost_matrix"].shape))
    print("transport_plan shape:", tuple(result["transport_plan"].shape))
    print("patch_scores:", result["patch_scores"].detach().cpu().tolist())
    print("global_score:", float(result["global_score"]))


if __name__ == "__main__":
    main()
'''.lstrip()

    report_md = '''
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
'''.lstrip()

    readme_md = f'''
# {project_name}

Minimal generated project for GuardAlign-style OP/Sinkhorn verification.

## Run tests

```bash
python -m pytest tests -q --tb=short
```

## Run toy example

```bash
python examples/run_toy_guardalign_op.py
```

## Note

This project does not replace mature OT libraries such as POT or GeomLoss.
It is a compact paper-specific OP module used to test an AI research code-generation workflow.
'''.lstrip()

    return {
        "algorithm_spec.json": json.dumps(algorithm_spec, ensure_ascii=False, indent=2),
        "src/__init__.py": "",
        "src/guardalign_op.py": op_py,
        "tests/test_guardalign_op.py": tests_py,
        "examples/run_toy_guardalign_op.py": example_py,
        "implementation_report.md": report_md,
        "README.md": readme_md,
    }


def _mock_repair_files(current_files: Dict[str, str]) -> Dict[str, str]:
    """A deterministic repair: remove the demo row-normalization bug if present."""
    modified: Dict[str, str] = {}
    src = current_files.get("src/guardalign_op.py", "")
    if "_INTENTIONALLY_ROW_NORMALIZE = True" in src:
        modified["src/guardalign_op.py"] = src.replace(
            "_INTENTIONALLY_ROW_NORMALIZE = True", "_INTENTIONALLY_ROW_NORMALIZE = False"
        )

    report = current_files.get("implementation_report.md", "# Implementation Report\n")
    addition = """

## Repair History
- Root cause: transport plan was incorrectly row-normalized to sum to 1 instead of satisfying uniform OT marginals.
- Modified files: `src/guardalign_op.py`
- Fix summary: removed row-normalization branch so Sinkhorn plan preserves row sums 1/M and column sums 1/N.
"""
    if "## Repair History" not in report:
        modified["implementation_report.md"] = report.rstrip() + addition
    elif modified:
        modified["implementation_report.md"] = report

    return modified


def _build_repair_prompt(req: RepairProjectRequest, project_files: Dict[str, str]) -> str:
    return f"""
You are repairing a generated AI research algorithm implementation.

The project failed its tests.

Your task:
1. Read algorithm_spec.json.
2. Read the current source files.
3. Read the pytest failure log.
4. Identify the minimal root cause.
5. Patch only necessary files.
6. Preserve public function signatures.
7. Do not rewrite the whole project.
8. Do not delete tests just to make them pass.
9. Do not weaken mathematical property tests unless they are demonstrably incorrect.
10. If a test is wrong, explain why before modifying it.
11. Update implementation_report.md with failure cause, files changed, fix summary, and remaining limitations.

Return valid JSON only:
{{
  "modified_files": {{
    "relative/path.py": "new file content"
  }},
  "root_cause": "...",
  "repair_summary": "...",
  "remaining_limitations": ["..."]
}}

[algorithm_spec.json]
{json.dumps(req.algorithm_spec, ensure_ascii=False, indent=2)}

[current_project_files]
{json.dumps(project_files, ensure_ascii=False, indent=2)}

[pytest_result]
{json.dumps(req.pytest_result, ensure_ascii=False, indent=2)}
""".strip()


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": "guardalien-bridge",
        "base_dir": str(BASE_DIR),
        "mock_codegen": os.getenv("MOCK_CODEGEN", "1"),
        "mock_repair": os.getenv("MOCK_REPAIR", "1"),
    }


@app.post("/generate_project")
def generate_project(req: GenerateProjectRequest) -> Dict[str, Any]:
    project_name = req.project_name or req.task_name or "guardalign_op"
    algorithm_spec = _default_algorithm_spec(req)

    use_mock = os.getenv("MOCK_CODEGEN", "1") == "1"
    intentionally_buggy = os.getenv("MOCK_GENERATE_BUG", "0") == "1"

    if use_mock:
        files = _mock_project_files(project_name, algorithm_spec, intentionally_buggy=intentionally_buggy)
        return {
            "status": "generated",
            "mode": "mock",
            "project_name": project_name,
            "files": files,
            "summary": "Generated deterministic GuardAlign-style OP project in mock mode.",
            "assumptions": algorithm_spec.get("engineering_assumptions", []),
        }

    result = call_codex_json(req.codex_instruction)
    files = result.get("files", {})
    if "algorithm_spec.json" not in files:
        files["algorithm_spec.json"] = json.dumps(algorithm_spec, ensure_ascii=False, indent=2)
    return {
        "status": "generated",
        "mode": "codex",
        "project_name": result.get("project_name", project_name),
        "files": files,
        "summary": result.get("summary", ""),
        "assumptions": result.get("assumptions", []),
    }


@app.post("/save_project")
def save_project(req: SaveProjectRequest) -> Dict[str, Any]:
    project_dir = _safe_project_dir(req.project_name)
    _write_files(project_dir, req.files)
    return {
        "status": "saved",
        "project_name": req.project_name,
        "project_dir": str(project_dir),
        "file_count": len(req.files),
    }


@app.post("/run_tests")
def run_tests(req: RunTestsRequest) -> Dict[str, Any]:
    project_dir = _safe_project_dir(req.project_name)
    if not project_dir.exists():
        return {"status": "error", "message": f"Project not found: {req.project_name}"}

    start_time = time.time()

    pytest_proc = subprocess.run(
        ["python", "-m", "pytest", "tests", "-q", "--tb=short"],
        cwd=project_dir,
        env={**os.environ, "PYTHONPATH": str(project_dir)},
        capture_output=True,
        text=True,
        timeout=60,
    )

    example_path = project_dir / "examples" / "run_toy_guardalign_op.py"
    if example_path.exists():
        example_proc = subprocess.run(
            ["python", "examples/run_toy_guardalign_op.py"],
            cwd=project_dir,
            env={**os.environ, "PYTHONPATH": str(project_dir)},
            capture_output=True,
            text=True,
            timeout=60,
        )
    else:
        example_proc = subprocess.CompletedProcess(args=[], returncode=0, stdout="No example script found; skipped.\n", stderr="")

    elapsed = time.time() - start_time
    passed = pytest_proc.returncode == 0 and example_proc.returncode == 0

    result = {
        "status": "pass" if passed else "fail",
        "project_name": req.project_name,
        "elapsed_seconds": elapsed,
        "pytest": {
            "command": "python -m pytest tests -q --tb=short",
            "returncode": pytest_proc.returncode,
            "stdout": pytest_proc.stdout,
            "stderr": pytest_proc.stderr,
        },
        "example": {
            "command": "python examples/run_toy_guardalign_op.py",
            "returncode": example_proc.returncode,
            "stdout": example_proc.stdout,
            "stderr": example_proc.stderr,
        },
    }

    reports_dir = project_dir / "reports"
    reports_dir.mkdir(exist_ok=True)
    (reports_dir / "pytest_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


@app.post("/repair_project")
def repair_project(req: RepairProjectRequest) -> Dict[str, Any]:
    project_dir = _safe_project_dir(req.project_name)
    if not project_dir.exists():
        return {"status": "error", "message": f"Project not found: {req.project_name}"}

    # Sequential workflow compatibility: if tests already pass, skip repair.
    if req.pytest_result.get("status") == "pass":
        record = {
            "status": "skipped",
            "root_cause": "No repair needed because tests already passed.",
            "repair_summary": "Skipped repair.",
            "remaining_limitations": [],
            "modified_files": [],
        }
        return record

    project_files = _read_project_files(project_dir)
    use_mock = os.getenv("MOCK_REPAIR", "1") == "1"

    if use_mock:
        modified_files = _mock_repair_files(project_files)
        root_cause = "Detected a likely Sinkhorn marginal/normalization issue or used deterministic mock repair."
        repair_summary = "Applied deterministic repair patch if the known demo bug was present."
        remaining_limitations = ["Mock repair only fixes the known marginal-normalization demo bug."]
    else:
        repair_prompt = _build_repair_prompt(req, project_files)
        result = call_codex_json(repair_prompt)
        modified_files = result.get("modified_files", {})
        root_cause = result.get("root_cause", "")
        repair_summary = result.get("repair_summary", "")
        remaining_limitations = result.get("remaining_limitations", [])

    if modified_files:
        _write_files(project_dir, modified_files)

    repair_record = {
        "status": "repaired" if modified_files else "no_changes",
        "root_cause": root_cause,
        "repair_summary": repair_summary,
        "remaining_limitations": remaining_limitations,
        "modified_files": list(modified_files.keys()),
    }

    reports_dir = project_dir / "reports"
    reports_dir.mkdir(exist_ok=True)
    history_path = reports_dir / "repair_history.json"
    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))
    else:
        history = []
    history.append(repair_record)
    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

    return repair_record


@app.get("/read_project_summary/{project_name}")
def read_project_summary(project_name: str) -> Dict[str, Any]:
    project_dir = _safe_project_dir(project_name)
    if not project_dir.exists():
        return {"status": "error", "message": f"Project not found: {project_name}"}

    files = sorted(path.relative_to(project_dir).as_posix() for path in project_dir.rglob("*") if path.is_file())
    report_path = project_dir / "implementation_report.md"
    report = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    return {
        "status": "ok",
        "project_name": project_name,
        "project_dir": str(project_dir),
        "files": files,
        "implementation_report": report,
    }
