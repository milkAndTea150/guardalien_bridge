# GuardAlien Research Code Bridge

This is the local FastAPI bridge for the Dify + Codex workflow:

```text
Generate Project → Save Project → Run Tests → Repair Once → Run Tests Again → Report
```

It is designed for the upgraded DSL workflow `GuideAlien_generate_test_repair_report.dsl.yml`.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Check bridge status |
| POST | `/generate_project` | Generate a project file dictionary from algorithm spec / Codex instruction |
| POST | `/save_project` | Save generated files under `outputs/<project_name>/` |
| POST | `/run_tests` | Run `pytest` and the toy example |
| POST | `/repair_project` | If tests failed, repair once; if tests passed, return `skipped` |
| GET | `/read_project_summary/{project_name}` | Inspect saved project and implementation report |

## Install

```bash
cd guardalien_bridge
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Start

```bash
./run_bridge.sh
```

The server listens on:

```text
http://127.0.0.1:8787
```

Dify should call:

```text
http://host.docker.internal:8787
```

## Health check

```bash
curl -s http://127.0.0.1:8787/health | python -m json.tool
```

## Default demo mode

By default, this bridge uses deterministic mock generation and mock repair:

```bash
export MOCK_CODEGEN=1
export MOCK_REPAIR=1
```

This means the Monday demo does not depend on Codex availability. It will generate a runnable GuardAlign-style OP/Sinkhorn project with:

```text
algorithm_spec.json
src/guardalign_op.py
tests/test_guardalign_op.py
examples/run_toy_guardalign_op.py
implementation_report.md
README.md
```

## Demo with an intentional bug

To show repair behavior, start the bridge with:

```bash
MOCK_GENERATE_BUG=1 ./run_bridge.sh
```

Then the first test run should fail on marginal constraints. `/repair_project` will patch the known row-normalization bug, and the second test run should pass.

## Connect to your real Codex bridge later

The only function you need to replace is:

```python
# codex_client.py
call_codex_json(prompt: str) -> dict
```

Current behavior:

- If `MOCK_CODEGEN=1`, `/generate_project` returns deterministic local project files.
- If `MOCK_CODEGEN=0`, it calls `call_codex_json()`.
- If `MOCK_REPAIR=1`, `/repair_project` uses deterministic repair for the known demo bug.
- If `MOCK_REPAIR=0`, it calls `call_codex_json()` with the repair prompt.

## Manual API test

```bash
curl -s -X POST http://127.0.0.1:8787/generate_project \
  -H 'Content-Type: application/json' \
  -d '{
    "task_name": "guardalign_op_demo",
    "algorithm_spec": {},
    "paper_excerpt": "GuardAlign models image patch embeddings and unsafe text embeddings as two distributions.",
    "equations": ["C(m,n)=1-cos(x_m,z_n)", "patch_score(m)=sum_n T(m,n)*C(m,n)"],
    "constraints": ["Use PyTorch", "Add unit tests", "Use Sinkhorn"],
    "codex_instruction": "Generate project",
    "project_name": "guardalign_op_demo"
  }' > /tmp/generated_project.json

python - <<'PY'
import json, requests
with open('/tmp/generated_project.json') as f:
    data=json.load(f)
requests.post('http://127.0.0.1:8787/save_project', json={
    'project_name': data['project_name'],
    'files': data['files'],
}).raise_for_status()
print(requests.post('http://127.0.0.1:8787/run_tests', json={'project_name': data['project_name']}).json())
PY
```

## Generated outputs

Projects are saved to:

```text
guardalien_bridge/outputs/<project_name>/
```

Test and repair records are saved to:

```text
guardalien_bridge/outputs/<project_name>/reports/
```
