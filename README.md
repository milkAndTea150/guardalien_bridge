# GuardAlien Bridge

This repository contains two parts:

- a FastAPI bridge used by the Dify/Codex workflow
- a standalone `sinkhorn_clip_demo/` folder for testing Sinkhorn OT with CLIP image/text embeddings

## Directory Structure

```text
guardalien_bridge/
├── app.py
├── codex_client.py
├── run_bridge.sh
├── requirements.txt
├── test_health.sh
├── debug_codex_workspace/
├── tmp/
└── sinkhorn_clip_demo/
    ├── README.md
    ├── requirements.txt
    ├── src/
    │   ├── __init__.py
    │   └── guardalign_ot.py
    ├── tests/
    │   └── test_guardalign_ot.py
    └── examples/
        └── test_with_clip.py
```

## Main Files

- `app.py`: FastAPI service entry for the bridge
- `codex_client.py`: local Codex CLI wrapper and workspace handling
- `run_bridge.sh`: start script for the bridge
- `debug_codex_workspace/`: local prompt/stdout/stderr traces for Codex runs
- `tmp/`: temporary payloads and test artifacts
- `sinkhorn_clip_demo/`: self-contained Sinkhorn + CLIP test project

## Start The Service

Install bridge dependencies:

```bash
pip install -r requirements.txt
```

Start the bridge:

```bash
MOCK_CODEGEN=0 MOCK_REPAIR=0 \
CODEX_CMD='codex exec -m gpt-5.4 --full-auto -' \
OUTPUTS_DIR='/path/to/guardalien_bridge/outputs' \
./run_bridge.sh
```

The service listens on:

```text
http://127.0.0.1:8787
```

Optional health check:

```bash
./test_health.sh
```

## Sinkhorn CLIP Demo

The demo in `sinkhorn_clip_demo/` extracts CLIP image patch embeddings and text embeddings, runs Sinkhorn OT, and saves the final patch-score heatmap.

Install demo dependencies:

```bash
cd sinkhorn_clip_demo
pip install -r requirements.txt
```

Run unit tests:

```bash
PYTHONPATH=. pytest -q tests/test_guardalign_ot.py
```

Run the final CLIP test:

```bash
PYTHONPATH=. python examples/test_with_clip.py --image ./privacy.jpg --text "The image contains private contens."
```

The final heatmap will be written under:

```text
sinkhorn_clip_demo/reports/clip_demo/patch_scores_heatmap.png
```
