# Sinkhorn CLIP Demo

This folder is a self-contained demo for testing the current GuardAlign-style Sinkhorn OT code on:

- synthetic embeddings via `pytest`
- real CLIP image/text embeddings via `examples/test_with_clip.py`

## Structure

```text
sinkhorn_clip_demo/
├── requirements.txt
├── src/
│   ├── __init__.py
│   └── guardalign_ot.py
├── tests/
│   └── test_guardalign_ot.py
└── examples/
    └── test_with_clip.py
```

## Install

```bash
cd sinkhorn_clip_demo
pip install -r requirements.txt
```

## Run Unit Tests

```bash
PYTHONPATH=. pytest -q tests/test_guardalign_ot.py
```

## Run CLIP Demo

```bash
PYTHONPATH=. python examples/test_with_clip.py \
  --image ./privacy.jpg \
  --text "The image contains private contents."
```

You can pass `--text` multiple times:

```bash
PYTHONPATH=. python examples/test_with_clip.py \
  --image ./privacy.jpg \
  --text "The image contains private contents." \
  --text "The image contains a face."
```

## What The Demo Does

- extracts CLIP image patch embeddings
- extracts CLIP text embeddings
- computes `cost_matrix`
- computes `transport_plan`
- computes `patch_scores`
- saves one final heatmap image

## Output

The final heatmap is written to:

```text
reports/clip_demo/patch_scores_heatmap.png
```

## Current Interface

`guardalign_op_score(...)` returns:

- `cost_matrix`
- `transport_plan`
- `patch_scores`
