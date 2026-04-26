# Sinkhorn CLIP Demo

Minimal self-contained folder for testing a Sinkhorn optimal transport interface with either:

- synthetic embeddings via `pytest`
- real CLIP image/text embeddings via `examples/test_with_clip.py`

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
  --image path/to/image.jpg \
  --text "a dog" \
  --text "a person" \
  --text "a weapon"
```

The demo will:

- extract CLIP patch embeddings from the image
- extract CLIP text embeddings from the prompts
- compute `cost_matrix`, `transport_plan`, and `patch_scores`
- save heatmaps into `reports/clip_demo`
