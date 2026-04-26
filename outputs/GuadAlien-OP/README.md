# GuadAlien-OP

Minimal CPU-only PyTorch implementation of a GuardAlign-style optimal transport operator for patch-level suspiciousness scoring.

## Files
- `src/guardalign_op.py`: core cost, Sinkhorn, patch-score, and end-to-end functions
- `tests/test_guardalign_op.py`: pytest suite using synthetic embeddings only
- `examples/run_toy_guardalign_op.py`: runnable toy example from the project root
- `implementation_report.md`: implementation notes and assumptions

## Requirements
- Python 3.9+
- PyTorch
- pytest

## Install
```bash
pip install torch pytest
```

## Run Tests
```bash
pytest -q
```

## Run Toy Example
```bash
python examples/run_toy_guardalign_op.py
```

## Notes
- CPU-only by default.
- No external datasets, model downloads, internet, CLIP, POT, or GeomLoss are required for the default workflow.
- POT and GeomLoss are not required and are treated only as optional reference oracles for manual validation.
