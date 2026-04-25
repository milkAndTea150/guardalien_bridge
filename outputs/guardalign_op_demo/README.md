# guardalign_op_demo

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
