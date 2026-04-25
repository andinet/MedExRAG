# Contributing to MedExRAG

Thanks for your interest in contributing.

## Development setup

```bash
git clone https://github.com/<your-fork>/MedExRAG.git
cd MedExRAG

python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate            # Windows

pip install -e .
pip install black isort flake8 pytest pytest-cov
```

## Running tests

```bash
pytest tests/ -v
pytest tests/ --cov=src/medexrag --cov-report=term-missing
```

The unit tests don't require the VLM or any GPU — they run in under two minutes on a laptop.

## Code style

This repo uses **black**, **isort**, and **flake8**. Configuration lives in [pyproject.toml](pyproject.toml) and [.flake8](.flake8). Run all three before opening a PR:

```bash
black src/medexrag/
isort src/medexrag/
flake8 src/medexrag/
```

CI runs the same checks ([.github/workflows/ci.yml](.github/workflows/ci.yml)).

## Submitting changes

1. Fork the repo and create a branch off `main`.
2. Keep PRs focused — one logical change per PR.
3. Add or update tests when changing behavior.
4. Update relevant docs in [docs/](docs/) if you change public API or config.
5. Open a PR with a clear description of the change and the motivation.

## Reporting bugs / requesting features

Use the issue templates under [.github/ISSUE_TEMPLATE/](.github/ISSUE_TEMPLATE/).

## Scope

This project is focused on agentic RAG for radiology workflows. Contributions that fit:

- New VLM backends, retrieval strategies, or evaluation metrics
- Observability improvements (additional metrics, traces, dashboards)
- Deployment targets (additional cloud providers, on-prem patterns)
- Documentation and examples

Out of scope: clinical advice, regulatory claims, or anything implying production medical use without proper validation. This is research/engineering software.

## License

By contributing you agree that your contributions are licensed under the [Apache License 2.0](LICENSE).
