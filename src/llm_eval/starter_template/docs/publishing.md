# Publishing

This repo is ready to publish as both a GitHub project and a PyPI package.

## One-Time Setup

1. Create the GitHub repository.

   ```bash
   gh repo create keitabroadwater/apst-starter-kit --public --source . --remote origin --push
   ```

2. Create the PyPI project name by publishing the first release, or reserve it manually if PyPI
   allows that flow for your account.

3. Configure PyPI Trusted Publishing for the GitHub repository:

   - PyPI project: `apst-starter-kit`
   - Owner: `keitabroadwater`
   - Repository: `apst-starter-kit`
   - Workflow name: `publish.yml`
   - Environment: `pypi`

## Local Build Check

```bash
python -m pip install -e ".[dev]"
python -m build
python -m twine check dist/*
```

For a clean local install test:

```bash
python -m venv /tmp/apst-package-test
/tmp/apst-package-test/bin/python -m pip install dist/apst_starter_kit-0.1.0-py3-none-any.whl
/tmp/apst-package-test/bin/apst init /tmp/apst-demo
cd /tmp/apst-demo
/tmp/apst-package-test/bin/apst run --config configs/demo_mock.yaml --no-resume
```

## Release

1. Update `version` in `pyproject.toml` and `__version__` in `src/llm_eval/__init__.py`.
2. Commit the release changes.
3. Tag the release:

   ```bash
   git tag v0.1.0
   git push origin main --tags
   ```

4. Create a GitHub Release from the tag. The `publish.yml` workflow publishes to PyPI when the
   release is published.

After release, users can run:

```bash
pip install apst-starter-kit
apst init my-apst-demo
cd my-apst-demo
apst run --config configs/demo_mock.yaml
```
