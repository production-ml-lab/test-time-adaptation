# test-time-adaptation

## Installation

1. Install uv.

    ```bash
    # method 1.
    pip install uv

    # method 2.
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2. Make virtualenv (Recommend: make it in project root directory)
    ```bash
    uv venv .venv --python=python3.11.11
    ```
3. Active the virtual env
    ```bash
    source .venv/bin/activate
    ```
4. Install uv packages with developer mode.
    ```bash
    uv pip install -e .
    ```
5. Run test code.
    ```bash
    make run-source
    ```

## Run TTA

1. Source test example
    ```bash
    python run_tta.py --config config/cifar10c/source.yaml
    ```

## For Developer

Install pre-commit.

```bash
uv pip install pre-commit
```

Install the git hook scripts.

```bash
pre-commit install
```

Run pre-commit files

```bash
make pre-commit
```

### How to add new method

1. Write code under the `tta/method/my_new_method.py`
2. New method class should use `@ADAPTATION_REGISTRY.register()` decorator to be used with config.
    ```python
    @ADAPTATION_REGISTRY.register()
    class MyNewMethod(BaseMethod):
        ...
    ```
3. Add model name to `tta/method/__init__.py`

    ```python
    from .base import BaseMethod
    ...
    from .my_new_method import MyNewMethod

    __all__ = ["BaseMethod", ..., "MyNewMethod"]
    ```
