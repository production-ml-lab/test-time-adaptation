# Test Time Adaptation

## Installation

### Packages

1. Install uv.
    1. Install with pip
        ```bash
        pip install uv
        ```
    2. Install with curl
        ```bash
        curl -LsSf https://astral.sh/uv/install.sh | sh
        ```
2. Make virtualenv (Recommend: make it in project root directory)
    ```bash
    uv venv .venv --python=3.11.11
    ```
3. Active the virtual env
    ```bash
    source .venv/bin/activate
    ```
4. Install uv packages with developer mode.
    ```bash
    uv pip install -e .
    ```

### Login to Huggingface

1. Install `huggingface-cli`.

    ```bash
    pip install -U "huggingface_hub[cli]"
    ```

2. Login to huggingface hub.

    ```bash
    huggingface-cli login
    ```

3. Check login auth.
    ```bash
    huggingface-cli whoami
    ```
4. Check orgs is `mlp-tta`.
    ```bash
    > huggingface-cli whoami
    aiden-jeon
    orgs:  mlp-tta
    ```

### Test

1. Run test code.
    ```bash
    make run-source
    ```

## How to run TTA with config

In our package we use registry that is used on Facebook.
With configured yaml file you can easily run several experiments.

For example to test with source yaml file.
As we recommend to use venv, you can run with `uv run` and `--config` arguments.

```bash
uv run python run_tta.py --config config/cifar10c/source.yaml
```

## For Developer

### Install pre-commit

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

4. Write baseline method to `config/cifar10/my_new_method.yaml`
    ```yaml
    MODEL:
        ADAPTATION: MyNewMethod
        NAME: resnet26
        BACKEND: custom
    DATA:
        BATCH_SIZE: YOUR_BATCH_SIZE
    SHIFT:
        SEVERITY: ...
        TYPE: ...
    ```

### Upload Huggingface Backend Model

1. Check login auth.
    ```bash
    huggingface-cli whoami
    ```
2. Check orgs is `mlp-tta`.
    ```bash
    aiden-jeon
    orgs:  mlp-tta
    ```
3. Upload model to hub.
    ```bash
    huggingface-cli upload mlp-tta/resnet-26 models/resnet-26 .
    ```

## Run streamlit demo app

1. Install dev extras
    ```bash
    uv sync --all-extras
    ```
2. Run streamlit run app
    ```bash
    uv run streamlit run app/app.py
    ```
