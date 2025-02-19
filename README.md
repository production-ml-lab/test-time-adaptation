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
    python tests/installation_check.py
    ```
