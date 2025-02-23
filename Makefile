pre-commit:
	pre-commit run --all-files

run-source:
	uv run python run_tta.py --config config/cifar10c/source.yaml
