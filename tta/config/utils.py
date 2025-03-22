from tta.misc.registry import DATASET_REGISTRY
from tta.config.cifar10c.default import cfg as Cifar10Config

dataset_registry = DATASET_REGISTRY


def load_default_config(dataset_name: str = "Cifar10CDataset"):
    if dataset_name == "Cifar10CDataset":
        return Cifar10Config
    else:
        raise ValueError(f"Not valid dataset_name: {dataset_name}")
