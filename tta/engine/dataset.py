from tta.misc.registry import DATASET_REGISTRY
from tta.config.cifar10c.default import cfg as Cifar10Config

dataset_registry = DATASET_REGISTRY


def load_default_config(dataset_name: str = "cifar10"):
    if dataset_name == "cifar10":
        return Cifar10Config
    else:
        raise ValueError(f"Not valid dataset_name: {dataset_name}")


def load_engine_dataset(config):
    dataset = dataset_registry.get(config.DATA.NAME)(
        corrupt_domain_orders=[shift_name],
        severity=severity_level,
    )
