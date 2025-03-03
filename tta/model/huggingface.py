import yaml
from pathlib import Path
from huggingface_hub import hf_hub_download

from tta.path import WEIGHT_DIR, CONFIG_DIR


with open(CONFIG_DIR / "huggingface.yaml", "r") as file:
    model_configs = yaml.safe_load(file)


def download_model(
    model_name: str,
    data_name: str,
    dst_path: str = WEIGHT_DIR,
):
    model_config = model_configs[data_name][model_name]
    model_dst_path = Path(dst_path) / model_name
    model_filepath = model_dst_path / model_config["filename"]
    if not (model_filepath).exists():
        hf_hub_download(
            repo_id=model_config["repo_id"],
            local_dir=model_dst_path,
            filename=model_config["filename"],
            revision=model_config["revision"],
        )
    return model_filepath
