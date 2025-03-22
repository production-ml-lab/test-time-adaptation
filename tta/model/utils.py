import yaml
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from robustbench import load_model as robustbench_load_model

from tta.utils.path import WEIGHT_DIR


with open(Path(__file__).parent / "huggingface.yaml", "r") as file:
    model_configs = yaml.safe_load(file)


def download_huggingface_model(
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


def load_huggingface_model(
    model_name: str,
    data_name: str,
    device: str = "cpu",
    dst_path: str = WEIGHT_DIR,
):
    model_filepath = download_huggingface_model(model_name, data_name, dst_path)
    state_dict = torch.load(
        model_filepath,
        map_location=device,
        weights_only=True,
    )
    if "state_dict" in state_dict.keys():
        state_dict = state_dict["state_dict"]
    return state_dict


def load_robustbench_model(data_name: str):
    model = robustbench_load_model(
        model_name="Standard",
        model_dir=WEIGHT_DIR / "robustbench",
        dataset=data_name,
        threat_model="Linf",
    )
    return model
