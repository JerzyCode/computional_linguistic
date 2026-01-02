from dataclasses import dataclass

import torch
import yaml


@dataclass(frozen=True)
class Config:
    model_name: str
    cuda_visible_devices: str = "0"
    temperature: float = 0.0
    max_new_tokens: int = 1024
    torch_dtype: torch.dtype = torch.bfloat16

    @staticmethod
    def load_from_yaml(file_path: str) -> "Config":
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)

        model_data = data.get("model")
        if not model_data:
            raise ValueError("model_name is required in the configuration file.")

        dtype_str = model_data.get("dtype", "bfloat16").lower()
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }

        return Config(
            model_name=model_data["name"],
            cuda_visible_devices=data.get("cuda_visible_devices", "0"),
            temperature=model_data.get("temperature", 0.0),
            max_new_tokens=model_data.get("max_new_tokens", 1024),
            torch_dtype=dtype_map.get(dtype_str, torch.bfloat16),
        )
