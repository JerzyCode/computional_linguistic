import os

import uvicorn
from src.config import Config


def _export_envs(visible_devices: str) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
    os.environ["HF_HOME"] = "/net/tscratch/people/plgboksa/master/model"
    os.environ["HF_DATASETS_CACHE"] = "/net/tscratch/people/plgboksa/master/cache"
    os.environ["HF_METRICS_CACHE"] = "/net/tscratch/people/plgboksa/master/cache"


if __name__ == "__main__":
    config = Config.load_from_yaml("../../configs/config.yaml")
    _export_envs(config.cuda_visible_devices)

    from src.app import create_app

    app = create_app(config)

    uvicorn.run(app, host="0.0.0.0", port=8000)
