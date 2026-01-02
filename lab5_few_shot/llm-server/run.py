import argparse
import os

import uvicorn
from src.config import Config


def _export_envs(visible_devices: str) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
    os.environ["HF_HOME"] = "/net/tscratch/people/plgboksa/master/model"
    os.environ["HF_DATASETS_CACHE"] = "/net/tscratch/people/plgboksa/master/cache"
    os.environ["HF_METRICS_CACHE"] = "/net/tscratch/people/plgboksa/master/cache"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LLM server with specified config file."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file for the model.",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host for the Uvicorn server."
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port for the Uvicorn server."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    config = Config.load_from_yaml(args.config)
    _export_envs(config.cuda_visible_devices)

    from src.app import create_app

    app = create_app(config)

    uvicorn.run(app, host=args.host, port=args.port)
