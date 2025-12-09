import argparse
import logging
import os

from training import ModelType, Trainer, TrainingParameters
from transformer import ModelArgs, ModelFactory
from transformers import AutoTokenizer

from lab4_flash_attention.dataset import load_data

TOKENIZER = "radlab/polish-gpt2-small-v2"


def create_logger(model_type: ModelType) -> logging.Logger:
    logger = logging.getLogger(f"transformer_training.{model_type.value}")
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)

    log_path = f"logs/training_{model_type.value}.log"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)

    if not logger.hasHandlers():
        logger.addHandler(ch)

    return logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=[m.value for m in ModelType],
        help="Type of model to run",
    )

    args = parser.parse_args()
    model_type = ModelType(args.model_type)
    logger = create_logger(model_type)

    logger.info(f"Running for model type: {model_type.value}")

    logger.info(f"Loading tokenizer: {TOKENIZER}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

    model_args = ModelArgs(
        model_type=model_type,
        vocab_size=tokenizer.vocab_size,
        embedding_dim=64,  # FOR TESTING SMALLER
        num_heads=3,
        blocks_count=2,
        window_size=8,
    )

    logger.info(f"Creating model with args: \n{model_args}")

    model = ModelFactory.create_model(model_args)

    param = next(model.parameters())
    device = param.device
    dtype = param.dtype

    training_params = TrainingParameters(
        model_type=model_type,
        device=device,
        dtype=dtype,
        logger=logger,
        batch_size=32,
    )

    logger.info("Loading datasets")
    train_loader, eval_loader = load_data(
        seq_len=model_args.seq_len,
        dtype=dtype,
        batch_size=training_params.batch_size,
        tokenizer=tokenizer,
        logger=logger,
    )

    logger.info(f"Training parameters: \n{training_params}")

    trainer = Trainer(
        training_parameters=training_params,
        train_loader=train_loader,
        eval_loader=eval_loader,
    )

    trainer.train(model)

    logger.info("Run completed.")
