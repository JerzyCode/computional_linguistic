import argparse
import logging
import os

import torch
from dataset import load_data
from training import ModelType, Trainer, TrainingParameters
from transformer import ModelArgs, ModelFactory
from transformers import AutoTokenizer

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

    log_path = f"results/{model_type.value}/training.log"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)

    if not logger.hasHandlers():
        logger.addHandler(ch)

    return logger

def find_max_batch_size(
    model,
    seq_len: int,
    vocab_size: int,
    lr: float,
    logger: logging.Logger,
    device: str = "cuda",
    max_memory_fraction: float = 0.9,
) -> int:
    best = 1
    batch_sizes = [16 * i for i in range(1, 200)]  

    for batch in batch_sizes:
        logger.info(f"Testing batch size: {batch}")
        try:
            torch.cuda.empty_cache()
            model.zero_grad(set_to_none=True)

            X = torch.randint(
                0, vocab_size, (batch, seq_len), device=device, dtype=torch.long
            )
            y = X.clone()

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            out, _ = model(X)
            loss = torch.nn.functional.cross_entropy(
                out.view(-1, out.size(-1)),
                y.view(-1)
            )
            loss.backward()
            optimizer.step()

            torch.cuda.synchronize()

            free, total = torch.cuda.mem_get_info(device)
            used_ratio = 1 - free / total
            if used_ratio > max_memory_fraction:
                logger.info(f"Exceeded memory fraction: {used_ratio:.2f} > {max_memory_fraction}")
                break

            best = batch

            del X, y, out, loss, optimizer

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                logger.info(f"Out of memory at batch size: {batch}")
                break
            else:
                raise e

    safe_batch = int(best * 0.9)  # margni 10%
    logger.info(f"Maximum safe batch size: {safe_batch}")
    return max(1, safe_batch)



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
        num_heads=2,
        blocks_count=2,
        window_size=8,
    )

    logger.info(f"Creating model with args: \n{model_args}")

    model = ModelFactory.create_model(model_args)

    param = next(model.parameters())
    device = param.device
    dtype = param.dtype

    logger.info("Calculating maximum batch size")
    max_batch_size = find_max_batch_size(
        model=model,
        lr=1e-4,
        seq_len=model_args.seq_len,
        vocab_size=tokenizer.vocab_size,
        device=device,
        logger=logger,
    )
    logger.info(f"Maximum batch size determined: {max_batch_size}")

    training_params = TrainingParameters(
        model_type=model_type,
        device=device,
        dtype=dtype,
        logger=logger,
        batch_size=max_batch_size,
        seq_len=model_args.seq_len,
        eval_interval=25,
    )

    logger.info(f"Training parameters: \n{training_params}")

    logger.info("Loading datasets")
    train_loader, eval_loader = load_data(
        seq_len=model_args.seq_len,
        dtype=dtype,
        batch_size=max_batch_size,
        tokenizer=tokenizer,
        logger=logger,
    )

    logger.info("TrainBatches size: {}".format(len(train_loader)))
    logger.info("EvalBatches size: {}".format(len(eval_loader)))

    trainer = Trainer(
        training_parameters=training_params,
        train_loader=train_loader,
        eval_loader=eval_loader,
        max_batch_size=max_batch_size,
    )

    metrics = trainer.train(model)

    logger.info("Run completed.")

    logger.info("Saving metrics")
    metrics.save_metrics()

