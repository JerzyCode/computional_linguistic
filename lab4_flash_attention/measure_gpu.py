import argparse

import torch
from dataset import load_data
from run import create_logger
from training import ModelType, Trainer, TrainingParameters
from transformer import ModelArgs, ModelFactory
from transformers import AutoTokenizer

TOKENIZER = "radlab/polish-gpt2-small-v2"


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
        window_size=8,
        blocks_count=8,
    )

    logger.info(f"Creating model with args: \n{model_args}")

    model = ModelFactory.create_model(model_args)

    param = next(model.parameters())
    device = param.device
    dtype = param.dtype
    batch_size = 128

    training_params = TrainingParameters(
        model_type=model_type,
        device=device,
        dtype=dtype,
        logger=logger,
        batch_size=batch_size,
        seq_len=model_args.seq_len,
        eval_interval=10,
    )

    logger.info(f"Training parameters: \n{training_params}")

    logger.info("Loading datasets")
    train_loader, eval_loader = load_data(
        seq_len=model_args.seq_len,
        dtype=dtype,
        batch_size=batch_size,
        tokenizer=tokenizer,
        logger=logger,
    )

    logger.info("TrainBatches size: {}".format(len(train_loader)))
    logger.info("EvalBatches size: {}".format(len(eval_loader)))

    trainer = Trainer(
        training_parameters=training_params,
        train_loader=train_loader,
        eval_loader=eval_loader,
        max_batch_size=batch_size,
    )

    # metrics = trainer.train(model)

    memory_used = 0
    durations = 0
    steps = 10
    for i in range(steps):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        avg_loss, evaluation_duration_avg = trainer._perform_evaluation(model)
        torch.cuda.synchronize()
        print(
            f"Step {i + 1}/{steps} - Eval Loss: {avg_loss}, Duration: {evaluation_duration_avg}, Max Memory: {torch.cuda.max_memory_allocated()}"
        )
        durations += evaluation_duration_avg
        memory_used += torch.cuda.max_memory_allocated()

    logger.info(f"Average evaluation loss over {steps} steps: {avg_loss}")
    logger.info(f"Average evaluation duration over {steps} steps: {durations / steps}")
    logger.info(f"Average GPU memory used over {steps} steps: {memory_used / steps}")
