import torch
from training import (
    MAX_EVAL_ELEMENTS,
    MAX_TRAIN_ELEMENTS,
    create_data_loader,
    device,
    logger,
    tokenizer,
    train,
    train_path,
    val_path,
)
from transformer import TransformerLanguageModel

if __name__ == "__main__":
    max_new_tokens = 100
    seq_len = 128
    lr = 1e-4
    dropout = 0.3
    batch_size = 196
    heads_count = 8
    embedding_dim = 512
    blocks_count = 8
    training_minutes = 60 * 10

    # max_new_tokens = 100
    # seq_len = 16
    # lr = 1e-3
    # batch_size = 16
    # heads_count = 2
    # embedding_dim = 32
    # blocks_count = 2
    # training_minutes = 3

    logger.info(f"Start training with device: {device}")
    logger.info("Net and training parameters:")
    logger.info(f"seq_len: {seq_len}")
    logger.info(f"lr: {lr}")
    logger.info(f"batch_size: {batch_size}")
    logger.info(f"heads_count: {heads_count}")
    logger.info(f"embedding_dim: {embedding_dim}")
    logger.info(f"blocks_count: {blocks_count}")
    logger.info(f"training_minutes: {training_minutes}")

    logger.info("Start loading training data")
    training_data_loader = create_data_loader(
        train_path,
        batch_size=batch_size,
        seq_len=seq_len,
        max_elements=MAX_TRAIN_ELEMENTS,
    )
    logger.info("Finished training data")

    logger.info("Start loading eval data")
    eval_data_loader = create_data_loader(
        val_path, batch_size=batch_size, seq_len=seq_len, max_elements=MAX_EVAL_ELEMENTS
    )
    logger.info("Finished eval data")

    model = TransformerLanguageModel(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=embedding_dim,
        num_heads=heads_count,
        blocks_count=blocks_count,
        seq_len=seq_len,
        device=device,
        dropout=dropout,
    ).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {trainable_params}")

    generated_before_training = model.generate_text(
        tokenizer, max_new_tokens=max_new_tokens
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    train(
        model,
        optimizer,
        training_data_loader,
        eval_data_loader,
        training_minutes=training_minutes,
        log_freq=1,
        save_freq=3,
        batch_log_interval=100,
    )

    generated_after_training = model.generate_text(
        tokenizer, max_new_tokens=max_new_tokens
    )

    logger.info("Generated text before training:")
    logger.info(generated_before_training)

    logger.info("\n\nGenerated text after training:")
    logger.info(generated_after_training)
