
import torch
from training import (
    create_data_loader,
    device,
    generate_text,
    logger,
    small_train_path,
    tokenizer,
    train,
    val_path,
)
from transformer import TransformerLanguageModel

if __name__ == "__main__":
    max_new_tokens = 100
    seq_len = 128
    lr = 1e-3
    batch_size = 64
    heads_count = 4
    embedding_dim = 64
    blocks_count = 4
    training_minutes = 3

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
        small_train_path, batch_size=batch_size, seq_len=seq_len
    )
    logger.info("Finished training data")

    logger.info("Start loading eval data")
    eval_data_loader = create_data_loader(
        val_path, batch_size=batch_size, seq_len=seq_len
    )
    logger.info("Finished eval data")

    model = TransformerLanguageModel(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=embedding_dim,
        num_heads=heads_count,
        blocks_count=blocks_count,
        seq_len=seq_len,
        device=device
    ).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {trainable_params}")

    generated_before_training = generate_text(model, max_new_tokens=max_new_tokens)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    train(
        model,
        optimizer,
        training_data_loader,
        eval_data_loader,
        training_minutes=training_minutes,
        log_freq=1,
        save_freq=25,
        batch_log_interval=25
    ).to(device)


    generated_after_training = generate_text(model, max_new_tokens=max_new_tokens)

    logger.info("Generated text before training:")
    logger.info(generated_before_training)

    logger.info("\n\nGenerated text after training:")
    logger.info(generated_after_training)
