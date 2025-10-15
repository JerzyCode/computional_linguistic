from datetime import datetime

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from lstm.lstm_model import Lstm
from lstm.utils import EpochTracker, ModelSaver


def perform_batch(
    model: Lstm,
    batch: torch.Tensor,
    optim: Adam,
    loss_fn: CrossEntropyLoss,
    device: str,
):
    X, y = batch
    X = X.to(device)
    y = y.to(device)
    optim.zero_grad()

    batch_size, seq_len = X.shape
    hidden_size = model.W_f.shape[0]

    H_t = torch.zeros(batch_size, hidden_size, device=device)
    C_t = torch.zeros(batch_size, hidden_size, device=device)

    total_loss = 0.0

    for t in range(seq_len - 1):
        input_ids = X[:, t]
        target_ids = y[:, t]

        logits, H_t, C_t = model(input_ids, H_t, C_t)
        loss = loss_fn(logits, target_ids)
        total_loss += loss

    total_loss = total_loss / (seq_len - 1)
    total_loss.backward()
    optim.step()
    return total_loss.item()


def evaluate_model(
    model: Lstm, loader: DataLoader, loss_fn: CrossEntropyLoss, device: str
):
    model.eval()
    total_loss = 0.0
    total_batches = 0

    with torch.inference_mode():
        for batch in loader:
            X, y = batch
            X = X.to(device)
            y = y.to(device)

            batch_size, seq_len = X.shape
            hidden_size = model.W_f.shape[0]

            H_t = torch.zeros(batch_size, hidden_size, device=device)
            C_t = torch.zeros(batch_size, hidden_size, device=device)

            batch_loss = 0.0
            for t in range(seq_len - 1):
                input_ids = X[:, t]
                target_ids = y[:, t]

                logits, H_t, C_t = model(input_ids, H_t, C_t)
                loss = loss_fn(logits, target_ids)
                batch_loss += loss

            batch_loss = batch_loss / (seq_len - 1)
            total_loss += batch_loss.item()
            total_batches += 1

    avg_loss = total_loss / total_batches if total_batches > 0 else float("nan")
    return avg_loss


def train(
    model: Lstm,
    training_set: DataLoader,
    evaluation_set: DataLoader,
    end_training_date: datetime,
    epoch_tracker: EpochTracker,
    model_saver: ModelSaver,
    lr=1e-3,
    device: str = "cpu",
    log_freq: int = 1,
):
    print(f"Start training - end_time: {end_training_date}, device: {device}, lr={lr}")

    epoch = 0
    model.to(device)
    optim = Adam(model.parameters(), lr=lr)

    loss_fn = CrossEntropyLoss()

    while datetime.now() < end_training_date:
        epoch += 1
        epoch_tracker.start_epoch()

        model.train()
        running_loss = 0.0
        batches = 0

        for batch in training_set:
            loss = perform_batch(model, batch, optim, loss_fn, device)
            running_loss += loss
            batches += 1

        avg_train_loss = running_loss / batches
        eval_loss = evaluate_model(model, evaluation_set, loss_fn, device)

        if epoch % log_freq == 0:
            print(
                f"Epoch: {epoch + 1}, train_loss: {avg_train_loss}, eval_loss: {eval_loss}"
            )

        epoch_tracker.end_epoch()
        epoch_tracker.insert_stats(epoch, avg_train_loss, eval_loss)

        if model_saver is not None:
            model_saver.save_model_checkpoint(model, epoch)

    if model_saver is not None:
        model_saver.save_model_checkpoint(model, epoch)
    print(f"Training is done with {epoch} epochs, real end_time: {datetime.now()}")
