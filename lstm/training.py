from datetime import datetime

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from lstm.logger import log
from lstm.lstm_model import Lstm
from lstm.utils import EpochTracker, ModelSaver

# TODO - wyuczyć mały odel na 1 batch tzn. przeuczyć go i zobaczyć czy działa w ogóle trening.

DTYPE = torch.float16


def perform_batch(
    model: Lstm,
    batch: torch.Tensor,
    optim: AdamW,
    loss_fn: CrossEntropyLoss,
    device: str,
    scaler: torch.amp.GradScaler,
    use_amp: bool = False,
):
    optim.zero_grad()

    X, y = batch
    X = X.to(device)
    y = y.to(device)

    num_examples = min(5, X.shape[0])
    examples_X = X[:num_examples]
    examples_y = y[:num_examples]

    log.debug(
        f"Batch: X.shape={tuple(X.shape)}, "
        f"y.shape={tuple(y.shape)}, "
        f"examples_X: {examples_X}..."
        f"examples_y: {examples_y}..."
    )

    batch_size, seq_len = X.shape
    hidden_size = model.W_f.shape[0]

    H_t = torch.zeros(batch_size, hidden_size, device=device, dtype=DTYPE)
    C_t = torch.zeros(batch_size, hidden_size, device=device, dtype=DTYPE)

    ctx = torch.autocast(device, dtype=DTYPE) if use_amp else torch.enable_grad()

    total_loss = 0.0
    with ctx:
        # TODO - podmienić to, tak żeby działało dla LSTMa i Transformera na jednej pętli treningowej
        for t in range(seq_len - 1):
            input_ids = X[:, t]
            target_ids = y[:, t]
            logits, H_t, C_t = model(input_ids, H_t, C_t)

            num_to_log = min(3, logits.shape[0])
            log.debug(f"t={t}, logits {logits[:num_to_log]}... ")

            loss = loss_fn(logits, target_ids)
            total_loss += loss

    total_loss = total_loss / (seq_len - 1)

    if use_amp and scaler is not None:
        scaler.scale(total_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optim)
        scaler.update()
    else:
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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

                with torch.autocast(device, dtype=torch.float16):
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
    use_amp,
    lr=1e-3,
    device: str = "cpu",
    log_freq: int = 1,
    log_batch_freq: int = 100,
    patience: int = 20,
):
    log.info(
        f"Start training - end_time: {end_training_date}, device: {device}, lr={lr}"
    )

    epoch = 0
    model.to(device)
    scaler = torch.amp.GradScaler()
    optim = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optim, mode="min", factor=0.5, patience=patience)

    loss_fn = CrossEntropyLoss()

    while datetime.now() < end_training_date:
        epoch += 1
        epoch_tracker.start_epoch()

        model.train()
        running_loss = 0.0
        batches = 0

        prev_lr = optim.param_groups[0]["lr"]

        for batch_idx, batch in enumerate(training_set, start=1):
            loss = perform_batch(
                model, batch, optim, loss_fn, device, scaler=scaler, use_amp=use_amp
            )
            running_loss += loss
            batches += 1

            if batch_idx % log_batch_freq == 0 or batch_idx == len(training_set):
                log.info(
                    f"[Epoch {epoch}] Performed {batch_idx}/{len(training_set)} batches, "
                    f"current batch loss: {loss}, running avg loss: {running_loss / batch_idx}"
                )

        avg_train_loss = running_loss / batches
        eval_loss = evaluate_model(model, evaluation_set, loss_fn, device)
        scheduler.step(eval_loss)

        if epoch % log_freq == 0:
            log.info(
                f"[Epoch]: {epoch}, train_loss: {avg_train_loss}, eval_loss: {eval_loss}"
            )

        new_lr = optim.param_groups[0]["lr"]
        if new_lr < prev_lr:
            log.info(f"LR decreased from {prev_lr:.6f} to {new_lr:.6f}")

        epoch_tracker.end_epoch()
        epoch_tracker.insert_stats(epoch, avg_train_loss, eval_loss)

        if model_saver is not None:
            model_saver.save_model_checkpoint(model, epoch)

    if model_saver is not None:
        model_saver.save_model_checkpoint(model, epoch)
        log.info(
            f"Training is done with {epoch} epochs, real end_time: {datetime.now()}"
        )
