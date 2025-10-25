import logging
import os
import time
from datetime import datetime

import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer

writer = SummaryWriter(f"runs/lstm_training_{time.ctime(time.time())}")

tokenizer = AutoTokenizer.from_pretrained("dkleczek/bert-base-polish-uncased-v1")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

PREPARED_DATASET_DIR = "./prepared_data"
MAX_TRAIN_ELEMENTS = 196_00
MAX_EVAL_ELEMENTS = 1_96
log_file = f"logs/training_{timestamp}.log"

os.makedirs(os.path.dirname(log_file), exist_ok=True)

small_train_path = os.path.join(PREPARED_DATASET_DIR, "small_train_data.pt")
small_val_path = os.path.join(PREPARED_DATASET_DIR, "small_val_data.pt")
train_path = os.path.join(PREPARED_DATASET_DIR, "train_data.pt")
val_path = os.path.join(PREPARED_DATASET_DIR, "val_data.pt")

device = "cpu"
dtype = torch.float32

torch.manual_seed(1234)

logger = logging.getLogger("Training")
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(formatter)


logger.addHandler(file_handler)


class TokenDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) // self.seq_len - 1

    def __getitem__(self, idx):
        i = idx * self.seq_len
        x = self.data[i : i + self.seq_len]
        y = self.data[i + 1 : i + 1 + self.seq_len]
        return x, y


def load_dataset(path: str, seq_len: int) -> TokenDataset:
    loaded_data = torch.load(path)
    return TokenDataset(loaded_data, seq_len)


def create_data_loader(
    path, batch_size: int = 32, seq_len=16, max_elements: int = None
) -> DataLoader:
    dataset = load_dataset(path, seq_len=seq_len)

    if max_elements is not None:
        dataset = Subset(dataset, list(range(max_elements)))

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    num_batches = len(data_loader)
    logger.info(f"DataLoader created: {num_batches} batches, batch_size={batch_size}")

    return data_loader


def save_model(path: str, model: nn.Module):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    logger.info(f"Model saved to {path}")


@torch.no_grad()
def eval(model: nn.Module, eval_data: DataLoader) -> float:
    model.eval()
    loss_sum = 0.0
    num_batches = 0

    for batch in eval_data:
        x, y = batch
        x, y = x.to(device=device), y.to(device=device)

        _, loss = model(x, y)
        loss_sum += loss.item()
        num_batches += 1

    return loss_sum / num_batches


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    training_data: DataLoader,
    eval_data: DataLoader,
    training_minutes: int = 5,
    log_freq: int = 1,
    save_freq: int = 10,
    batch_log_interval: int = 50,
    model_save_dir: str = "models/",
):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.70,
        patience=2,
        min_lr=5e-7,
    )

    logger.info(f"Starting training with lr={optimizer.param_groups[0]['lr']}")
    model.train()
    epoch = 0
    global_step = 0

    end_time = time.time() + 60 * training_minutes
    logger.info(f"Training will and at: {time.ctime(end_time)}")

    eval_loss = eval(model, eval_data)
    writer.add_scalars("Loss", {"Eval": eval_loss}, epoch)
    logger.info(f"First eval: {eval_loss}")

    while time.time() < end_time:
        loss_sum = 0.0
        num_batches = 0
        batch_start_time = time.time()

        model.train()
        for batch in training_data:
            x, y = batch
            x, y = x.to(device=device), y.to(device=device)

            optimizer.zero_grad()

            _, loss = model(x, y)
            loss_sum += loss.item()
            num_batches += 1

            loss.backward()
            optimizer.step()

            global_step += 1
            writer.add_scalar("Train/Loss_batch", loss.item(), global_step)

            if num_batches % batch_log_interval == 0:
                avg_loss = loss_sum / num_batches
                batch_time = time.time() - batch_start_time
                logger.info(
                    f"[Batch {num_batches}] Avg loss: {avg_loss:.4f} | "
                    f"Time for last {batch_log_interval} batches: {batch_time:.2f}s"
                )
                batch_start_time = time.time()

        epoch += 1

        epoch_loss = loss_sum / num_batches
        eval_loss = eval(model, eval_data)

        writer.add_scalars("Loss", {"Train": epoch_loss, "Eval": eval_loss}, epoch)

        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(eval_loss)
        new_lr = optimizer.param_groups[0]["lr"]

        writer.add_scalar("Train/LR", new_lr, epoch)

        if new_lr < old_lr:
            logger.info(f"LR reduced from {old_lr:.6f} to {new_lr:.6f}")

        if epoch % log_freq == 0:
            logger.info(
                f"Epoch: {epoch}, train_loss: {epoch_loss}, eval_loss: {eval_loss}"
            )

        if epoch % save_freq == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(model_save_dir, f"lstm_{epoch}_{timestamp}.pt")
            save_model(path, model)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(model_save_dir, f"lstm_{epoch}_{timestamp}.pt")
    save_model(path, model)


def plot_losses(train_losses, eval_losses, save_path="plots/training.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Training Loss", c="blue")
    plt.plot(eval_losses, label="Eval Loss", c="green")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss over Iterations")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    logger.info(f"Loss plots saved to {save_path}")
