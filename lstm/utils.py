import csv
import os
import time

import torch

from lstm.lstm_model import Lstm


class ModelSaver:
    def __init__(self, save_dir: str, save_frequency: int = 5):
        self.save_dir = save_dir
        self.save_frequency = save_frequency
        os.makedirs(save_dir, exist_ok=True)

    def save_model_checkpoint(self, model: Lstm, epoch: int):
        if epoch % self.save_frequency:
            model_path = os.path.join(
                self.save_dir, f"lstm_checkpoint_epoch_{epoch:04d}.pt"
            )
            torch.save(model.state_dict(), model_path)
            print(f"Model saved at: {model_path}")


class EpochTracker:
    def __init__(self, save_csv_path: str):
        self.save_csv_path = save_csv_path
        self.start_time = None

        dir_path = os.path.dirname(save_csv_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

        if not os.path.exists(save_csv_path):
            with open(save_csv_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "train_loss", "eval_loss", "duration_sec"])

    def start_epoch(self):
        self.start_time = time.time()

    def end_epoch(self):
        if self.start_time is None:
            raise RuntimeError("start_epoch() must be called before end_epoch()")
        self.end_time = time.time()

    def insert_stats(self, epoch, train_loss, eval_loss):
        if self.start_time is None:
            raise RuntimeError(
                "No epoch_time provided and start_epoch() was not called"
            )

        epoch_duration = self.end_time - self.start_time

        with open(self.save_csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, eval_loss, epoch_duration])
