import json
import logging
import os
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchmetrics.text import Perplexity
from transformer import ModelType
from typing_extensions import Dict, Optional, Tuple


@dataclass(frozen=True)
class TrainingParameters:
    model_type: ModelType
    device: torch.device
    dtype: torch.dtype
    logger: logging.Logger
    epochs: int = 1
    lr: float = 1e-4
    eval_interval: int = 50
    batch_size: int = 64
    seq_len: int = 128

    def __str__(self) -> str:
        return (
            f"TrainingParameters(\n"
            f"  model_type={self.model_type.value},\n"
            f"  epochs={self.epochs},\n"
            f"  device={self.device},\n"
            f"  dtype={self.dtype},\n"
            f"  lr={self.lr},\n"
            f"  eval_interval={self.eval_interval}\n"
            f"  batch_size={self.batch_size}\n"
            f"  seq_len={self.seq_len}\n"
            f")"
        )


class Metrics:
    def __init__(self, results_dir: str, eval_freq: int) -> None:
        self.data = {}
        self.final_training_data = {}
        self.results_dir = results_dir
        self.eval_freq = eval_freq

        os.makedirs(self.results_dir, exist_ok=True)

    def add(
        self,
        step: int,
        train_loss: float,
        forward_duration: float,
        backward_duration: float,
        before_gpu_mem: float,
        forward_gpu_mem: float,
        backward_gpu_mem: float,
        peak_gpu_mem: float,
        eval_loss: Optional[float],
        eval_duration: Optional[float],
    ) -> None:
        data = {
            "train_loss": train_loss,
            "forward_duration": forward_duration, # per batch_size * 1
            "backward_duration": backward_duration, # per batch_size * 1
            "total_duration": forward_duration + backward_duration,
            "before_gpu_mem": before_gpu_mem,
            "forward_gpu_mem": forward_gpu_mem,
            "backward_gpu_mem": backward_gpu_mem,
            "peak_gpu_mem": peak_gpu_mem,
            "eval_loss": None,
            "eval_duration": None,
        }

        if eval_loss is not None:
            data["eval_loss"] = eval_loss

        if eval_duration is not None:
            data["eval_duration"] = eval_duration

        self.data[step] = data

    def add_final_training_info(
        self,
        avg_epoch_time: float,
        steps_performed: int,
        batch_size: int,
        final_perplexity: float,
        final_eval_loss: float,
        seq_len: int,
        lr: float,
        max_batch_size: int,
    ) -> None:
        avgs_gpu_mem = self._calculate_avg_gpu_mem()
        train_and_eval_avg_durations = self._calculate_eval_and_train_avg_duration()

        self.final_training_data = {
            "avg_epoch_time": avg_epoch_time, 
            "steps_performed": steps_performed,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "lr": lr,
            "final_perplexity": final_perplexity,
            "final_eval_loss": final_eval_loss,
            "avg_before_gpu_mem": round(avgs_gpu_mem["before_gpu_mem"], 2),
            "avg_forward_gpu_mem": round(avgs_gpu_mem["forward_gpu_mem"], 2),
            "avg_backward_gpu_mem": round(avgs_gpu_mem["backward_gpu_mem"], 2),
            "avg_peak_gpu_mem": round(avgs_gpu_mem["peak_gpu_mem"], 2),
            "avg_eval_duration_eval_loader": train_and_eval_avg_durations[
                "avg_eval_duration"
            ],
            "avg_train_duration_per_batch": train_and_eval_avg_durations[
                "avg_train_duration"
            ],
            "max_batch_size": max_batch_size,
        }

    def save_metrics(self) -> None:
        self._drop_to_csv()

        json.dump(
            self.final_training_data,
            open(self.results_dir + "/final_training_data.json", "w"),
        )

        self._save_training_plots()

    def _drop_to_csv(self) -> None:
        import pandas as pd

        df = pd.DataFrame.from_dict(self.data, orient="index")
        df.index.name = "step"
        df = df.reset_index()
        df.to_csv(f"{self.results_dir}/metrics.csv", index=False)

    def _save_training_plots(self) -> None:
        steps_with_eval = [
            step
            for step in self.data.keys()
            if self.data[step]["eval_loss"] is not None
        ]

        if not steps_with_eval:
            return

        train_losses = [self.data[step]["train_loss"] for step in steps_with_eval]
        eval_losses = [self.data[step]["eval_loss"] for step in steps_with_eval]

        plt.figure(figsize=(8, 5))
        plt.plot(steps_with_eval, train_losses, label="Train Loss", c="g")
        plt.plot(steps_with_eval, eval_losses, label="Eval Loss", c="r")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Training and Evaluation Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/loss_plot.png")
        plt.close()

    def _calculate_avg_gpu_mem(self) -> Dict[str, float]:
        total_steps = len(self.data)
        sums = {
            "before_gpu_mem": 0.0,
            "forward_gpu_mem": 0.0,
            "backward_gpu_mem": 0.0,
            "peak_gpu_mem": 0.0,
        }

        for step_data in self.data.values():
            sums["before_gpu_mem"] += step_data["before_gpu_mem"]
            sums["forward_gpu_mem"] += step_data["forward_gpu_mem"]
            sums["backward_gpu_mem"] += step_data["backward_gpu_mem"]
            sums["peak_gpu_mem"] += step_data["peak_gpu_mem"]

        avgs = {key: value / total_steps for key, value in sums.items()}

        return avgs

    def _calculate_eval_and_train_avg_duration(self) -> Dict[str, float]:
        total_eval_duration = 0.0
        total_train_duration = 0.0
        eval_steps = 0
        train_steps = 0

        for step_data in self.data.values():
            if step_data["eval_duration"] is not None:
                total_eval_duration += step_data["eval_duration"]
                eval_steps += 1
            total_train_duration += step_data["total_duration"]
            train_steps += 1

        avg_eval_duration = total_eval_duration / eval_steps if eval_steps > 0 else 0.0
        avg_train_duration = (
            total_train_duration / train_steps if train_steps > 0 else 0.0
        )

        return {
            "avg_eval_duration": avg_eval_duration,
            "avg_train_duration": avg_train_duration,
        }


class Trainer:
    def __init__(
        self,
        training_parameters: TrainingParameters,
        train_loader: DataLoader,
        eval_loader: DataLoader,
        max_batch_size: int,
    ) -> None:
        self.training_parameters = training_parameters
        self.logger = training_parameters.logger
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.max_batch_size = max_batch_size

        metrics_dir = self._get_save_dir()
        self.metrics = Metrics(results_dir=metrics_dir, eval_freq=training_parameters.eval_interval)

    def train(self, model: torch.nn.Module) -> Metrics:
        self.logger.info(
            f"Starting training for model_type: {self.training_parameters.model_type}"
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.training_parameters.lr
        )
        steps_performed = 0
        start_time = time.perf_counter()

        for _ in range(self.training_parameters.epochs):
            model.train()

            for batch in self.train_loader:
                X, y = batch
                X = X.to(device=self.training_parameters.device)
                y = y.to(device=self.training_parameters.device)

                step_data = self._perform_step(model, X, y, optimizer)
                steps_performed += 1

                eval_loss, eval_avg_duration_per_batch = None, None
                if (
                    steps_performed % self.training_parameters.eval_interval == 0
                    or steps_performed == 1 or steps_performed == len(self.train_loader)
                ):
                    eval_loss, eval_avg_duration_per_batch = self._perform_evaluation(
                        model
                    )
                    self.logger.info(
                        f"Step {steps_performed}: train_loss = {step_data['loss']}, eval_loss = {eval_loss}"
                    )

                self.metrics.add(
                    step=steps_performed,
                    train_loss=step_data["loss"],
                    forward_duration=step_data["forward_duration_per_batch"],
                    backward_duration=step_data["backward_duration_per_batch"],
                    before_gpu_mem=step_data["mem_before"],
                    forward_gpu_mem=step_data["mem_forward"],
                    backward_gpu_mem=step_data["mem_backward"],
                    peak_gpu_mem=step_data["mem_peak"],
                    eval_loss=eval_loss,
                    eval_duration=eval_avg_duration_per_batch,
                )

        end_time = time.perf_counter()
        training_duration = end_time - start_time
        self.logger.info(
            f"Training completed in {round(training_duration / 60, 2)} minutes"
        )

        final_eval_loss, _ = self._perform_evaluation(model)
        final_perplexity = self._get_final_perplexity(model)
        self.logger.info(
            f"Final Perplexity for model {self.training_parameters.model_type}: {final_perplexity}"
        )

        self.metrics.add_final_training_info(
            avg_epoch_time=round(training_duration / 60, 2),  # minutes
            steps_performed=steps_performed,
            batch_size=self.training_parameters.batch_size,
            final_perplexity=final_perplexity,
            final_eval_loss=final_eval_loss,
            seq_len=self.training_parameters.seq_len,
            lr=self.training_parameters.lr,
            max_batch_size=self.max_batch_size,
        )

        return self.metrics

    def _perform_step(self, model, X, y, optimizer):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        mem_before = torch.cuda.memory_allocated()

        forward_start_time = time.perf_counter()
        _, loss = model(X, y)
        forward_end_time = time.perf_counter()
        forward_duration = forward_end_time - forward_start_time

        torch.cuda.synchronize()
        mem_forward = torch.cuda.memory_allocated()

        backward_start_time = time.perf_counter()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        backward_end_time = time.perf_counter()
        backward_duration = backward_end_time - backward_start_time
        torch.cuda.synchronize()

        mem_backward = torch.cuda.memory_allocated()
        mem_peak = torch.cuda.max_memory_allocated()

        step_duration = forward_duration + backward_duration

        return {
            "loss": loss.item(),
            "forward_duration_per_batch": forward_duration,
            "backward_duration_per_batch": backward_duration,
            "step_duration_per_batch": step_duration,
            "mem_before": mem_before,
            "mem_forward": mem_forward,
            "mem_backward": mem_backward,
            "mem_peak": mem_peak,
        }

    def _perform_evaluation(self, model) -> Tuple[float, float]:
        model.eval()
        total_loss = 0.0
        num_batches = 0

        start_time = time.perf_counter()
        with torch.no_grad():
            for batch in self.eval_loader:
                X, y = batch
                X = X.to(device=self.training_parameters.device)
                y = y.to(device=self.training_parameters.device)

                _, loss = model(X, y)
                total_loss += loss.item()
                num_batches += 1

        end_time = time.perf_counter()
        evaluation_duration = end_time - start_time # per batch_size * len(eval_loader)
        evalution_duration_avg = evaluation_duration / num_batches

        avg_loss = total_loss / num_batches
        return avg_loss, evalution_duration_avg

    def _get_final_perplexity(self, model) -> float:
        batch_count = 0
        model.eval()

        perplexity = Perplexity().to(self.training_parameters.device)

        with torch.no_grad():
            for X, y in self.eval_loader:
                X = X.to(device=self.training_parameters.device)
                y = y.to(device=self.training_parameters.device)

                logits, _ = model(X)
                perplexity.update(logits, y)
                batch_count += 1

        return perplexity.compute().item() / batch_count

    def _get_save_dir(self) -> str:
        os.makedirs("results", exist_ok=True)

        path = f"results/{self.training_parameters.model_type.value}"
        return path

    @staticmethod
    def format_mb(x) -> str:
        return round(x / (1024 * 1024), 2)
