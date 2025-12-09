import logging
import os
import time
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from torchmetrics.text import Perplexity
from transformer import ModelType
from typing_extensions import Optional, Tuple


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
            f")"
        )


class Metrics:
    def __init__(self, save_path: str) -> None:
        self.data = {}
        self.save_path = save_path

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
            "forward_duration": forward_duration,
            "backward_duration": backward_duration,
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

    def drop_to_csv(self) -> None:
        import pandas as pd

        df = pd.DataFrame.from_dict(self.data, orient="index")
        df.index.name = "step"
        df = df.reset_index()
        df.to_csv(self.save_path, index=False)


class Trainer:
    def __init__(
        self,
        training_parameters: TrainingParameters,
        train_loader: DataLoader,
        eval_loader: DataLoader,
    ) -> None:
        self.training_parameters = training_parameters
        self.logger = training_parameters.logger
        self.train_loader = train_loader
        self.eval_loader = eval_loader

        metrics_path = self._get_save_path()
        self.metrics = Metrics(save_path=metrics_path)

    def train(self, model: torch.nn.Module):
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
                x = X.to(dtype=self.training_parameters.dtype)
                y = y.to(device=self.training_parameters.device)
                y = y.to(dtype=self.training_parameters.dtype)

                step_data = self._perform_step(model, x, y, optimizer)
                steps_performed += 1

                eval_loss, eval_avg_duration_per_batch = None, None
                if (
                    steps_performed % self.training_parameters.eval_interval == 0
                    or steps_performed == 1
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

        final_perplexity = self._get_final_perplexity(model)
        self.logger.info(
            f"Final Perplexity for model {self.training_parameters.model_type}: {final_perplexity}"
        )
        self.metrics.drop_to_csv()

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
                X = X.to(dtype=self.training_parameters.dtype)
                y = y.to(device=self.training_parameters.device)
                y = y.to(dtype=self.training_parameters.dtype)

                _, loss = model(X, y)
                total_loss += loss.item()
                num_batches += 1

        end_time = time.perf_counter()
        evaluation_duration = end_time - start_time
        evalution_duration_avg = evaluation_duration / num_batches

        avg_loss = total_loss / num_batches
        return avg_loss, evalution_duration_avg

    def _get_final_perplexity(self, model) -> float:
        batch_count = 0
        model.eval()

        perplexity = Perplexity()

        with torch.no_grad():
            for X, y in self.eval_loader:
                X = X.to(device=self.training_parameters.device)
                X = X.to(dtype=self.training_parameters.dtype)
                y = y.to(device=self.training_parameters.device)
                y = y.to(dtype=self.training_parameters.dtype)

                logits, _ = model(X)
                perplexity.update(logits, y)
                batch_count += 1

        return perplexity.compute().item() / batch_count

    def _get_save_path(self) -> str:
        os.makedirs("results", exist_ok=True)

        path = f"results/{self.training_parameters.model_type.value}_metrics.csv"
        return path

    @staticmethod
    def format_mb(x) -> str:
        return round(x / (1024 * 1024), 2)
