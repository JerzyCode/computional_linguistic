import os
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

# Add lab6_trm to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "lab6_trm"))

from train import SudokuDataset
from trm import TRM

CONFIG = {
    "batch_size": 768,
    "data_dir": "data",
    "device": "cpu",
    "eval_samples": 5,
    "H_cycles": 3,
    "hidden_size": 384,
    "L_cycles": 6,
    "lr": 0.00015,
    "max_steps": 8,
    "n_layers": 2,
    "num_workers": 8,
    "save_path": "model",
}


class EvalSudokuDataset(SudokuDataset):
    def __getitem__(self, idx):
        puzzle, solution = super().__getitem__(idx)
        row = self.df.iloc[idx]
        rating = row["rating"] if "rating" in self.df.columns else 0.0
        return puzzle, solution, rating


def check_sudoku_validity(grid: torch.Tensor):
    grid_np = grid.cpu().numpy().reshape(9, 9)
    conflict_mask = np.zeros((9, 9), dtype=bool)
    is_valid = True

    if (grid_np == 0).any():
        is_valid = False

    def mark_duplicates(arr, indices):
        unique, counts = np.unique(arr, return_counts=True)
        duplicates = unique[counts > 1]
        for val in duplicates:
            if val == 0:
                continue  # 0s treated as empty, skip for conflict check logic
            for idx in indices:
                r, c = idx
                if grid_np[r, c] == val:
                    conflict_mask[r, c] = True

    # Rows
    for i in range(9):
        mark_duplicates(grid_np[i, :], [(i, j) for j in range(9)])

    # Cols
    for j in range(9):
        mark_duplicates(grid_np[:, j], [(i, j) for i in range(9)])

    # 3x3 Boxes
    for bi in range(3):
        for bj in range(3):
            indices = []
            vals = []
            for i in range(3):
                for j in range(3):
                    r, c = bi * 3 + i, bj * 3 + j
                    indices.append((r, c))
                    vals.append(grid_np[r, c])
            mark_duplicates(vals, indices)

    if conflict_mask.any():
        is_valid = False

    return is_valid, torch.from_numpy(conflict_mask)


def print_sudoku_grid(tensor: torch.Tensor, title: str = ""):
    grid = tensor.cpu().numpy().reshape(9, 9).astype(int)
    if title:
        print(f"\n{title}")
    print("+" + "-" * 7 + "+" + "-" * 7 + "+" + "-" * 7 + "+")
    for i in range(9):
        row = "|"
        for j in range(9):
            val = grid[i, j]
            row += f" {val if val != 0 else '.'}"
            if (j + 1) % 3 == 0:
                row += " |"
        print(row)
        if (i + 1) % 3 == 0:
            print("+" + "-" * 7 + "+" + "-" * 7 + "+" + "-" * 7 + "+")


def print_sample(sample: dict, label: str = ""):
    if label:
        print(f"\n{'=' * 30} {label} {'=' * 30}")
    print_sudoku_grid(sample["puzzle_tensor"], "Puzzle")

    if "solution_tensor" in sample:
        print_sudoku_grid(sample["solution_tensor"], "Solution")
    if "prediction_tensor" in sample:
        print_sudoku_grid(sample["prediction_tensor"], "Prediction")


def load_dataset(
    path: str, samples: int, stratify: bool = False
) -> torch.utils.data.Dataset:
    full_dataset = EvalSudokuDataset(path)

    if not stratify:
        if len(full_dataset) > samples:
            indices = np.random.choice(len(full_dataset), samples, replace=False)
            return Subset(full_dataset, indices)
        return full_dataset

    df = full_dataset.df
    if "rating" not in df.columns:
        print("Warning: 'rating' column not found. Falling back to random sampling.")
        if len(full_dataset) > samples:
            indices = np.random.choice(len(full_dataset), samples, replace=False)
            return Subset(full_dataset, indices)
        return full_dataset

    unique_ratings = df["rating"].unique()
    n_ratings = len(unique_ratings)
    if n_ratings > 0:
        samples_per_rating = samples // n_ratings
    else:
        samples_per_rating = 0

    selected_indices = []

    for rating in unique_ratings:
        rating_indices = df[df["rating"] == rating].index.tolist()
        n_available = len(rating_indices)

        if n_available > samples_per_rating:
            selected = np.random.choice(
                rating_indices, samples_per_rating, replace=False
            )
        else:
            selected = rating_indices
        selected_indices.extend(selected)

    if len(selected_indices) < samples:
        remaining_needed = samples - len(selected_indices)
        all_indices = set(df.index)
        used_indices = set(selected_indices)
        available_indices = list(all_indices - used_indices)

        if len(available_indices) >= remaining_needed:
            extra_indices = np.random.choice(
                available_indices, remaining_needed, replace=False
            )
            selected_indices.extend(extra_indices)
        else:
            selected_indices.extend(available_indices)

    np.random.shuffle(selected_indices)

    return Subset(full_dataset, selected_indices)


def load_model(path: str) -> TRM:
    device = torch.device(CONFIG["device"] if torch.cuda.is_available() else "cpu")

    model = TRM(
        vocab_size=10,
        seq_len=81,
        hidden_size=CONFIG["hidden_size"],
        n_layers=CONFIG["n_layers"],
        H_cycles=CONFIG["H_cycles"],
        L_cycles=CONFIG["L_cycles"],
    ).to(device)

    print(f"Loading model from {path}")
    checkpoint = torch.load(path, map_location=device)

    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    return model


def evaluate(
    model: TRM, dataloader: DataLoader, device: torch.device, max_steps: int = 16
) -> List[Dict]:
    model.eval()

    results = []

    with torch.no_grad():
        for i, (puzzle, solution, rating) in enumerate(dataloader):
            puzzle = puzzle.to(device)
            solution = solution.to(device)

            batch_size = puzzle.size(0)

            predictions = model.solve(puzzle, max_steps=max_steps)

            cell_correct = (predictions == solution).float()

            puzzle_correct = cell_correct.all(dim=1).float()

            mask = puzzle == 0

            for idx in range(batch_size):
                p_acc = puzzle_correct[idx].item()
                c_acc = cell_correct[idx].mean().item()

                is_valid, conflict_mask = check_sudoku_validity(predictions[idx])

                r = rating[idx]
                if isinstance(r, torch.Tensor):
                    r = r.item()

                sample_mask = mask[idx]
                if sample_mask.sum() > 0:
                    u_acc = cell_correct[idx][sample_mask].mean().item()
                else:
                    u_acc = 1.0

                results.append(
                    {
                        "rating": r,
                        "puzzle_acc": p_acc,
                        "cell_acc": c_acc,
                        "unsolved_acc": u_acc,
                        "valid_puzzle_acc": 1.0 if is_valid else 0.0,
                        "conflict_mask": conflict_mask,
                        "puzzle_tensor": puzzle[idx].cpu(),
                        "solution_tensor": solution[idx].cpu(),
                        "prediction_tensor": predictions[idx].cpu(),
                    }
                )

            if i % 10 == 0:
                current_p_acc = sum([x["puzzle_acc"] for x in results]) / len(results)
                print(f"Step {i}/{max_steps}: Avg Puzzle Acc: {current_p_acc:.4f}")

    return results


def plot_and_save_sudoku_grid(ax, grid, title, conflict_mask=None):
    ax.imshow(grid, cmap="Blues", vmin=0, vmax=9)
    ax.set_title(title)
    ax.axis("off")

    for i in range(10):
        lw = 2 if i % 3 == 0 else 0.5
        ax.axhline(i - 0.5, color="black", linewidth=lw)
        ax.axvline(i - 0.5, color="black", linewidth=lw)

    grid_np = grid if isinstance(grid, np.ndarray) else grid.numpy()
    if conflict_mask is not None:
        mask_np = (
            conflict_mask
            if isinstance(conflict_mask, np.ndarray)
            else conflict_mask.numpy()
        )
    else:
        mask_np = np.zeros_like(grid_np, dtype=bool)

    for i in range(9):
        for j in range(9):
            val = int(grid_np[i, j])
            if val != 0:
                color = "red" if mask_np[i, j] else "black"
                weight = "bold" if mask_np[i, j] else "normal"
                ax.text(
                    j,
                    i,
                    str(val),
                    ha="center",
                    va="center",
                    fontsize=12,
                    color=color,
                    fontweight=weight,
                )


def save_single_sudoku_plot(
    puzzle, true_solution, prediction, conflict_mask, title, path
):
    sudoku_np = puzzle.numpy().reshape(9, 9)
    true_np = true_solution.numpy().reshape(9, 9)
    pred_np = prediction.numpy().reshape(9, 9)
    conflict_np = conflict_mask.numpy().reshape(9, 9)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plot_and_save_sudoku_grid(axes[0], sudoku_np, "Input Puzzle")
    plot_and_save_sudoku_grid(axes[1], true_np, "Ground Truth")
    plot_and_save_sudoku_grid(
        axes[2], pred_np, "Prediction (Conflicts Red)", conflict_mask=conflict_np
    )

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Saved visualization: {path}")


def analyze_and_plot(results: List[Dict]):
    if not results:
        print("No results to analyze.")
        return

    df = pd.DataFrame(results)

    overall_puzzle_acc = df["puzzle_acc"].mean()
    overall_valid_acc = df["valid_puzzle_acc"].mean()
    overall_cell_acc = df["cell_acc"].mean()
    print("\n" + "=" * 40)
    print("Overall Metrics:")
    print(f"Puzzle Accuracy (Exact): {overall_puzzle_acc:.4f}")
    print(f"Puzzle Validity:         {overall_valid_acc:.4f}")
    print(f"Cell Accuracy:           {overall_cell_acc:.4f}")
    print("=" * 40 + "\n")

    rating_stats = (
        df.groupby("rating")[["puzzle_acc", "valid_puzzle_acc"]].mean().reset_index()
    )
    rating_stats = rating_stats.sort_values("rating")

    plt.figure(figsize=(10, 6))
    plt.plot(
        rating_stats["rating"],
        rating_stats["puzzle_acc"],
        marker="o",
        linestyle="-",
        label="Exact Match",
    )
    plt.plot(
        rating_stats["rating"],
        rating_stats["valid_puzzle_acc"],
        marker="x",
        linestyle="--",
        label="Valid Puzzle",
    )
    plt.title("Average Accuracy vs Rating")
    plt.xlabel("Rating")
    plt.ylabel("Avg Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig("evaluation_results.png")
    plt.close()
    print("Saved line plot to evaluation_results.png")

    min_rating = df["rating"].min()
    max_rating = df["rating"].max()
    median_rating = df["rating"].median()

    unique_ratings = sorted(df["rating"].unique())
    if len(unique_ratings) > 0:
        idx_med = (np.abs(np.array(unique_ratings) - median_rating)).argmin()
        actual_median_rating = unique_ratings[idx_med]
    else:
        actual_median_rating = median_rating

    def get_sample(target_rating):
        subset = df[df["rating"] == target_rating]
        if len(subset) == 0:
            return None
        return subset.iloc[0]

    sample_low = get_sample(min_rating)
    sample_med = get_sample(actual_median_rating)
    sample_high = get_sample(max_rating)

    print_sample(sample_low, "Low Rating")
    print_sample(sample_med, "Median Rating")
    print_sample(sample_high, "High Rating")

    if sample_low is not None:
        save_single_sudoku_plot(
            sample_low["puzzle_tensor"],
            sample_low["solution_tensor"],
            sample_low["prediction_tensor"],
            sample_low["conflict_mask"],
            f"Sudoku (Low Rating {min_rating})",
            "sudoku_easy.png",
        )

    if sample_med is not None:
        save_single_sudoku_plot(
            sample_med["puzzle_tensor"],
            sample_med["solution_tensor"],
            sample_med["prediction_tensor"],
            sample_med["conflict_mask"],
            f"Sudoku (Median Rating {actual_median_rating})",
            "sudoku_med.png",
        )

    if sample_high is not None:
        save_single_sudoku_plot(
            sample_high["puzzle_tensor"],
            sample_high["solution_tensor"],
            sample_high["prediction_tensor"],
            sample_high["conflict_mask"],
            f"Sudoku (High Rating {max_rating})",
            "sudoku_hard.png",
        )


if __name__ == "__main__":
    device = torch.device(CONFIG["device"] if torch.cuda.is_available() else "cpu")
    data_dir = Path(CONFIG["data_dir"])

    test_csv = data_dir / "test.csv"
    if not test_csv.exists():
        pass

    print(f"Using device: {device}")

    print("Loading dataset...")
    test_dataset = load_dataset(str(test_csv), CONFIG["eval_samples"], stratify=True)
    print(f"Dataset loaded. Model will be evaluated on {len(test_dataset)} samples.")

    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
    )

    model_path = Path(CONFIG["save_path"]) / "final_model.pt"
    if not model_path.exists():
        model_path = Path(CONFIG["save_path"]) / "best_model.pt"

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        exit(1)

    model = load_model(str(model_path))

    print("Starting evaluation...")
    results = evaluate(model, test_loader, device, max_steps=CONFIG["max_steps"])

    analyze_and_plot(results)
