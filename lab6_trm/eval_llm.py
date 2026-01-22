from typing import Dict, Optional, Tuple

import numpy as np
import torch
from eval import check_sudoku_validity, print_sample, save_single_sudoku_plot
from train import SudokuDataset


def parse_sudoku_string(s: str) -> torch.Tensor:
    s = str(s).replace(".", "0")
    s = "".join(s.split())

    if len(s) != 81:
        digits = [c for c in s if c.isdigit()]
        if len(digits) == 81:
            vals = [int(c) for c in digits]
            return torch.tensor(vals, dtype=torch.long).reshape(9, 9)
        raise ValueError(f"Sudoku string must be 81 characters long, got {len(s)}")

    vals = [int(c) for c in s]
    return torch.tensor(vals, dtype=torch.long).reshape(9, 9)


prompt = """
You are an expert Sudoku solver. Your task is to solve the given Sudoku puzzle correctly and completely by strictly following the rules below.

Rules:
1. The puzzle is a 9x9 grid, where each row, column, and 3x3 subgrid must contain the digits 1-9 exactly once.
2. The puzzle is given as a string of 81 characters, where each character represents a cell in the grid.
3. The characters '0' or '.' represent empty cells that need to be filled.
4. The characters '1'-'9' represent the digits that are already placed in the grid.
5. The puzzle is guaranteed to have a unique solution.
6. You must solve the puzzle by reasoning step by step, showing your work and explaining your thought process.
7. You must not make any assumptions or guesses about the solution.
8. You must not use any shortcuts or strategies that are not based on logical reasoning.

Sudoku:

+-------+-------+-------+
| 4 . . | 6 . . | 5 . . |
| . 6 . | . . 9 | . . . |
| . . 8 | . 2 . | . 3 6 |
+-------+-------+-------+
| . . . | 7 1 . | 3 . . |
| . . . | . . 2 | . 7 . |
| . . 2 | . . . | 1 . 8 |
+-------+-------+-------+
| 9 . . | 2 . . | . . . |
| . 5 . | . . 4 | . . . |
| . . 7 | . 3 . | . 1 . |
+-------+-------+-------+
"""

simple_sudoku = (
    "...5..83...27.4...14..96...39.8.5.21..761...82....975.9.3...5.2...9..6.7.1..4..83"
)
simple_sudoku_solution = (
    "769521834532784169148396275396875421457612398281439756973168542824953617615247983"
)

simple_gemini_solution = (
    "769521834532784169148396275394875621857612348281439756973168542425953617615247983"
)


simple_sample = {
    "puzzle_tensor": parse_sudoku_string(simple_sudoku),
    "solution_tensor": parse_sudoku_string(simple_sudoku_solution),
    "prediction_tensor": parse_sudoku_string(simple_gemini_solution),
}

hard_sudoku = (
    "..3.2.5....5.7.2...9.65.....3..........8....4..5.3.7....12...753.....1..5...1..2."
)
hard_sudoku_solution = (
    "873129546615478293294653817139745682752861934468932751941286375327594168586317429"
)
hard_gemini_solution = (
    "473128596165973248892654317734561982219847635685239751941286375327495168586317429"
)

hard_sample = {
    "puzzle_tensor": parse_sudoku_string(hard_sudoku),
    "solution_tensor": parse_sudoku_string(hard_sudoku_solution),
    "prediction_tensor": parse_sudoku_string(hard_gemini_solution),
}

medium_sudoku = (
    "4..6..5...6...9.....8.2..36...71.3.......2.7...2...1.89..2......5...4.....7.3..1."
)
medium_sudoku_solution = (
    "421683597763459281598127436649718352815342679372965148934271865156894723287536914"
)
medium_gemini_solution = (
    "423687591765349821198521736589716342613492575742953168934275681351864927287931415"
)

medium_sample = {
    "puzzle_tensor": parse_sudoku_string(medium_sudoku),
    "solution_tensor": parse_sudoku_string(medium_sudoku_solution),
    "prediction_tensor": parse_sudoku_string(medium_gemini_solution),
}


class EvalSudokuDataset(SudokuDataset):
    def __getitem__(self, idx):
        puzzle, solution = super().__getitem__(idx)
        row = self.df.iloc[idx]
        rating = row["rating"] if "rating" in self.df.columns else 0.0
        return puzzle, solution, rating


def get_representative_puzzles(
    data_path: str = "data/test.csv",
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, float]]:
    dataset = EvalSudokuDataset(data_path)
    df = dataset.df

    if "rating" not in df.columns:
        print("Warning: 'rating' column missing in dataset.")
        return {}

    min_rating = df["rating"].min()
    max_rating = df["rating"].max()
    median_rating = df["rating"].median()

    unique_ratings = sorted(df["rating"].unique())
    if len(unique_ratings) == 0:
        return {}

    idx_med = (np.abs(np.array(unique_ratings) - median_rating)).argmin()
    actual_median_rating = unique_ratings[idx_med]

    def get_sample(target_rating) -> Optional[Tuple[torch.Tensor, torch.Tensor, float]]:
        matches = df.index[df["rating"] == target_rating].tolist()
        if not matches:
            return None
        label_idx = matches[0]
        pos_idx = df.index.get_loc(label_idx)
        return dataset[pos_idx]

    return {
        "easy": get_sample(min_rating),
        "medium": get_sample(actual_median_rating),
        "hard": get_sample(max_rating),
    }


if __name__ == "__main__":
    is_valid, conflict_mask = check_sudoku_validity(medium_sample["prediction_tensor"])
    print(f"Gemini solution valid: {is_valid}")

    medium_sample["conflict_mask"] = conflict_mask

    print_sample(medium_sample, "Medium Sudoku (Gemini)")

    save_single_sudoku_plot(
        medium_sample["puzzle_tensor"],
        medium_sample["solution_tensor"],
        medium_sample["prediction_tensor"],
        medium_sample["conflict_mask"],
        "Medium Sudoku - Gemini Solution",
        "sudoku_gemini_medium.png",
    )
