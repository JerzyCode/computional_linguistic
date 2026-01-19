import argparse
from pathlib import Path

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Download Sudoku-Extreme dataset")
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--train-samples", type=int, default=None, help="Subsample training set"
    )
    parser.add_argument(
        "--test-samples", type=int, default=None, help="Subsample test set"
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading Sudoku-Extreme dataset from HuggingFace...")
    dataset = load_dataset("sapientinc/sudoku-extreme")

    train_data = dataset["train"]
    test_data = dataset["test"]

    print(f"Full dataset: {len(train_data)} train, {len(test_data)} test")

    if args.train_samples:
        train_data = train_data.shuffle(seed=42).select(
            range(min(args.train_samples, len(train_data)))
        )
        print(f"Subsampled to {len(train_data)} training examples")

    if args.test_samples:
        test_data = test_data.shuffle(seed=42).select(
            range(min(args.test_samples, len(test_data)))
        )
        print(f"Subsampled to {len(test_data)} test examples")

    train_path = args.output_dir / "train.csv"
    test_path = args.output_dir / "test.csv"

    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)

    print(f"Saved train set to {train_path}")
    print(f"Saved test set to {test_path}")

    print("\nSample puzzle:")
    sample = train_data[0]
    print(f"Dataset columns: {sample.keys()}")

    if "puzzle" in sample:
        puzzle_key = "puzzle"
        sol_key = "solution"
    elif "quizzes" in sample:
        puzzle_key = "quizzes"
        sol_key = "solutions"
    else:
        keys = list(sample.keys())
        puzzle_key = keys[0]
        sol_key = keys[1]

    puzzle = sample[puzzle_key]
    solution = sample[sol_key]

    print("Puzzle:")
    for i in range(9):
        row = puzzle[i * 9 : (i + 1) * 9]
        print(" ".join(c if c != "0" else "." for c in row))

    print("\nSolution:")
    for i in range(9):
        row = solution[i * 9 : (i + 1) * 9]
        print(" ".join(row))


if __name__ == "__main__":
    main()
