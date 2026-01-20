import argparse
import copy
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from lion import Lion
from tango import Tango
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from trm import TRM, count_parameters


class SudokuDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        cols = list(self.df.columns)
        print(f"CSV columns: {cols}")

        self.puzzle_col = cols[1]
        self.solution_col = cols[2]
        print(f"Using: puzzle='{self.puzzle_col}', solution='{self.solution_col}'")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        puzzle_str = str(row[self.puzzle_col]).replace(".", "0")
        solution_str = str(row[self.solution_col])

        puzzle = torch.tensor([int(c) for c in puzzle_str], dtype=torch.long)
        solution = torch.tensor([int(c) for c in solution_str], dtype=torch.long)
        return puzzle, solution


def update_ema(model, ema_model, decay=0.999):
    with torch.no_grad():
        for param, ema_param in zip(model.parameters(), ema_model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)


def compute_accuracy(logits, y_true):
    predictions = logits.argmax(dim=-1)

    cell_correct = (predictions == y_true).float()
    cell_accuracy = cell_correct.mean().item()

    puzzle_correct = cell_correct.all(dim=1).float()
    puzzle_accuracy = puzzle_correct.mean().item()

    return cell_accuracy, puzzle_accuracy


def train_epoch(
    model,
    ema_model,
    dataloader,
    optimizer,
    device,
    max_steps=16,
    use_act=True,
    use_wandb=False,
    global_step=0,
    test_loader=None,
    eval_every=500,
    save_every=1000,
    save_path=None,
    best_puzzle_acc=0.0,
):
    model.train()
    total_loss = 0
    total_cell_acc = 0
    total_puzzle_acc = 0
    total_steps = 0
    total_supervision_steps = 0

    progress = tqdm(dataloader, desc="Training")
    for puzzle, solution in progress:
        puzzle = puzzle.to(device)
        solution = solution.to(device)
        batch_size = puzzle.size(0)

        carry = model.get_initial_carry(batch_size)
        halted = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for step in range(max_steps):
            carry, logits, q_halt = model(puzzle, carry)

            active_mask = ~halted
            if not active_mask.any():
                break

            loss = F.cross_entropy(
                logits[active_mask].view(-1, 10), solution[active_mask].view(-1)
            )

            if use_act:
                correct_mask = (logits.argmax(dim=-1) == solution).all(dim=1).float()
                act_loss = F.binary_cross_entropy_with_logits(
                    q_halt[active_mask], correct_mask[active_mask]
                )
                loss = loss + act_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if ema_model is not None:
                update_ema(model, ema_model)

            total_loss += loss.item()
            cell_acc, puzzle_acc = compute_accuracy(logits, solution)
            total_cell_acc += cell_acc
            total_puzzle_acc += puzzle_acc
            total_steps += 1
            total_supervision_steps += 1

            global_step += 1

            if use_act:
                newly_halted = (q_halt > 0) & ~halted
                halted = halted | newly_halted

                if halted.all():
                    break

        progress.set_postfix(
            {
                "loss": total_loss / total_steps if total_steps > 0 else 0,
                "cell": total_cell_acc / total_steps if total_steps > 0 else 0,
                "puzzle": total_puzzle_acc / total_steps if total_steps > 0 else 0,
            }
        )

        if use_wandb:
            wandb.log(
                {
                    "step": global_step,
                    "step/loss": total_loss / total_steps if total_steps > 0 else 0,
                    "step/cell_acc": total_cell_acc / total_steps
                    if total_steps > 0
                    else 0,
                    "step/puzzle_acc": total_puzzle_acc / total_steps
                    if total_steps > 0
                    else 0,
                },
                step=global_step,
            )

        if test_loader is not None and global_step % eval_every == 0:
            model.eval()
            eval_metrics = evaluate(ema_model, test_loader, device, max_steps=max_steps)
            print(
                f"\n[Step {global_step}] Eval - Cell: {eval_metrics['cell_acc']:.4f}, "
                f"Puzzle: {eval_metrics['puzzle_acc']:.4f}"
            )
            if use_wandb:
                wandb.log(
                    {
                        "eval/cell_acc": eval_metrics["cell_acc"],
                        "eval/puzzle_acc": eval_metrics["puzzle_acc"],
                    },
                    step=global_step,
                )
            if eval_metrics["puzzle_acc"] > best_puzzle_acc:
                best_puzzle_acc = eval_metrics["puzzle_acc"]
                if save_path is not None:
                    torch.save(
                        {
                            "global_step": global_step,
                            "model_state_dict": model.state_dict(),
                            "ema_model_state_dict": ema_model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "best_puzzle_acc": best_puzzle_acc,
                        },
                        save_path / "best_model.pt",
                    )
                    print(f"New best model saved! Puzzle Acc: {best_puzzle_acc:.4f}")
            model.train()

        if save_path is not None and global_step % save_every == 0:
            torch.save(
                {
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "ema_model_state_dict": ema_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_puzzle_acc": best_puzzle_acc,
                },
                save_path / f"checkpoint_step_{global_step}.pt",
            )
            print(f"\n[Step {global_step}] Checkpoint saved")

    avg_sup_steps = total_supervision_steps / len(dataloader)
    return {
        "loss": total_loss / total_steps if total_steps > 0 else 0,
        "cell_acc": total_cell_acc / total_steps if total_steps > 0 else 0,
        "puzzle_acc": total_puzzle_acc / total_steps if total_steps > 0 else 0,
        "avg_sup_steps": avg_sup_steps,
        "global_step": global_step,
        "best_puzzle_acc": best_puzzle_acc,
    }


@torch.no_grad()
def evaluate(model, dataloader, device, max_steps=16):
    model.eval()
    total_cell_acc = 0
    total_puzzle_acc = 0
    total_samples = 0

    for puzzle, solution in tqdm(dataloader, desc="Evaluating"):
        puzzle = puzzle.to(device)
        solution = solution.to(device)

        predictions = model.solve(puzzle, max_steps=max_steps)

        cell_correct = (predictions == solution).float()
        cell_accuracy = cell_correct.mean().item()

        puzzle_correct = cell_correct.all(dim=1).float()
        puzzle_accuracy = puzzle_correct.mean().item()

        total_cell_acc += cell_accuracy * puzzle.size(0)
        total_puzzle_acc += puzzle_accuracy * puzzle.size(0)
        total_samples += puzzle.size(0)

    return {
        "cell_acc": total_cell_acc / total_samples,
        "puzzle_acc": total_puzzle_acc / total_samples,
    }


def visualize_prediction(puzzle, solution, prediction):
    print("\nPuzzle:")
    for i in range(9):
        row = puzzle[i * 9 : (i + 1) * 9]
        print(" ".join(str(c.item()) if c.item() != 0 else "." for c in row))

    print("\nGround Truth:")
    for i in range(9):
        row = solution[i * 9 : (i + 1) * 9]
        print(" ".join(str(c.item()) for c in row))

    print("\nPrediction:")
    for i in range(9):
        row = prediction[i * 9 : (i + 1) * 9]
        print(" ".join(str(c.item()) for c in row))

    correct = (prediction == solution).all().item()
    print(f"\nCorrect: {correct}")


def main():
    parser = argparse.ArgumentParser(description="Train TRM on Sudoku-Extreme")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument(
        "--optimizer",
        type=str,
        default="lion",
        choices=["lion", "tango"],
        help="Optimizer to use (lion: lr=1e-4, tango: lr=1.0)",
    )
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument(
        "--H-cycles", type=int, default=2, help="T in paper (outer recursions)"
    )
    parser.add_argument(
        "--L-cycles", type=int, default=3, help="n in paper (inner recursions)"
    )
    parser.add_argument(
        "--max-steps", type=int, default=16, help="Max supervision steps"
    )
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--eval-every", type=int, default=500, help="Evaluate every N steps"
    )
    parser.add_argument(
        "--save-every", type=int, default=1000, help="Save checkpoint every N steps"
    )
    parser.add_argument("--save-path", type=Path, default=Path("checkpoints"))
    parser.add_argument(
        "--no-act", action="store_true", help="Disable ACT early stopping"
    )

    parser.add_argument(
        "--eval-samples", type=int, default=1000, help="Number of test samples for eval"
    )
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb-project", type=str, default="trm-sudoku")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    args.save_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {args.data_dir}")
    train_dataset = SudokuDataset(args.data_dir / "train.csv")
    test_dataset = SudokuDataset(args.data_dir / "test.csv")

    if args.eval_samples and args.eval_samples < len(test_dataset):
        indices = torch.randperm(len(test_dataset))[: args.eval_samples].tolist()
        test_dataset = torch.utils.data.Subset(test_dataset, indices)

    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = TRM(
        vocab_size=10,
        seq_len=81,
        hidden_size=args.hidden_size,
        n_layers=args.n_layers,
        H_cycles=args.H_cycles,
        L_cycles=args.L_cycles,
    ).to(args.device)

    ema_model = copy.deepcopy(model)
    for param in ema_model.parameters():
        param.requires_grad = False

    print(f"Model parameters: {count_parameters(model):,}")
    print(f"H_cycles (T): {args.H_cycles}, L_cycles (n): {args.L_cycles}")
    print(
        f"Effective depth per step: {args.n_layers * (args.L_cycles * args.H_cycles + args.H_cycles)}"
    )

    if args.optimizer == "lion":
        lr = args.lr if args.lr != 1e-4 else 1e-4
        optimizer = Lion(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.99),
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "tango":
        lr = args.lr if args.lr != 1e-4 else 1.0
        optimizer = Tango(
            model.parameters(),
            lr=lr,
        )

    print(f"Optimizer: {args.optimizer}, LR: {lr}")

    best_puzzle_acc = 0

    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )
        wandb.watch(model, log="all", log_freq=100)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'=' * 60}")

        train_metrics = train_epoch(
            model,
            ema_model,
            train_loader,
            optimizer,
            args.device,
            max_steps=args.max_steps,
            use_act=not args.no_act,
            use_wandb=args.wandb,
            global_step=global_step,
            test_loader=test_loader,
            eval_every=args.eval_every,
            save_every=args.save_every,
            save_path=args.save_path,
            best_puzzle_acc=best_puzzle_acc,
        )
        global_step = train_metrics["global_step"]
        best_puzzle_acc = train_metrics["best_puzzle_acc"]

        print(
            f"\nTrain - Loss: {train_metrics['loss']:.4f}, "
            f"Cell Acc: {train_metrics['cell_acc']:.4f}, "
            f"Puzzle Acc: {train_metrics['puzzle_acc']:.4f}, "
            f"Avg Steps: {train_metrics['avg_sup_steps']:.2f}"
        )

        if args.wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "epoch/train_loss": train_metrics["loss"],
                    "epoch/train_cell_acc": train_metrics["cell_acc"],
                    "epoch/train_puzzle_acc": train_metrics["puzzle_acc"],
                    "epoch/avg_sup_steps": train_metrics["avg_sup_steps"],
                }
            )

    print(f"\nTraining complete! Best puzzle accuracy: {best_puzzle_acc:.4f}")

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
