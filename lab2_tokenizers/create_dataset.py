import argparse
from enum import Enum
from pathlib import Path

import torch
from tokenizer import WhitespaceTokenizer
from torch.utils.data import Dataset
from transformers import AutoTokenizer

DATA_DIR = "./data"
CLEAN_DATA_FILE = "./data/cleaned.txt"


class TokenizerType(Enum):
    GPT = "gpt2"
    SENTENCEPIECE = "sentence_piece"
    WHITESPACE = "whitespace"


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


def load_text_data(file_path: str) -> str:
    path = Path(file_path)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return text


def load_tokenizer(load_path: str, custom_class=None):
    if custom_class is not None:
        tokenizer = custom_class.from_pretrained(load_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(load_path)
    print(f"Tokenizer loaded from: {load_path}")
    return tokenizer


def create_torch_dataset(
    text: str,
    tokenizer_type: TokenizerType,
    dataset_type: str = "train",
):
    tokenizer_path = f"./tokenizers/{tokenizer_type.value}"
    print(f"Tokenizer_path: {tokenizer_path}")

    tokenizer_class = (
        None if tokenizer_type != TokenizerType.WHITESPACE else WhitespaceTokenizer
    )
    tokenizer = load_tokenizer(tokenizer_path, tokenizer_class)

    output_dir = Path("dataset")
    output_dir.mkdir(parents=True, exist_ok=True)
    final_output_path = output_dir / f"{tokenizer_type.value}_{dataset_type}_data.pt"

    all_tokens = []
    lines = text.splitlines()
    total_lines = len(lines)

    for i, line in enumerate(lines, start=1):
        tokens = tokenizer.encode(line)
        print(type(tokens))
        all_tokens.extend(tokens)

        if i % 10000 == 0 or i == total_lines:
            print(
                f"Processed {i}/{total_lines} lines ({i / total_lines:.2%}), tokens={len(all_tokens)}"
            )

    if all_tokens:
        final_data = torch.tensor(all_tokens)
        torch.save(final_data, final_output_path)
        print(
            f"Final Torch dataset saved to {final_output_path} (length={len(final_data):,} tokens)"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type",
        type=str,
        choices=[t.value for t in TokenizerType],
        default=TokenizerType.GPT.value,
    )
    parser.add_argument("--text_path", type=str, default=CLEAN_DATA_FILE)
    parser.add_argument("--dataset_type", type=str, default="train")

    args = parser.parse_args()
    text_path = args.text_path
    tokenizer_type = TokenizerType(args.type)
    datset_type = args.dataset_type

    print(f"text_path: {text_path}")
    print(f"tokenizer_type: {tokenizer_type}")
    print(f"dataset_type: {datset_type}")

    text_data = load_text_data(text_path)
    text_data = text_data[:10000]

    create_torch_dataset(text_data, tokenizer_type, dataset_type=datset_type)


# srun --time=00:05:00 --mem=10G --ntasks 1 --nodes=1 --partition=plgrid --account=plgar2025-gpu-a100 --pty /bin/bash
# 64 369 661
# 2 908 489
