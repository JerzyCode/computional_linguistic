import logging

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from typing_extensions import Tuple

VAL_TEXT_PATH = "prepared_data/eval.txt"
TRAIN_TEXT_PATH = "prepared_data/train.txt"
TOKENIZER = "radlab/polish-gpt2-small-v2"

logger = logging.getLogger("dataset.py")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


def load_text_data(path: str) -> str:
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


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


def load_data(
    seq_len: int,
    dtype: torch.dtype,
    batch_size: int,
    tokenizer: AutoTokenizer,
    logger: logging.Logger = logger,
) -> Tuple[DataLoader, DataLoader]:
    logger.info(f"Loading text data from {TRAIN_TEXT_PATH} and {VAL_TEXT_PATH}")
    logger.info(
        f"Parameters:\n"
        f"  seq_len   = {seq_len}\n"
        f"  dtype     = {dtype}\n"
        f"  batch_size= {batch_size}"
    )
    training_text = load_text_data(TRAIN_TEXT_PATH)
    eval_text = load_text_data(VAL_TEXT_PATH)

    logger.info(f"Using tokenizer: {TOKENIZER}")

    logger.info("Tokenizing data")

    sep = tokenizer.sep_token if tokenizer.sep_token else tokenizer.eos_token
    train_tokens = torch.tensor(tokenizer.encode(training_text + sep))
    eval_tokens = torch.tensor(tokenizer.encode(eval_text + sep))

    logger.info(f"Created {len(train_tokens)} training tokens")
    logger.info(f"Created {len(eval_tokens)} eval tokens")

    train_dataset = TokenDataset(train_tokens, seq_len)
    eval_dataset = TokenDataset(eval_tokens, seq_len)

    logger.info("Creating dataloaders")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(42),
    )
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)

    return train_dataloader, eval_dataloader
