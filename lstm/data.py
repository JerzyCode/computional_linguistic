import os
import random

import torch
from torch.utils.data import DataLoader, Dataset, Subset


class LanguageModelingDataset(Dataset):
    def __init__(self, inputs, labels):
        print(f"Inputs type: {type(inputs)}, len={len(inputs)}")
        print(f"Labels type: {type(labels)}, len={len(labels)}")
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]


def load_data(path: str, device: str, batch_size, shuffle: bool = False) -> DataLoader:
    if not os.path.exists(path):
        raise FileNotFoundError(f"There is no file with path: {path}")

    checkpoint = torch.load(path, map_location=device)
    dataset = LanguageModelingDataset(checkpoint["inputs"], checkpoint["labels"])

    num_samples = len(dataset)
    num_batches = (num_samples + batch_size - 1) // batch_size

    print(f"[load_data] Loaded dataset from {path}")
    print(f"[load_data] Number of samples: {num_samples}")
    print(f"[load_data] Batch size: {batch_size}")
    print(f"[load_data] Total batches: {num_batches}")

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def load_data_subset(
    path: str, device: str, batch_size: int, subset_size: int = 1000
) -> DataLoader:
    if not os.path.exists(path):
        raise FileNotFoundError(f"There is no file with path: {path}")

    checkpoint = torch.load(path, map_location=device)
    dataset = LanguageModelingDataset(checkpoint["inputs"], checkpoint["labels"])

    subset_size = min(subset_size, len(dataset))
    subset_indices = random.sample(range(len(dataset)), subset_size)
    subset = Subset(dataset, subset_indices)

    num_samples = len(subset)
    num_batches = (num_samples + batch_size - 1) // batch_size

    print(f"[load_data_subset] Loaded subset from {path}")
    print(f"[load_data_subset] Subset size: {num_samples}")
    print(f"[load_data_subset] Batch size: {batch_size}")
    print(f"[load_data_subset] Total batches: {num_batches}")

    return DataLoader(subset, batch_size=batch_size, shuffle=False)
