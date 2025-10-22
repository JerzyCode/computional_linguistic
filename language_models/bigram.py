import logging
import os

import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("dkleczek/bert-base-polish-uncased-v1")

PREPARED_DATASET_DIR = "./prepared_data"
small_train_path = os.path.join(PREPARED_DATASET_DIR, "small_train_data.pt")
small_val_path = os.path.join(PREPARED_DATASET_DIR, "small_val_data.pt")
train_path = os.path.join(PREPARED_DATASET_DIR, "train_data.pt")
train_val_path = os.path.join(PREPARED_DATASET_DIR, "val_data.pt")

device = "cpu"
dtype = torch.float32

torch.manual_seed(1234)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("Training")


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


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, seq_len: int = 128, embedding_dim: int = 512):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(seq_len, embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, idx, targets=None):
        """
            B = batch_size
            T = seq_len (długość sekwencji)
            C = embedding_dim

            idx (torch.LongTensor): (B, T)
            targets (torch.LongTensor, optional): (B, T)

        Returns:
            logits (torch.FloatTensor): (B, T, C),
            loss (torch.FloatTensor or None)
        """

        B, T = idx.shape

        tok_emb = self.token_embedding(idx)  # (B, T, C)

        # pos-encoding
        pos_emb = self.position_embedding(
            torch.arrange(T, device=device)  # (T, C)
        )  # ints from  0 ... T-1

        x = tok_emb + pos_emb  # (B, T, C)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape  # due to loss of torch dimensions required
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens: int):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


def load_dataset(path: str, seq_len: int) -> TokenDataset:
    loaded_data = torch.load(path)
    return TokenDataset(loaded_data, seq_len)


def create_data_loader(
    path, batch_size: int = 32, seq_len=16, max_elements: int = None
) -> DataLoader:
    dataset = load_dataset(path, seq_len=seq_len)

    if max_elements is not None:
        dataset = Subset(dataset, list(range(max_elements)))

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    num_batches = len(data_loader)
    print(f"DataLoader created: {num_batches} batches, batch_size={batch_size}")

    return data_loader


def generate_text(model: BigramLanguageModel, text: str = "", max_new_tokens: int = 15):
    input_tokens = tokenizer.encode(text)

    cls_token_tensor = torch.tensor([input_tokens], dtype=torch.long, device=device)

    generated_tokens = model.generate(cls_token_tensor, max_new_tokens=max_new_tokens)

    generated_list = generated_tokens[0].tolist()
    generated_text = tokenizer.decode(generated_list)

    return generated_text


@torch.no_grad()
def eval(model: nn.Module, eval_data: DataLoader) -> float:
    model.eval()
    loss_sum = 0.0
    num_batches = 0

    for batch in eval_data:
        x, y = batch
        x, y = x.to(device=device), y.to(device=device)

        _, loss = model(x, y)
        loss_sum += loss.item()
        num_batches += 1

    return loss_sum / num_batches


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    training_data: DataLoader,
    eval_data: DataLoader,
    epochs: int = 25,
    log_freq: int = 1,
):
    logger.info(f"Starting training with lr={optimizer.param_groups[0]['lr']}")
    model.train()
    train_losses = []
    eval_losses = []

    for epoch in range(epochs):
        loss_sum = 0.0
        num_batches = 0

        model.train()
        for batch in training_data:
            x, y = batch
            x, y = x.to(device=device), y.to(device=device)

            optimizer.zero_grad()

            _, loss = model(x, y)
            loss_sum += loss.item()
            num_batches += 1

            loss.backward()
            optimizer.step()

        epoch_loss = loss_sum / num_batches
        train_losses.append(epoch_loss)

        eval_loss = eval(model, eval_data)
        eval_losses.append(eval_loss)

        if epoch % log_freq == 0 or epoch == epochs - 1:
            logger.info(
                f"Epoch: {epoch}/{epochs}, train_loss: {epoch_loss}, eval_loss: {eval_loss}"
            )

    return train_losses, eval_losses


def plot_losses(train_losses, eval_losses, save_path="plots/training.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Training Loss", c="blue")
    plt.plot(eval_losses, label="Eval Loss", c="green")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss over Iterations")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    logger.info(f"Loss plots saved to {save_path}")


if __name__ == "__main__":
    print(f"Start training with device: {device}")
    seq_len = 16
    lr = 5e-3
    batch_size = 8

    logger.info("Start loading training data")
    training_data_loader = create_data_loader(
        small_train_path, batch_size=batch_size, seq_len=seq_len
    )
    logger.info("Finished training data")

    logger.info("Start loading eval data")
    eval_data_loader = create_data_loader(
        small_val_path, batch_size=batch_size, seq_len=seq_len
    )
    logger.info("Finished eval data")

    model = BigramLanguageModel(vocab_size=tokenizer.vocab_size, embedding_dim=32)

    generated_before_training = generate_text(model, max_new_tokens=50)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    train_losses, eval_losses = train(
        model, optimizer, training_data_loader, eval_data_loader, epochs=50, log_freq=1
    )
    plot_losses(train_losses, eval_losses)

    generated_after_training = generate_text(model, max_new_tokens=50)

    logger.info("Generated text before training:")
    logger.info(generated_before_training)

    logger.info("\n\nGenerated text after training:")
    logger.info(generated_after_training)
