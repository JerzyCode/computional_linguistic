import logging
import os
import time

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
val_path = os.path.join(PREPARED_DATASET_DIR, "val_data.pt")

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


class FeedForward(nn.Module):
    def __init__(self, embedding_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),  # * 4 from paper
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class AttentionHead(nn.Module):
    def __init__(
        self, head_size: int, embedding_dim: int, seq_len: int, dropout: float
    ):
        super().__init__()
        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(seq_len, seq_len)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        # B = batch_size, T = seq_len, C = head_size

        k: torch.Tensor = self.key(x)  # (B, T, C)
        q: torch.Tensor = self.query(x)  # (B, T, C)

        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v  # (B, T, T) x (B, T, C) -> (B, T, C)

        return out  # (B, T, C)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        embedding_dim: int,
        seq_len: int,
        dropout: float,
    ):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                AttentionHead(head_size, embedding_dim, seq_len, dropout)
                for _ in range(num_heads)
            ]
        )
        self.projection = nn.Linear(embedding_dim, embedding_dim)  # Wo - in paper
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, C)
        out = self.projection(out)
        out = self.dropout(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(
        self, num_heads: int, embedding_dim: int, seq_len: int, dropout: float
    ):
        super().__init__()
        head_size = embedding_dim // num_heads
        self.sa_heads = MultiHeadAttention(
            num_heads, head_size, embedding_dim, seq_len, dropout
        )
        self.feed_forward = FeedForward(embedding_dim, dropout)
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

    # note that layer norm are applied before transformation
    def forward(self, x):
        x = x + self.sa_heads(self.layer_norm1(x))  # (B, T, C)
        x = x + self.feed_forward(self.layer_norm2(x))
        return x


class TransformerLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int = 128,
        embedding_dim: int = 512,
        num_heads: int = 8,
        blocks_count: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(seq_len, embedding_dim)
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(num_heads, embedding_dim, seq_len, dropout)
                for _ in range(blocks_count)
            ],
        )
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size)  # language_model_head

        self.seq_len = seq_len

    def forward(self, idx, targets=None):
        """
            B = batch_size
            T = seq_len
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
            torch.arange(T, device=device)  # (T, C)
        )  # ints from  0 ... T-1

        x = tok_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)  # (B, T, C)
        x = self.layer_norm(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

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
            idx_cond = idx[:, -self.seq_len :]  # to not pass idx more than block size

            logits, _ = self(idx_cond)
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


def generate_text(
    model: TransformerLanguageModel, text: str = "", max_new_tokens: int = 15
):
    input_tokens = tokenizer.encode(text)

    cls_token_tensor = torch.tensor([input_tokens], dtype=torch.long, device=device)

    generated_tokens = model.generate(cls_token_tensor, max_new_tokens=max_new_tokens)

    generated_list = generated_tokens[0].tolist()
    generated_text = tokenizer.decode(generated_list)

    return generated_text


def save_model(path: str, model: nn.Module):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    logger.info(f"Model saved to {path}")


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
    training_minutes: int = 5,
    log_freq: int = 1,
    save_freq: int = 10,
    model_save_dir: str = "models/",
):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.75,
        patience=5,
        min_lr=1e-6,
    )

    logger.info(f"Starting training with lr={optimizer.param_groups[0]['lr']}")
    model.train()
    train_losses = []
    eval_losses = []
    epoch = 0

    end_time = time.time() + 60 * training_minutes

    logger.info(f"Training will and at: {time.ctime(end_time)}")
    while time.time() < end_time:
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
            print("Calculated batch")

        epoch_loss = loss_sum / num_batches
        train_losses.append(epoch_loss)

        eval_loss = eval(model, eval_data)
        eval_losses.append(eval_loss)

        epoch += 1

        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(eval_loss)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr < old_lr:
            logger.info(f"LR reduced from {old_lr:.6f} to {new_lr:.6f}")

        if epoch % log_freq == 0:
            logger.info(
                f"Epoch: {epoch}, train_loss: {epoch_loss}, eval_loss: {eval_loss}"
            )

        if epoch % save_freq == 0:
            path = os.path.join(model_save_dir, f"model_{epoch}.pt")
            save_model(path, model)

    path = os.path.join(model_save_dir, f"transformer_{epoch}.pt")
    save_model(path, model)

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
    max_new_tokens = 100
    seq_len = 128
    lr = 1e-3
    batch_size = 64
    heads_count = 4
    embedding_dim = 64
    blocks_count = 4
    training_minutes = 3

    logger.info(f"Start training with device: {device}")
    logger.info("Net and training parameters:")
    logger.info(f"seq_len: {seq_len}")
    logger.info(f"lr: {lr}")
    logger.info(f"batch_size: {batch_size}")
    logger.info(f"heads_count: {heads_count}")
    logger.info(f"embedding_dim: {embedding_dim}")
    logger.info(f"blocks_count: {blocks_count}")
    logger.info(f"training_minutes: {training_minutes}")

    logger.info("Start loading training data")
    training_data_loader = create_data_loader(
        train_path, batch_size=batch_size, seq_len=seq_len
    )
    logger.info("Finished training data")

    logger.info("Start loading eval data")
    eval_data_loader = create_data_loader(
        val_path, batch_size=batch_size, seq_len=seq_len
    )
    logger.info("Finished eval data")

    model = TransformerLanguageModel(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=embedding_dim,
        num_heads=heads_count,
        blocks_count=blocks_count,
        seq_len=seq_len,
    )

    generated_before_training = generate_text(model, max_new_tokens=max_new_tokens)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    train_losses, eval_losses = train(
        model,
        optimizer,
        training_data_loader,
        eval_data_loader,
        training_minutes=training_minutes,
        log_freq=1,
    )

    plot_losses(train_losses, eval_losses)

    generated_after_training = generate_text(model, max_new_tokens=max_new_tokens)

    logger.info("Generated text before training:")
    logger.info(generated_before_training)

    logger.info("\n\nGenerated text after training:")
    logger.info(generated_after_training)
