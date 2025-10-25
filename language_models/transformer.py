
import torch
import torch.nn.functional as F
from torch import nn


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
        device: str = "cpu"
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
        self.device= device

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
            torch.arange(T, device=self.device)  # (T, C)
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
