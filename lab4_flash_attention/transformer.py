from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func
from torch import nn
from typing_extensions import Optional, Sequence


class ModelType(Enum):
    BASE = "base"
    FLASH_ATTENTION_V2 = "flash_attention_v2"
    BFLOAT16 = "bfloat16"
    WINDOWED = "windowed"
    GROUPED_KV_ATTENTION = "grouped_kv_flash_attention"


@dataclass(frozen=True)
class ModelArgs:
    vocab_size: int
    device: str = "cuda"
    model_tyoe: ModelType = ModelType.BASE
    seq_len: int = 128
    embedding_dim: int = 512
    blocks_count: int = 4
    num_heads: int = 8
    dropout: float = 0.15
    window_size: Optional[int] = None


class LanguageModel(nn.Module):
    def __init__(
        self,
        model_args: ModelArgs,
        transformer_blocks: Sequence[nn.Module],
    ):
        super().__init__()
        self.tymodel_type = model_args.model_tyoe
        self.seq_len = model_args.seq_len
        self.device = model_args.device

        self.blocks = nn.ModuleList(transformer_blocks)
        self.token_embedding = nn.Embedding(
            model_args.vocab_size, model_args.embedding_dim
        )
        self.position_embedding = nn.Embedding(
            model_args.seq_len, model_args.embedding_dim
        )
        self.layer_norm = nn.LayerNorm(model_args.embedding_dim)
        self.lm_head = nn.Linear(model_args.embedding_dim, model_args.vocab_size)

    def forward(self, x, targets=None):
        B, T = x.shape

        tok_emb = self.token_embedding(x)  # (B, T, C)
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
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss


class _FeedForward(nn.Module):
    def __init__(self, embedding_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class _AttentionHead(nn.Module):
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


class _MultiHeadAttention(nn.Module):
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
                _AttentionHead(head_size, embedding_dim, seq_len, dropout)
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


class _TransformerBlock(nn.Module):
    def __init__(
        self,
        num_heads: int,
        embedding_dim: int,
        seq_len: int,
        dropout: float,
    ):
        super().__init__()
        head_size = embedding_dim // num_heads
        self.sa_heads = _MultiHeadAttention(
            num_heads,
            head_size,
            embedding_dim,
            seq_len,
            dropout,
        )
        self.feed_forward = _FeedForward(embedding_dim, dropout)
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        x = x + self.sa_heads(self.layer_norm1(x))  # (B, T, C)
        x = x + self.feed_forward(self.layer_norm2(x))
        return x


class _MultiHeadFlashAttention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.key = nn.Linear(embedding_dim, self.head_dim, bias=False)
        self.query = nn.Linear(embedding_dim, self.head_dim, bias=False)
        self.value = nn.Linear(embedding_dim, self.head_dim, bias=False)
        self.projection = nn.Linear(embedding_dim, embedding_dim)

        self.dropout = dropout

    def forward(self, x):
        B, T, C = x.shape

        Q = self.query(x).view(B, T, self.num_heads, self.head_dim)
        K = self.key(x).view(B, T, self.num_heads, self.head_dim)
        V = self.value(x).view(B, T, self.num_heads, self.head_dim)

        attention_out = flash_attn_func(
            Q,
            K,
            V,
            causal=True,
            softmax_scale=None,
            dropout_p=self.dropout if self.training else 0.0,
        )  # [B, H, T, D]
        attention_out = attention_out.view(B, T, C)  # (B, T, C)
        attention_out = self.projection(attention_out)
        return attention_out


class _TransformerBlockFlashAttention(nn.Module):
    def __init__(self, num_heads: int, embedding_dim: int, dropout: float):
        super().__init__()
        self.sa_heads = _MultiHeadFlashAttention(embedding_dim, num_heads, dropout)
        self.feed_forward = _FeedForward(embedding_dim, dropout)
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        x = x + self.sa_heads(self.layer_norm1(x))  # (B, T, C)
        x = x + self.feed_forward(self.layer_norm2(x))
        return x


class _MultiHeadWindowedFlashAttention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dropout: float,
        window_left: int,
        window_right: int,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.key = nn.Linear(embedding_dim, self.head_dim, bias=False)
        self.query = nn.Linear(embedding_dim, self.head_dim, bias=False)
        self.value = nn.Linear(embedding_dim, self.head_dim, bias=False)
        self.projection = nn.Linear(embedding_dim, embedding_dim)

        self.dropout = dropout
        self.window = (window_left, window_right)

    def forward(self, x):
        B, T, C = x.shape

        Q = self.query(x).view(B, T, self.num_heads, self.head_dim)
        K = self.key(x).view(B, T, self.num_heads, self.head_dim)
        V = self.value(x).view(B, T, self.num_heads, self.head_dim)

        # flash-attn sliding/windowed
        attention_out = flash_attn_func(
            Q,
            K,
            V,
            causal=True,
            softmax_scale=None,
            dropout_p=self.dropout if self.training else 0.0,
            window_size=self.window,
        )  # -> [B, H, T, D]

        attention_out = attention_out.view(B, T, C)  # (B, T, C)
        attention_out = self.projection(attention_out)
        return attention_out


class _TransformerBlockWindowedFlashAttention(nn.Module):
    def __init__(
        self, num_heads: int, embedding_dim: int, dropout: float, window_size: int
    ):
        super().__init__()
        self.sa_heads = _MultiHeadWindowedFlashAttention(
            embedding_dim, num_heads, dropout, window_size, window_size
        )
        self.feed_forward = _FeedForward(embedding_dim, dropout)
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        x = x + self.sa_heads(self.layer_norm1(x))  # (B, T, C)
        x = x + self.feed_forward(self.layer_norm2(x))
        return x


class ModelFactory:
    @staticmethod
    def _create_baseline_model(model_args: ModelArgs) -> LanguageModel:
        transformer_blocks = [
            _TransformerBlock(
                model_args.num_heads,
                model_args.embedding_dim,
                model_args.seq_len,
                model_args.dropout,
            )
            for _ in range(model_args.blocks_count)
        ]
        model = LanguageModel(model_args, transformer_blocks)
        model = model.to(model_args.device)
        model = model.to(torch.float32)
        return model

    @staticmethod
    def _create_fp16_model(model_args: ModelArgs) -> LanguageModel:
        transformer_blocks = [
            _TransformerBlock(
                model_args.num_heads,
                model_args.embedding_dim,
                model_args.seq_len,
                model_args.dropout,
            )
            for _ in range(model_args.blocks_count)
        ]
        model = LanguageModel(model_args, transformer_blocks)
        model = model.to(model_args.device)
        model = model.to(torch.bfloat16)
        return model

    @staticmethod
    def _create_flash_attention_model(model_args: ModelArgs) -> LanguageModel:
        transformer_blocks = [
            _TransformerBlockFlashAttention(
                model_args.num_heads,
                model_args.embedding_dim,
                model_args.dropout,
            )
            for _ in range(model_args.blocks_count)
        ]
        model = LanguageModel(model_args, transformer_blocks)
        model = model.to(model_args.device)
        model = model.to(torch.bfloat16)
        return model

    @staticmethod
    def _create_windowed_model(model_args: ModelArgs) -> LanguageModel:
        transformer_blocks = [
            _TransformerBlockWindowedFlashAttention(
                model_args.num_heads,
                model_args.embedding_dim,
                model_args.dropout,
                model_args.window_size if model_args.window_size is not None else 8,
            )
            for _ in range(model_args.blocks_count)
        ]
        model = LanguageModel(model_args, transformer_blocks)
        model = model.to(model_args.device)
        model = model.to(torch.bfloat16)
        return model

    @staticmethod
    def create_model(model_args: ModelArgs) -> LanguageModel:
        if model_args.model_tyoe == ModelType.BASE:
            return ModelFactory._create_baseline_model(model_args)
        elif model_args.model_tyoe == ModelType.BFLOAT16:
            return ModelFactory._create_fp16_model(model_args)
        elif model_args.model_tyoe == ModelType.FLASH_ATTENTION_V2:
            return ModelFactory._create_flash_attention_model(model_args)
        elif model_args.model_tyoe == ModelType.WINDOWED:
            return ModelFactory._create_windowed_model(model_args)
        else:
            raise ValueError(f"Unknown model type: {model_args.model_tyoe}")
