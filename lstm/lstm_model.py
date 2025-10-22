import torch
from torch import Tensor, nn
from transformers import AutoTokenizer


class Lstm(nn.Module):
    def __init__(
        self,
        device: str,
        tokenizer: AutoTokenizer,
        emb_dim: int = 256,
        hidden_size: int = 512,
        dtype: type = torch.float32,
        seq_len: int = 127,
    ):
        super().__init__()
        self.dtype = dtype
        self.device = device
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.E = nn.Embedding(tokenizer.vocab_size, emb_dim, dtype=dtype).to(device)

        self.W_f = nn.Parameter(
            torch.empty(hidden_size, emb_dim + hidden_size, dtype=dtype, device=device)
        )
        self.b_f = nn.Parameter(torch.zeros(hidden_size, 1, dtype=dtype, device=device))

        self.W_i = nn.Parameter(
            torch.empty(hidden_size, emb_dim + hidden_size, dtype=dtype, device=device)
        )
        self.b_i = nn.Parameter(torch.zeros(hidden_size, 1, dtype=dtype, device=device))

        self.W_c = nn.Parameter(
            torch.empty(hidden_size, emb_dim + hidden_size, dtype=dtype, device=device)
        )
        self.b_c = nn.Parameter(torch.zeros(hidden_size, 1, dtype=dtype, device=device))

        self.W_o = nn.Parameter(
            torch.empty(hidden_size, emb_dim + hidden_size, dtype=dtype, device=device)
        )
        self.b_o = nn.Parameter(torch.zeros(hidden_size, 1, dtype=dtype, device=device))

        self.W_vocab = nn.Parameter(
            torch.empty(tokenizer.vocab_size, hidden_size, dtype=dtype, device=device)
        )
        self.b_vocab = nn.Parameter(
            torch.zeros(tokenizer.vocab_size, dtype=dtype, device=device)
        )

        self.seq_len = seq_len

        self.gain = 1.0

        self._init_weights()

    def _init_weights(self):
        for name, W in zip(
            ["W_f", "W_i", "W_c", "W_o"], [self.W_f, self.W_i, self.W_c, self.W_o]
        ):
            input_dim = self.E.embedding_dim
            nn.init.xavier_uniform_(W[:, :input_dim], gain=self.gain)
            nn.init.xavier_uniform_(W[:, input_dim:], gain=self.gain)

            print(
                f"{name} norm (Frobenius): {W.norm().item():.6f}, min: {W.min().item():.6f}, max: {W.max().item():.6f}"
            )

        nn.init.xavier_uniform_(self.W_vocab, gain=self.gain)

        nn.init.ones_(self.b_f)
        nn.init.zeros_(self.b_i)
        nn.init.zeros_(self.b_c)
        nn.init.zeros_(self.b_o)
        nn.init.zeros_(self.b_vocab)

    def count_lstm_parameters(self) -> int:
        total = 0

        for name, param in self.__dict__.items():
            if isinstance(param, torch.Tensor):
                total += param.numel()

        for param in self.parameters():
            total += param.numel()

        return total

    # IN
    # input_ids -> (batch_size, )
    # H_t -> (batch_size, hidden_size)
    # C_t -> (batch_size, hidden_size)

    # OUT
    # logits -> (batch_size, vocab_size)
    # H_{t+1} -> (batch_size, hidden_size)
    # C_{t+1} -> (batch_size, hidden_size)
    def forward(
        self, input_ids: Tensor, H_t: Tensor, C_t: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:  # logits, H_{t+1}, C_{t+1}
        x_t = self.E(input_ids)  # (batch_size, emb_dim)

        # forget gate layer
        concated = torch.cat([H_t, x_t], dim=1)  # (batch_size, hidden_size + emb_size)

        f_t = torch.sigmoid(
            (concated @ self.W_f.T) + self.b_f.T
        )  # (batch_size, hidden_size)

        # input gate layer
        i_t = torch.sigmoid(
            (concated @ self.W_i.T) + self.b_i.T
        )  # (batch_size, hidden_size)

        C_t_next_cand = torch.tanh(
            (concated @ self.W_c.T) + self.b_c.T
        )  # (batch_size, hidden_size)

        C_t_next = f_t * C_t + i_t * C_t_next_cand

        o_t = torch.sigmoid(
            (concated @ self.W_o.T) + self.b_o.T
        )  # (batch_size, hidden_size)

        H_t_next = o_t * torch.tanh(C_t_next)  # (batch_size, hidden_size)

        logits = H_t_next @ self.W_vocab.T + self.b_vocab
        return logits, H_t_next, C_t_next
