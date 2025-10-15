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
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.E = nn.Embedding(tokenizer.vocab_size, emb_dim, dtype=dtype).to(device)

        self.W_f = torch.empty(
            hidden_size, emb_dim + hidden_size, device=device, dtype=dtype
        )
        self.b_f = torch.zeros(hidden_size, 1, device=device, dtype=dtype)

        self.W_i = torch.empty(
            hidden_size, emb_dim + hidden_size, device=device, dtype=dtype
        )
        self.b_i = torch.zeros(hidden_size, 1, device=device, dtype=dtype)

        self.W_c = torch.empty(
            hidden_size, emb_dim + hidden_size, device=device, dtype=dtype
        )
        self.b_c = torch.zeros(hidden_size, 1, device=device, dtype=dtype)

        self.W_o = torch.empty(
            hidden_size, emb_dim + hidden_size, device=device, dtype=dtype
        )
        self.b_o = torch.zeros(hidden_size, 1, device=device, dtype=dtype)

        self.W_vocab = torch.empty(
            tokenizer.vocab_size, hidden_size, device=device, dtype=dtype
        )
        self.b_vocab = torch.zeros(tokenizer.vocab_size, device=device, dtype=dtype)

        self.seq_len = seq_len

        self._init_weights()

    def _init_weights(self):
        for w in [self.W_f, self.W_i, self.W_c, self.W_o, self.W_vocab]:
            nn.init.xavier_uniform_(w)

        nn.init.constant_(self.b_f, 0.5)
        nn.init.zeros_(self.b_i)
        nn.init.zeros_(self.b_c)
        nn.init.zeros_(self.b_o)
        nn.init.zeros_(self.b_vocab)

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
