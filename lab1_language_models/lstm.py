import torch
import torch.nn.functional as F
from torch import nn


class LstmCell(nn.Module):
    def __init__(self, embedding_dim: int = 512, hidden_size: int = 512):
        super().__init__()

        self.hidden_size = hidden_size
        self.input_gate = nn.Linear(embedding_dim + hidden_size, hidden_size)
        self.forget_gate = nn.Linear(embedding_dim + hidden_size, hidden_size)
        self.output_gate = nn.Linear(embedding_dim + hidden_size, hidden_size)
        self.candidate_gate = nn.Linear(embedding_dim + hidden_size, hidden_size)

        self._init_weights()

    def _init_weights(self):
        for gate in [
            self.input_gate,
            self.forget_gate,
            self.output_gate,
            self.candidate_gate,
        ]:
            nn.init.xavier_uniform_(gate.weight)
            nn.init.zeros_(gate.bias)

    def forward(self, x, hidden):
        h_prev, c_prev = hidden
        concated = torch.cat([h_prev, x], dim=1)  # (B, H+C)

        f_t = torch.sigmoid(self.forget_gate(concated))  # (B, H)
        i_t = torch.sigmoid(self.input_gate(concated))  # (B, H)
        g_t = torch.tanh(self.candidate_gate(concated))  # (B, H)
        c_t = f_t * c_prev + i_t * g_t  # (B, H)
        o_t = torch.sigmoid(self.output_gate(concated))  # (B, H)
        h_t = o_t * torch.tanh(c_t)  # (B, H)

        return h_t, (h_t, c_t)


class Lstm(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int = 128,
        embedding_dim: int = 512,
        hidden_size: int = 512,
        layers_count: int = 3,
        dropout: float = 0.2,
        device: str = "cpu",
    ):
        super().__init__()
        self.seq_len = seq_len
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.cells = nn.ModuleList(
            [LstmCell(embedding_dim, hidden_size) for _ in range(layers_count)]
        )
        self.dropout_layer = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

        self.num_layers = layers_count
        self.device = device
        self.hidden_size = hidden_size

    def forward(self, idx, targets=None):
        B, T = idx.shape  # B - batch size, T - seq_len
        x = self.token_embedding(idx)

        h = [
            torch.zeros(B, self.hidden_size).to(x.device)
            for _ in range(self.num_layers)
        ]
        c = [
            torch.zeros(B, self.hidden_size).to(x.device)
            for _ in range(self.num_layers)
        ]

        outputs = []

        for t in range(T):
            x_t = x[:, t, :]
            for i, cell in enumerate(self.cells):
                h[i], (h[i], c[i]) = cell(x_t, (h[i], c[i]))
                if i < self.num_layers - 1:
                    x_t = self.dropout_layer(h[i])
                else:
                    x_t = h[i]

            outputs.append(x_t.unsqueeze(1))  # (B, 1, H)

        out = torch.cat(outputs, dim=1)  # (B, T, H)
        out = self.layer_norm(out)
        logits = self.lm_head(out)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
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

    def generate_text(self, tokenizer, text: str = "", max_tokens: int = 128):
        self.eval()
        input_tokens = tokenizer.encode(text)

        cls_token_tensor = torch.tensor(
            [input_tokens], dtype=torch.long, device=self.device
        )

        generated_tokens = self.generate(
            cls_token_tensor, max_new_tokens=max_tokens - len(input_tokens)
        )

        generated_list = generated_tokens[0].tolist()
        generated_text = tokenizer.decode(generated_list)

        return generated_text
