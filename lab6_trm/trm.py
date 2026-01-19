from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def rms_norm(x, eps=1e-6):
    return x / torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TRMBlock(nn.Module):
    def __init__(self, hidden_size, seq_len):
        super().__init__()
        self.seq_mixer = nn.Linear(seq_len, seq_len, bias=False)
        self.ffn = SwiGLU(hidden_size)

    def forward(self, x):
        x = rms_norm(x + self.seq_mixer(x.transpose(1, 2)).transpose(1, 2))
        x = rms_norm(x + self.ffn(x))
        return x


class ReasoningModule(nn.Module):
    def __init__(self, hidden_size, seq_len, n_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [TRMBlock(hidden_size, seq_len) for _ in range(n_layers)]
        )

    def forward(self, hidden_states, input_injection):
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


@dataclass
class TRMCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


class TRM(nn.Module):
    def __init__(
        self,
        vocab_size=10,
        seq_len=81,
        hidden_size=256,
        n_layers=2,
        H_cycles=2,
        L_cycles=3,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.embed_scale = hidden_size**0.5

        self.L_level = ReasoningModule(hidden_size, seq_len, n_layers)

        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.q_head = nn.Linear(hidden_size, 1, bias=True)

        self.register_buffer("H_init", torch.randn(hidden_size) * 0.02)
        self.register_buffer("L_init", torch.randn(hidden_size) * 0.02)

        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

    def get_initial_carry(self, batch_size):
        z_H = self.H_init.view(1, 1, -1).expand(batch_size, self.seq_len, -1).clone()
        z_L = self.L_init.view(1, 1, -1).expand(batch_size, self.seq_len, -1).clone()
        return TRMCarry(z_H=z_H, z_L=z_L)

    def forward(self, x_input, carry=None):
        batch_size = x_input.size(0)

        x = self.embed_tokens(x_input) * self.embed_scale

        if carry is None:
            carry = self.get_initial_carry(batch_size)

        z_H, z_L = carry.z_H, carry.z_L

        with torch.no_grad():
            for _ in range(self.H_cycles - 1):
                for _ in range(self.L_cycles):
                    z_L = self.L_level(z_L, z_H + x)
                z_H = self.L_level(z_H, z_L)

        for _ in range(self.L_cycles):
            z_L = self.L_level(z_L, z_H + x)
        z_H = self.L_level(z_H, z_L)

        new_carry = TRMCarry(z_H=z_H.detach(), z_L=z_L.detach())
        logits = self.lm_head(z_H)
        q_halt = self.q_head(z_H[:, 0]).squeeze(-1)

        return new_carry, logits, q_halt

    @torch.no_grad()
    def solve(self, x_input, max_steps=16):
        self.eval()
        batch_size = x_input.size(0)

        carry = self.get_initial_carry(batch_size)

        for _ in range(max_steps):
            carry, logits, q_halt = self.forward(x_input, carry)
            if (q_halt > 0).all():
                break

        return logits.argmax(dim=-1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = TRM()
    print(f"Parameters: {count_parameters(model):,}")

    x = torch.randint(0, 10, (4, 81))
    carry, logits, q_halt = model(x)

    print(f"z_H: {carry.z_H.shape}, z_L: {carry.z_L.shape}")
    print(f"logits: {logits.shape}, q_halt: {q_halt.shape}")
    print(f"predictions: {model.solve(x).shape}")
