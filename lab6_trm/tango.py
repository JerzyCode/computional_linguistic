import torch
import torch.nn.functional as F


class Tango(torch.optim.Optimizer):
    """
    Tango (TANh + Gelu + sOftplus)

    A novel optimizer using nonlinear gradient transformations for stable training.
    """

    def __init__(self, params, lr=1.0):
        defaults = dict(lr=lr)
        super(Tango, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state["v"] = torch.zeros_like(p.data)

                v = state["v"]
                lr = group["lr"]

                v_tanh = torch.tanh(v)
                g_tanh = torch.tanh(g)

                max_tanh = torch.maximum(v_tanh, g_tanh)
                soft_max = F.softplus(max_tanh)

                g_norm = g / (torch.norm(g) + soft_max)
                v_norm = v / (torch.norm(v) + soft_max)

                g_updated = F.gelu(g_norm)
                v_updated = F.gelu(v_norm)

                p.data.sub_(g_updated * lr)
                state["v"] = v - v_updated

        return loss
