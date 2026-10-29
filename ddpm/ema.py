import copy

import torch


class EMA:
    """Exponential moving average of the weights; sampling from the EMA is markedly cleaner
    than from the raw, noisy-SGD weights."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay, self.steps = decay, 0
        self.model = copy.deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.steps += 1
        d = min(self.decay, (1 + self.steps) / (10 + self.steps))  # warm-up: track fast early on
        for e, p in zip(self.model.parameters(), model.parameters()):
            e.mul_(d).add_(p.detach(), alpha=1 - d)
        for e, b in zip(self.model.buffers(), model.buffers()):
            e.copy_(b)
