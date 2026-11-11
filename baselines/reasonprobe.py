"""ReasonProbe (Zhang et al., 2025, "Reasoning Models Know When They're Right: Probing Hidden States
for Self-Verification") — supervised hidden-state correctness probe.

Faithful "Final" setting (paper Sec. 4.4): last-layer hidden state at the last token of the final
answer; probe = σ(W2·ReLU(W1 e + b1) + b2) with hidden width d ∈ {0, 16, 32} (d = 0 is linear);
weighted BCE with pos_weight = α · (#neg / #pos); Adam, batch 64, ≤ 200 epochs, early stopping on
validation loss. Grid (paper Table 7): lr × α × weight decay × d = 3 × 8 × 3 × 3. Selection rule:
among the 10 configurations with the highest validation accuracy, take the one with the smallest d.
Trained on the training pool, early-stopped and selected on the validation pool.

``ReasonProbe-L<l>`` rows re-train the selected configuration on layer ``l`` (layer ablation).
"""
from __future__ import annotations

import itertools

import numpy as np

from baselines.base import Baseline, labels_of

POSITION = "answer"
GRID = {"lr": (1e-3, 1e-4, 1e-5), "alpha": (0.3, 0.5, 0.7, 0.9, 1.0, 1.5, 2.0, 3.0),
        "weight_decay": (0.001, 0.01, 0.1), "d": (0, 16, 32)}
BATCH, MAX_EPOCHS, PATIENCE, SEED = 64, 200, 10, 0


def features(questions, layer_index: int) -> np.ndarray:
    pos = ("answer", "think_end", "last").index(POSITION)
    return np.stack([r["hidden"][pos, layer_index] for q in questions for r in q.replay]).astype(np.float32)


class _ProbeBank:
    """G independent probes of hidden width d trained in lock-step (one per grid configuration).

    Mirrors ``nn.Linear`` initialisation, ``BCEWithLogitsLoss(pos_weight)`` and ``torch.optim.Adam``
    (coupled L2 weight decay) per configuration; batching only removes Python/kernel overhead.
    """

    def __init__(self, dim: int, d: int, groups: int, device):
        import torch

        gen = torch.Generator(device="cpu").manual_seed(SEED)

        def uniform(shape, fan_in):
            bound = 1.0 / fan_in ** 0.5
            return ((torch.rand(shape, generator=gen) * 2 - 1) * bound).to(device).requires_grad_()

        self.d = d
        if d == 0:
            self.params = [uniform((groups, dim), dim), uniform((groups,), dim)]
        else:
            self.params = [uniform((groups, dim, d), dim), uniform((groups, d), dim),
                           uniform((groups, d), d), uniform((groups,), d)]

    def logits(self, x):
        import torch

        if self.d == 0:
            w, b = self.params
            return x @ w.T + b                                              # [B, G]
        w1, b1, w2, b2 = self.params
        h = torch.relu(torch.einsum("bi,gid->bgd", x, w1) + b1)
        return (h * w2).sum(-1) + b2


def _train_grid(x, y, xv, yv, configs: list[dict], device) -> list[tuple[float, dict, object]]:
    """Train every configuration (all with the same width d) with per-config early stopping."""
    import torch

    g = len(configs)
    bank = _ProbeBank(x.shape[1], configs[0]["d"], g, device)
    lr = torch.tensor([c["lr"] for c in configs], device=device)
    wd = torch.tensor([c["weight_decay"] for c in configs], device=device)
    ratio = (1 - y.mean()) / y.mean().clamp(min=1e-8)
    pos_weight = torch.tensor([c["alpha"] for c in configs], device=device) * ratio

    def loss_per_config(logits, target):                                # BCEWithLogits(pos_weight), mean
        log_sig, log_1m = torch.nn.functional.logsigmoid(logits), torch.nn.functional.logsigmoid(-logits)
        t = target[:, None]
        return -(pos_weight * t * log_sig + (1 - t) * log_1m).mean(0)  # [G]

    def per_group(v, p):                                                 # broadcast [G] over p's shape
        return v.view(-1, *([1] * (p.dim() - 1)))

    m = [torch.zeros_like(p) for p in bank.params]
    v = [torch.zeros_like(p) for p in bank.params]
    best = torch.full((g,), float("inf"), device=device)
    best_state = [p.detach().clone() for p in bank.params]
    wait = torch.zeros(g, dtype=torch.long, device=device)
    active = torch.ones(g, dtype=torch.bool, device=device)
    gen = torch.Generator(device="cpu").manual_seed(SEED)
    step, b1, b2, eps = 0, 0.9, 0.999, 1e-8
    for _ in range(MAX_EPOCHS):
        for idx in torch.randperm(len(x), generator=gen).split(BATCH):
            loss = loss_per_config(bank.logits(x[idx]), y[idx]).sum()
            grads = torch.autograd.grad(loss, bank.params)
            step += 1
            with torch.no_grad():
                for p, gr, mi, vi in zip(bank.params, grads, m, v, strict=True):
                    gr = gr + per_group(wd, p) * p
                    mi.mul_(b1).add_((1 - b1) * gr)
                    vi.mul_(b2).add_((1 - b2) * gr * gr)
                    update = per_group(lr, p) * (mi / (1 - b1 ** step)) / ((vi / (1 - b2 ** step)).sqrt() + eps)
                    p.sub_(update * per_group(active.float(), p))
        with torch.no_grad():
            val = loss_per_config(bank.logits(xv), yv)
            improved = active & (val < best - 1e-6)
            best = torch.where(improved, val, best)
            wait = torch.where(improved, torch.zeros_like(wait), wait + active.long())
            for s, p in zip(best_state, bank.params, strict=True):
                s[improved] = p.detach()[improved]
            active = active & (wait < PATIENCE)
        if not active.any():
            break
    with torch.no_grad():
        for p, s in zip(bank.params, best_state, strict=True):
            p.copy_(s)
        acc = ((bank.logits(xv) > 0).float() == yv[:, None]).float().mean(0)
    return [(float(acc[i]), configs[i], (bank, i)) for i in range(g)]


class ReasonProbe(Baseline):
    paper = "ReasonProbe"
    probabilistic = True

    def __init__(self, layer: int | None = None, config: dict | None = None):
        self.layer = layer            # None = last layer
        self.config = config          # fixed hyper-parameters (layer ablation)
        self.main = layer is None
        self.name = "ReasonProbe" if layer is None else f"ReasonProbe-L{layer}"
        self.model = None
        self.layer_index = -1

    def fit(self, train, val):
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        layers = list(train[0].replay[0]["layers"])
        self.layer_index = layers.index(self.layer) if self.layer is not None else len(layers) - 1
        self.layer = layers[self.layer_index]
        x = torch.tensor(features(train, self.layer_index), device=device)
        xv = torch.tensor(features(val, self.layer_index), device=device)
        y = torch.tensor(labels_of(train), dtype=torch.float32, device=device)
        yv = torch.tensor(labels_of(val), dtype=torch.float32, device=device)
        grid = [dict(zip(GRID, values, strict=True)) for values in itertools.product(*GRID.values())]
        if self.config is not None:
            grid = [self.config]
        runs = []
        for d in sorted({c["d"] for c in grid}):
            runs += _train_grid(x, y, xv, yv, [c for c in grid if c["d"] == d], device)
        top10 = sorted(runs, key=lambda r: -r[0])[:10]
        _, self.config, self.model = min(top10, key=lambda r: (r[1]["d"], -r[0]))
        self.device = device

    def score(self, q):
        import torch

        x = torch.tensor(features([q], self.layer_index), device=self.device)
        bank, i = self.model
        with torch.no_grad():
            return torch.sigmoid(bank.logits(x)[:, i]).cpu().numpy()

    def describe(self):
        return {"layer": int(self.layer), "position": POSITION, **(self.config or {})}
