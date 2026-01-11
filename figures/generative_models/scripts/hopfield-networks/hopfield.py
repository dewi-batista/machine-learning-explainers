from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

UpdateRule = Literal["sync", "async"]


def binarize(x: torch.Tensor, *, threshold: float = 0.5, low: float = -1.0, high: float = 1.0) -> torch.Tensor:
    """
    Binarize a tensor with values in [0, 1] into {low, high}.

    For MNIST tensors, `x` is typically `float32` in [0, 1] with shape (..., 1, 28, 28).
    """
    high_t = torch.as_tensor(high, device=x.device, dtype=x.dtype)
    low_t = torch.as_tensor(low, device=x.device, dtype=x.dtype)
    return torch.where(x >= threshold, high_t, low_t)


def to_pm_one(x: torch.Tensor, *, threshold: float = 0.0) -> torch.Tensor:
    """
    Convert a tensor to {-1, +1}.

    - If x is already in {-1, +1}, it is returned unchanged.
    - If x is in {0, 1} or generally non-negative, it is thresholded to {-1, +1}.
    """
    if x.numel() == 0:
        return x

    x = x.to(torch.float32) if not x.is_floating_point() else x

    x_min = float(x.min().item())
    x_max = float(x.max().item())
    if x_min >= -1.0 and x_max <= 1.0:
        # Common cases: {-1, +1}, {0, 1}, or [0, 1].
        if x_min >= 0.0:
            return torch.where(x > threshold, torch.ones_like(x), -torch.ones_like(x))
        # Already has negatives; assume it's already signed and just clamp to {-1, +1}.
        return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))

    return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))


def add_flip_noise(state_pm_one: torch.Tensor, p: float, *, generator: torch.Generator | None = None) -> torch.Tensor:
    """Flip each bit in `state_pm_one` independently with probability `p`."""
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"p must be in [0, 1], got {p}")
    if p == 0.0:
        return state_pm_one.clone()
    if p == 1.0:
        return -state_pm_one.clone()

    rand = torch.rand(state_pm_one.shape, device=state_pm_one.device, generator=generator)
    mask = rand < p
    out = state_pm_one.clone()
    out[mask] = -out[mask]
    return out


def _sign_pm_one(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))


@dataclass
class HopfieldNetwork:
    n_units: int
    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        self.n_units = int(self.n_units)
        self.W = torch.zeros((self.n_units, self.n_units), device=self.device, dtype=self.dtype)
        self.theta = torch.zeros((self.n_units,), device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def fit_hebbian(self, patterns_pm_one: torch.Tensor) -> "HopfieldNetwork":
        """
        Train weights using the classic Hebbian rule:
            W = (1/N) * sum_p x_p x_p^T, with diag(W) = 0
        where x_p in {-1, +1}^N.
        """
        if patterns_pm_one.ndim != 2:
            raise ValueError(f"patterns must have shape (P, N), got {tuple(patterns_pm_one.shape)}")
        if patterns_pm_one.shape[1] != self.n_units:
            raise ValueError(f"patterns N={patterns_pm_one.shape[1]} does not match n_units={self.n_units}")

        x = to_pm_one(patterns_pm_one).to(device=self.device, dtype=self.dtype)
        self.W = (x.T @ x) / float(self.n_units)
        self.W.fill_diagonal_(0)
        return self

    @torch.no_grad()
    def fit_pseudoinverse(self, patterns_pm_one: torch.Tensor, *, ridge: float = 1e-4) -> "HopfieldNetwork":
        """
        Train weights using the pseudoinverse learning rule.

        This rule stores the given patterns as (approximate) fixed points and tends to work much
        better than Hebbian learning for highly-correlated patterns like binarized MNIST.
        """
        if patterns_pm_one.ndim != 2:
            raise ValueError(f"patterns must have shape (P, N), got {tuple(patterns_pm_one.shape)}")
        if patterns_pm_one.shape[1] != self.n_units:
            raise ValueError(f"patterns N={patterns_pm_one.shape[1]} does not match n_units={self.n_units}")
        if ridge < 0:
            raise ValueError(f"ridge must be >= 0, got {ridge}")

        x = to_pm_one(patterns_pm_one).to(device=self.device, dtype=torch.float64)
        p = int(x.shape[0])
        if p == 0:
            raise ValueError("Need at least 1 pattern")

        gram = x @ x.T  # (P, P)
        if ridge > 0:
            gram = gram + float(ridge) * torch.eye(p, device=gram.device, dtype=gram.dtype)

        # W = X^T (X X^T)^-1 X
        solved = torch.linalg.solve(gram, x)  # (P, N)
        w = x.T @ solved  # (N, N)
        w.fill_diagonal_(0)

        self.W = w.to(device=self.device, dtype=self.dtype)
        self.theta.zero_()
        return self

    @torch.no_grad()
    def energy(self, state_pm_one: torch.Tensor) -> torch.Tensor:
        """Compute Hopfield energy E(s) for a single state vector `s` in {-1, +1}^N."""
        s = to_pm_one(state_pm_one).to(device=self.device, dtype=self.dtype).view(-1)
        if s.numel() != self.n_units:
            raise ValueError(f"state has {s.numel()} units, expected {self.n_units}")
        return -0.5 * (s @ (self.W @ s)) + (self.theta @ s)

    @torch.no_grad()
    def step(
        self,
        state_pm_one: torch.Tensor,
        *,
        rule: UpdateRule = "sync",
        async_updates: int = 1,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """
        Apply one update step.

        - rule="sync": synchronous update over all units.
        - rule="async": asynchronous updates of `async_updates` randomly-chosen units.
        """
        s = to_pm_one(state_pm_one).to(device=self.device, dtype=self.dtype).view(-1)
        if s.numel() != self.n_units:
            raise ValueError(f"state has {s.numel()} units, expected {self.n_units}")

        if rule == "sync":
            field = self.W @ s - self.theta
            return _sign_pm_one(field)

        if rule != "async":
            raise ValueError(f"Unknown rule: {rule}")
        if async_updates <= 0:
            raise ValueError(f"async_updates must be > 0, got {async_updates}")

        out = s.clone()
        # Sequential (in-place) updates on randomly sampled indices.
        indices = torch.randint(0, self.n_units, (async_updates,), device=out.device, generator=generator)
        for idx in indices.tolist():
            field_i = (self.W[idx] @ out) - self.theta[idx]
            out[idx] = 1.0 if float(field_i.item()) >= 0.0 else -1.0
        return out
