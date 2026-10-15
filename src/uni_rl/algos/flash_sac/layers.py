"""FlashSAC layers and lightweight normalization helpers."""

from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F


def safe_tanh_log_det_jacobian(x: torch.Tensor) -> torch.Tensor:
    """Stable log|det J_tanh(x)| term."""
    return cast(torch.Tensor, 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x)))


class UnitLinear(nn.Module):
    """Linear layer with post-step weight normalization."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.w = nn.Linear(input_dim, output_dim, bias=False)
        nn.init.orthogonal_(self.w.weight, gain=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.w(x))

    def normalize_parameters(self) -> None:
        with torch.no_grad():
            self.w.weight.copy_(F.normalize(self.w.weight, dim=-1, eps=1e-8))


class UnitBatchNorm(nn.Module):
    """BatchNorm variant with normalized affine parameters."""

    running_mean: torch.Tensor
    running_var: torch.Tensor

    def __init__(self, input_dim: int, momentum: float = 0.01, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(input_dim))
        self.bias = nn.Parameter(torch.zeros(input_dim))
        self.register_buffer("running_mean", torch.zeros(input_dim))
        self.register_buffer("running_var", torch.ones(input_dim))
        self.momentum = momentum
        self.eps = eps

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        return F.batch_norm(
            x,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            training=training,
            momentum=self.momentum,
            eps=self.eps,
        )

    def normalize_parameters(self) -> None:
        with torch.no_grad():
            sqsum = torch.sum(self.weight * self.weight + self.bias * self.bias, dim=-1)
            norm_factor = math.sqrt(float(self.weight.shape[-1])) * torch.rsqrt(sqsum + 1e-8)
            self.weight.mul_(norm_factor)
            self.bias.mul_(norm_factor)


class UnitRMSNorm(nn.Module):
    """RMSNorm with unit-length scale vector."""

    def __init__(self, input_dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(input_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight

    def normalize_parameters(self) -> None:
        with torch.no_grad():
            sqsum = torch.sum(self.weight * self.weight, dim=-1)
            norm_factor = math.sqrt(float(self.weight.shape[-1])) * torch.rsqrt(sqsum + 1e-8)
            self.weight.mul_(norm_factor)


class FlashSACEmbedder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.norm = UnitBatchNorm(input_dim)
        self.w = UnitLinear(input_dim, hidden_dim)

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        return cast(torch.Tensor, self.w(self.norm(x, training=training)))


class FlashSACBlock(nn.Module):
    def __init__(self, hidden_dim: int, expansion: int = 4):
        super().__init__()
        self.w1 = UnitLinear(hidden_dim, hidden_dim * expansion)
        self.norm1 = UnitBatchNorm(hidden_dim * expansion)
        self.w2 = UnitLinear(hidden_dim * expansion, hidden_dim)
        self.norm2 = UnitBatchNorm(hidden_dim)

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        residual = x
        x = self.w1(x)
        x = self.norm1(x, training=training)
        x = F.relu(x)
        x = self.w2(x)
        x = self.norm2(x, training=training)
        x = F.relu(x)
        return x + residual


def validate_policy_std(
    std_mode: str | None,
    initial_normalized_std: float,
    log_std_min: float,
    log_std_max: float,
) -> None:
    """Validate the Gaussian scale before constructing an actor or environment."""
    if std_mode not in (None, "state_independent", "state_dependent"):
        raise ValueError(f"Unsupported FlashSAC std mode: {std_mode!r}")
    if not (
        math.isfinite(log_std_min) and math.isfinite(log_std_max) and log_std_min < log_std_max
    ):
        raise ValueError("FlashSAC log_std_min and log_std_max must be finite and increasing")
    if std_mode is not None:
        if not math.isfinite(initial_normalized_std) or initial_normalized_std <= 0:
            raise ValueError("FlashSAC initial_normalized_std must be finite and positive")
        if not log_std_min < math.log(initial_normalized_std) < log_std_max:
            raise ValueError("FlashSAC initial log std must lie strictly inside the log std bounds")


class NormalTanhPolicy(nn.Module):
    """Bounded Gaussian scale; an explicit std mode enables zero-gain residual heads.

    ``None`` retains the original heads and checkpoint parameter names.
    """

    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        log_std_min: float = -10.0,
        log_std_max: float = 2.0,
        *,
        std_mode: str | None = None,
        initial_normalized_std: float = 0.02,
    ):
        super().__init__()
        validate_policy_std(std_mode, initial_normalized_std, log_std_min, log_std_max)
        self.std_mode = std_mode
        self.mean_w = UnitLinear(hidden_dim, action_dim)
        self.mean_bias = nn.Parameter(torch.zeros(action_dim))
        self.std_w = UnitLinear(hidden_dim, action_dim) if std_mode != "state_independent" else None
        self.std_bias = nn.Parameter(torch.zeros(action_dim))
        self.mean_gain = nn.Parameter(torch.zeros(action_dim)) if std_mode is not None else None
        self.std_gain = (
            nn.Parameter(torch.zeros(action_dim)) if std_mode == "state_dependent" else None
        )
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        if std_mode is not None:
            normalized_log_std = (
                2.0 * (math.log(initial_normalized_std) - log_std_min) / (log_std_max - log_std_min)
                - 1.0
            )
            nn.init.constant_(self.std_bias, math.atanh(normalized_log_std))

    def get_mean_and_std(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mean_gain is None:
            # Preserve the original computation for existing checkpoints.
            assert self.std_w is not None
            mean = F.linear(x, self.mean_w.w.weight, self.mean_bias)
            raw_log_std = F.linear(x, self.std_w.w.weight, self.std_bias)
        else:
            mean = self.mean_w(x) * self.mean_gain + self.mean_bias
            raw_log_std = self.std_bias.expand_as(mean)
            if self.std_w is not None:
                assert self.std_gain is not None
                raw_log_std = raw_log_std + self.std_w(x) * self.std_gain
        log_std = self.log_std_min + (self.log_std_max - self.log_std_min) * 0.5 * (
            1.0 + torch.tanh(raw_log_std)
        )
        std = torch.exp(log_std)
        return mean, std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        mean, std = self.get_mean_and_std(x)
        # ``std`` is positive by construction (exp of bounded log_std). The
        # distribution's generic validation performs a GPU-to-host truth check,
        # which CUDA forbids while this actor is captured into an optimizer graph.
        dist = torch.distributions.Normal(mean, std, validate_args=False)
        raw_action = dist.rsample()
        tanh_action = torch.tanh(raw_action)
        log_prob = dist.log_prob(raw_action)
        log_prob = log_prob - safe_tanh_log_det_jacobian(raw_action)
        log_prob = log_prob.sum(dim=-1)
        return tanh_action, {"log_prob": log_prob, "mean": mean, "std": std}


class EnsembleUnitLinear(nn.Module):
    def __init__(self, num_ensemble: int, input_dim: int, output_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_ensemble, output_dim, input_dim))
        for idx in range(num_ensemble):
            nn.init.orthogonal_(self.weight[idx], gain=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("nbi,noi->nbo", x, self.weight)

    def normalize_parameters(self) -> None:
        with torch.no_grad():
            self.weight.copy_(F.normalize(self.weight, dim=-1, eps=1e-8))


class EnsembleUnitBatchNorm(nn.Module):
    running_mean: torch.Tensor
    running_var: torch.Tensor

    def __init__(
        self, num_ensemble: int, input_dim: int, momentum: float = 0.01, eps: float = 1e-5
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_ensemble, input_dim))
        self.bias = nn.Parameter(torch.zeros(num_ensemble, input_dim))
        self.register_buffer("running_mean", torch.zeros(num_ensemble, input_dim))
        self.register_buffer("running_var", torch.ones(num_ensemble, input_dim))
        self.momentum = momentum
        self.eps = eps

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        if training:
            mean = x.mean(dim=1, keepdim=True)
            var = x.var(dim=1, correction=0, keepdim=True)
            with torch.no_grad():
                batch_size = max(x.shape[1], 1)
                correction = batch_size / max(batch_size - 1, 1)
                self.running_mean.lerp_(mean.squeeze(1).float(), self.momentum)
                self.running_var.lerp_((var.squeeze(1) * correction).float(), self.momentum)
            normed = (x - mean) * torch.rsqrt(var + self.eps)
        else:
            normed = (x - self.running_mean.unsqueeze(1)) * torch.rsqrt(
                self.running_var.unsqueeze(1) + self.eps
            )
        return normed * self.weight.unsqueeze(1) + self.bias.unsqueeze(1)

    def normalize_parameters(self) -> None:
        with torch.no_grad():
            sqsum = torch.sum(
                self.weight * self.weight + self.bias * self.bias, dim=-1, keepdim=True
            )
            norm_factor = math.sqrt(float(self.weight.shape[-1])) * torch.rsqrt(sqsum + 1e-8)
            self.weight.mul_(norm_factor)
            self.bias.mul_(norm_factor)


class EnsembleUnitRMSNorm(nn.Module):
    def __init__(self, num_ensemble: int, input_dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_ensemble, input_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight.unsqueeze(1)

    def normalize_parameters(self) -> None:
        with torch.no_grad():
            sqsum = torch.sum(self.weight * self.weight, dim=-1, keepdim=True)
            norm_factor = math.sqrt(float(self.weight.shape[-1])) * torch.rsqrt(sqsum + 1e-8)
            self.weight.mul_(norm_factor)


class EnsembleFlashSACEmbedder(nn.Module):
    def __init__(self, num_ensemble: int, input_dim: int, hidden_dim: int):
        super().__init__()
        self.norm = EnsembleUnitBatchNorm(num_ensemble, input_dim)
        self.w = EnsembleUnitLinear(num_ensemble, input_dim, hidden_dim)

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        return cast(torch.Tensor, self.w(self.norm(x, training=training)))


class EnsembleFlashSACBlock(nn.Module):
    def __init__(self, num_ensemble: int, hidden_dim: int, expansion: int = 4):
        super().__init__()
        self.w1 = EnsembleUnitLinear(num_ensemble, hidden_dim, hidden_dim * expansion)
        self.norm1 = EnsembleUnitBatchNorm(num_ensemble, hidden_dim * expansion)
        self.w2 = EnsembleUnitLinear(num_ensemble, hidden_dim * expansion, hidden_dim)
        self.norm2 = EnsembleUnitBatchNorm(num_ensemble, hidden_dim)

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        residual = x
        x = self.w1(x)
        x = self.norm1(x, training=training)
        x = F.relu(x)
        x = self.w2(x)
        x = self.norm2(x, training=training)
        x = F.relu(x)
        return x + residual


class EnsembleCategoricalValue(nn.Module):
    support: torch.Tensor

    def __init__(
        self,
        num_ensemble: int,
        hidden_dim: int,
        num_bins: int,
        min_v: float,
        max_v: float,
    ):
        super().__init__()
        self.logit_w = EnsembleUnitLinear(num_ensemble, hidden_dim, num_bins)
        self.logit_bias = nn.Parameter(torch.zeros(num_ensemble, num_bins))
        self.register_buffer("support", torch.linspace(min_v, max_v, num_bins))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        logits = self.logit_w(x) + self.logit_bias.unsqueeze(1)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        support = self.support.view(1, 1, -1)
        values = cast(torch.Tensor, torch.sum(probs * support, dim=-1))
        return values, {"log_prob": log_probs}
