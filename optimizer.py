"""Compare ordinary t-SNE attraction with two temporary exaggeration rules."""

from dataclasses import dataclass
import math
import numpy as np
from affinities import matrix, squared_distances, validate_joint


@dataclass(frozen=True)
class Settings:
    mode: str = "standard"
    iterations: int = 500
    exaggeration_steps: int = 150
    exaggeration: float = 6.0
    threshold: float = 0.0001
    understate: float = 0.5
    learning_rate: float = 50.0
    seed: int = 7

    def validate(self):
        if self.mode not in {"none", "standard", "selective"}:
            raise ValueError("mode must be none, standard, or selective")
        for name in ("iterations", "exaggeration_steps", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.iterations < 1 or self.exaggeration_steps > self.iterations:
            raise ValueError("require iterations >= 1 and exaggeration_steps <= iterations")
        for name in ("exaggeration", "threshold", "understate", "learning_rate"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.exaggeration < 1 or self.threshold < 0:
            raise ValueError("exaggeration must be >= 1 and threshold >= 0")
        if not 0 < self.understate <= 1 or self.learning_rate <= 0:
            raise ValueError("require 0 < understate <= 1 and learning_rate > 0")


def attractive_weights(p, iteration, settings):
    """Do not normalize: normalization would cancel uniform exaggeration."""
    settings.validate()
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise ValueError("iteration must be a nonnegative integer")
    if iteration >= settings.exaggeration_steps or settings.mode == "none":
        return p.copy()
    if settings.mode == "standard":
        return p * settings.exaggeration
    return p * np.where(p > settings.threshold, settings.exaggeration, settings.understate)


def energy_and_gradient(y, weights):
    """Energy = sum(W * log(1+d^2)) + log(sum(1/(1+d^2))).

    Its gradient is 4*sum_j((W_ij-Q_ij)*nu_ij*(y_i-y_j)). For
    normalized W=P, the energy differs from KL(P||Q) by a constant.
    For exaggerated W, it is deliberately not KL(W||Q).
    """
    y = matrix(y, "embedding", columns=2)
    w = np.asarray(weights, dtype=float)
    if w.shape != (len(y), len(y)) or not np.isfinite(w).all() or (w < 0).any():
        raise ValueError("weights must be a finite nonnegative n-by-n matrix")
    if not np.allclose(w, w.T, rtol=0, atol=1e-12) or np.any(np.diag(w) != 0):
        raise ValueError("weights must be symmetric with zero diagonal")
    distances = squared_distances(y)
    kernel = 1 / (1 + distances)
    np.fill_diagonal(kernel, 0)
    z = float(kernel.sum())
    if not math.isfinite(z) or z <= 0:
        raise ValueError("embedding has a degenerate normalization")
    q = kernel / z
    energy = float(np.sum(w * np.log1p(distances)) + np.log(z))
    forces = (w - q) * kernel
    gradient = 4 * (forces.sum(axis=1)[:, None] * y - forces @ y)
    return energy, gradient, q


def fit(p, settings=Settings(), initial=None):
    settings.validate()
    p = validate_joint(p)
    y = (np.random.default_rng(settings.seed).normal(0, 1e-4, (len(p), 2))
         if initial is None else matrix(initial, "initial embedding", columns=2).copy())
    if len(y) != len(p):
        raise ValueError("initial embedding and joint affinities need the same row count")
    y -= y.mean(axis=0)
    velocity, gains = np.zeros_like(y), np.ones_like(y)
    history = []
    for step in range(settings.iterations):
        w = attractive_weights(p, step, settings)
        _, gradient, _ = energy_and_gradient(y, w)
        gains = np.where(np.sign(gradient) != np.sign(velocity), gains + 0.2, gains * 0.8)
        gains = np.maximum(gains, 0.01)
        momentum = 0.5 if step < settings.exaggeration_steps else 0.8
        velocity = momentum * velocity - settings.learning_rate * gains * gradient
        y += velocity
        y -= y.mean(axis=0)
        if not np.isfinite(y).all():
            raise ValueError("optimizer diverged; reduce learning_rate")
        if step % 25 == 0 or step == settings.iterations - 1 or step == settings.exaggeration_steps:
            _, _, q = energy_and_gradient(y, p)
            positive = p > 0
            kl = float(np.sum(p[positive] * np.log(p[positive] / q[positive])))
            history.append({"iteration": step, "phase": "early" if step < settings.exaggeration_steps else "normal",
                            "attractive_mass": float(w.sum()), "kl_original_p": kl})
    return y, history
