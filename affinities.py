"""Exact, quadratic-space t-SNE affinities for small educational datasets."""

import math
import numpy as np


def matrix(values, name, columns=None):
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] < 3 or result.shape[1] < 1:
        raise ValueError(f"{name} must be a matrix with at least three rows")
    if columns is not None and result.shape[1] != columns:
        raise ValueError(f"{name} must have {columns} columns")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def squared_distances(values):
    # Subtraction avoids cancellation from ||a||^2 + ||b||^2 - 2*a.b.
    delta = values[:, None, :] - values[None, :, :]
    with np.errstate(over="ignore", invalid="ignore"):
        distances = np.einsum("ijk,ijk->ij", delta, delta)
    if not np.isfinite(distances).all():
        raise ValueError("pairwise distances overflow; rescale the input")
    return distances


def conditional_affinities(values, perplexity=10.0, tolerance=1e-7):
    """Binary-search each Gaussian precision to match a target entropy.

    Tied nearest neighbors can make small perplexities unattainable. The
    returned distribution remains valid; callers can inspect achieved values.
    """
    x = matrix(values, "data")
    n = len(x)
    if not math.isfinite(perplexity) or not 1 < perplexity < n:
        raise ValueError("perplexity must be finite and strictly between 1 and n")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be positive and finite")
    distances = squared_distances(x)
    result = np.zeros((n, n))
    target = math.log(perplexity)
    for i in range(n):
        mask = np.arange(n) != i
        d = distances[i, mask]
        d = d - d.min()
        scale = max(float(d.max()), np.finfo(float).tiny)
        d = d / scale
        low, high, beta = 0.0, math.inf, 1.0
        for _ in range(100):
            weights = np.exp(-beta * d)
            probs = weights / weights.sum()
            positive = probs > 0
            entropy = -float(np.sum(probs[positive] * np.log(probs[positive])))
            error = entropy - target
            if abs(error) < tolerance:
                break
            if error > 0:
                low = beta
                beta = beta * 2 if math.isinf(high) else (beta + high) / 2
            else:
                high = beta
                beta = (low + beta) / 2
        result[i, mask] = probs
    return result


def joint_affinities(values, perplexity=10.0):
    conditional = conditional_affinities(values, perplexity)
    return (conditional + conditional.T) / (2 * len(conditional))


def validate_joint(values):
    p = matrix(values, "joint affinities")
    if p.shape[0] != p.shape[1] or (p < 0).any():
        raise ValueError("joint affinities must be square and nonnegative")
    if not np.allclose(p, p.T, rtol=0, atol=1e-12):
        raise ValueError("joint affinities must be symmetric")
    if np.any(np.diag(p) != 0) or not np.isclose(p.sum(), 1, rtol=0, atol=1e-10):
        raise ValueError("joint affinities must have zero diagonal and total mass one")
    return p
