"""Seeded comparison of three attraction schedules; writes JSON and an SVG."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from affinities import joint_affinities
from optimizer import Settings, fit


def synthetic_data(seed=7, per_cluster=25):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 3, (3, 6))
    return (np.vstack([center + rng.normal(0, .65, (per_cluster, 6)) for center in centers]),
            np.repeat(np.arange(3), per_cluster))


def render_svg(results, labels):
    colors = ["#0072B2", "#D55E00", "#009E73"]
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="460" viewBox="0 0 1200 460">',
             '<rect width="1200" height="460" fill="white"/>',
             '<text x="24" y="28" font-family="sans-serif" font-size="18">t-SNE attraction schedules | synthetic data | independent panel scales</text>']
    for panel, (mode, result) in enumerate(results.items()):
        y = np.asarray(result["embedding"])
        span = np.ptp(y, axis=0)
        scaled = (y - y.min(axis=0)) / np.maximum(span, 1e-9)
        left = panel * 400
        parts.append(f'<text x="{left+24}" y="65" font-family="sans-serif" font-size="18">{mode}</text>')
        for point, label in zip(scaled, labels):
            cx, cy = left + 35 + point[0] * 330, 100 + point[1] * 290
            parts.append(f'<circle cx="{cx:.3f}" cy="{cy:.3f}" r="4" fill="{colors[int(label)]}" opacity="0.8"/>')
        kl = result["history"][-1]["kl_original_p"]
        parts.append(f'<text x="{left+24}" y="425" font-family="sans-serif" font-size="14">Final KL(P||Q): {kl:.5f}</text>')
    parts.append('</svg>')
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--early-steps", type=int, default=150)
    parser.add_argument("--perplexity", type=float, default=10)
    parser.add_argument("--threshold", type=float, default=0.0001)
    parser.add_argument("--exaggeration", type=float, default=6)
    parser.add_argument("--understate", type=float, default=.5)
    args = parser.parse_args()
    try:
        x, labels = synthetic_data()
        p = joint_affinities(x, args.perplexity)
        results = {}
        for mode in ("none", "standard", "selective"):
            settings = Settings(mode=mode, iterations=args.iterations, exaggeration_steps=args.early_steps,
                                threshold=args.threshold, exaggeration=args.exaggeration, understate=args.understate)
            y, history = fit(p, settings)
            results[mode] = {"settings": asdict(settings), "embedding": y.tolist(), "history": history}
        payload = {"synthetic": True, "labels_for_plot_only": labels.tolist(), "input_shape": list(x.shape),
                   "perplexity": args.perplexity, "results": results,
                   "caveat": "Selective attraction is an illustrative heuristic, not a guarantee of better embeddings."}
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "comparison.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        (args.output / "comparison.svg").write_text(render_svg(results, labels), encoding="utf-8")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({mode: result["history"][-1]["kl_original_p"] for mode, result in results.items()}, allow_nan=False))
    print(f"Wrote {args.output / 'comparison.json'} and {args.output / 'comparison.svg'}")


if __name__ == "__main__":
    main()
