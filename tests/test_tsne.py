import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
import numpy as np

from affinities import conditional_affinities, joint_affinities
from demo import render_svg, synthetic_data
from optimizer import Settings, attractive_weights, energy_and_gradient, fit


class AffinityTests(unittest.TestCase):
    def setUp(self):
        self.x = np.random.default_rng(1).normal(size=(12, 4))

    def test_joint_distribution(self):
        p = joint_affinities(self.x, 4)
        np.testing.assert_allclose(p, p.T)
        np.testing.assert_array_equal(np.diag(p), 0)
        self.assertAlmostEqual(float(p.sum()), 1)
        self.assertTrue((p >= 0).all())

    def test_perplexities_match(self):
        p = conditional_affinities(self.x, 4)
        entropy = -np.sum(p * np.log(np.maximum(p, np.finfo(float).tiny)), axis=1)
        np.testing.assert_allclose(np.exp(entropy), 4, atol=1e-6)

    def test_duplicate_points_are_finite(self):
        p = joint_affinities(np.zeros((5, 3)), 2)
        self.assertTrue(np.isfinite(p).all())
        np.testing.assert_allclose(p.sum(), 1)

    def test_affinity_rejects_invalid_data(self):
        for x in ([[1, 2]], [[1], [2], [float("nan")]], [[1e300], [-1e300], [0]]):
            with self.subTest(x=x), self.assertRaises(ValueError):
                joint_affinities(x, 2)

    def test_perplexity_bounds(self):
        for perplexity in (0, 1, 12, float("nan"), float("inf")):
            with self.subTest(perplexity=perplexity), self.assertRaises(ValueError):
                joint_affinities(self.x, perplexity)


class OptimizerTests(unittest.TestCase):
    def setUp(self):
        self.p = joint_affinities(np.random.default_rng(3).normal(size=(6, 3)), 3)
        self.y = np.random.default_rng(4).normal(size=(6, 2))

    def finite_difference(self, w):
        _, actual, _ = energy_and_gradient(self.y, w)
        expected = np.zeros_like(self.y)
        epsilon = 1e-6
        for i in range(6):
            for j in range(2):
                plus, minus = self.y.copy(), self.y.copy()
                plus[i, j] += epsilon
                minus[i, j] -= epsilon
                expected[i, j] = (energy_and_gradient(plus, w)[0] - energy_and_gradient(minus, w)[0]) / (2 * epsilon)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=2e-9)

    def test_normal_gradient_finite_difference(self):
        self.finite_difference(self.p)

    def test_standard_gradient_finite_difference(self):
        self.finite_difference(attractive_weights(self.p, 0, Settings()))

    def test_selective_gradient_finite_difference(self):
        self.finite_difference(attractive_weights(self.p, 0, Settings(mode="selective", threshold=.03)))

    def test_standard_mass_is_not_normalized_away(self):
        w = attractive_weights(self.p, 0, Settings(exaggeration=7))
        self.assertAlmostEqual(float(w.sum()), 7)
        np.testing.assert_allclose(w, 7 * self.p)

    def test_threshold_equality_is_understated(self):
        p = np.array([[0, .1, .2], [.1, 0, .2], [.2, .2, 0]])
        w = attractive_weights(p, 0, Settings(mode="selective", threshold=.1, understate=.25, exaggeration=4))
        self.assertAlmostEqual(w[0, 1], .025)
        self.assertAlmostEqual(w[0, 2], .8)
        np.testing.assert_array_equal(np.diag(w), 0)

    def test_phase_transition_restores_original_exactly(self):
        for mode in ("standard", "selective"):
            settings = Settings(mode=mode, exaggeration_steps=3)
            self.assertFalse(np.array_equal(attractive_weights(self.p, 2, settings), self.p))
            np.testing.assert_array_equal(attractive_weights(self.p, 3, settings), self.p)
            np.testing.assert_array_equal(attractive_weights(self.p, 4, settings), self.p)

    def test_translation_invariance_and_zero_net_force(self):
        energy, grad, q = energy_and_gradient(self.y, self.p)
        shifted_energy, shifted_grad, _ = energy_and_gradient(self.y + 17, self.p)
        self.assertAlmostEqual(energy, shifted_energy)
        np.testing.assert_allclose(grad, shifted_grad, atol=1e-13)
        np.testing.assert_allclose(grad.sum(axis=0), 0, atol=1e-13)
        self.assertAlmostEqual(float(q.sum()), 1)

    def test_fit_deterministic_centered_and_phase_logged(self):
        config = Settings(mode="selective", iterations=30, exaggeration_steps=10)
        y, history = fit(self.p, config)
        again, repeated = fit(self.p, config)
        np.testing.assert_array_equal(y, again)
        self.assertEqual(history, repeated)
        np.testing.assert_allclose(y.mean(axis=0), 0, atol=1e-12)
        boundary = next(row for row in history if row["iteration"] == 10)
        self.assertEqual(boundary["phase"], "normal")
        self.assertAlmostEqual(boundary["attractive_mass"], 1)

    def test_zero_early_steps_equals_none(self):
        base = Settings(mode="none", iterations=15, exaggeration_steps=0)
        expected, _ = fit(self.p, base)
        for mode in ("standard", "selective"):
            actual, _ = fit(self.p, replace(base, mode=mode))
            np.testing.assert_array_equal(actual, expected)

    def test_invalid_settings_and_matrices(self):
        for config in (Settings(mode="other"), Settings(iterations=0), Settings(exaggeration_steps=501),
                       Settings(understate=0), Settings(threshold=-1), Settings(learning_rate=float("inf")),
                       Settings(seed=-1), Settings(iterations=True)):
            with self.subTest(config=config), self.assertRaises(ValueError):
                fit(self.p, config)
        for bad in (self.p * 2, self.p + np.eye(6), self.p[:, :5]):
            with self.assertRaises(ValueError):
                fit(bad)
        with self.assertRaises(ValueError):
            energy_and_gradient(self.y, np.ones((6, 6)))


class ArtifactTests(unittest.TestCase):
    def test_cli_json_and_svg(self):
        with tempfile.TemporaryDirectory() as directory:
            process = subprocess.run([sys.executable, "-B", "demo.py", "--iterations", "25", "--early-steps", "10", "--output", directory],
                                     capture_output=True, text=True, check=False)
            self.assertEqual(process.returncode, 0, process.stderr)
            result = json.loads((Path(directory) / "comparison.json").read_text(encoding="utf-8"))
            self.assertTrue(result["synthetic"])
            self.assertEqual(set(result["results"]), {"none", "standard", "selective"})
            root = ET.fromstring((Path(directory) / "comparison.svg").read_text(encoding="utf-8"))
            self.assertEqual(len(root.findall("{http://www.w3.org/2000/svg}circle")), 225)


if __name__ == "__main__":
    unittest.main()
