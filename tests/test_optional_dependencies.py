"""Regression tests for the signal-transform dependencies (requirements.txt).

PyWavelets and Kymatio are declared requirements, so these tests import them
directly (a missing extra is a real failure, not a silent skip). They also lock
in the SciPy ``sph_harm`` compatibility shim that lets Kymatio's public 1D API
import on SciPy >= 1.15.
"""

import unittest

import numpy as np


class PyWaveletsTest(unittest.TestCase):
    def test_wavedec_waverec_roundtrip(self) -> None:
        import pywt

        signal = np.cumsum(np.random.default_rng(0).standard_normal(128)).astype(float)
        coeffs = pywt.wavedec(signal, "db4", level=3)
        reconstructed = pywt.waverec(coeffs, "db4")[: len(signal)]

        self.assertEqual(len(coeffs), 4)
        np.testing.assert_allclose(reconstructed, signal, atol=1e-8)

    def test_soft_threshold_denoise_runs(self) -> None:
        import pywt

        signal = np.sin(np.linspace(0, 4 * np.pi, 64)) + 0.1 * np.random.default_rng(1).standard_normal(64)
        coeffs = pywt.wavedec(signal, "db1", level=1)
        coeffs[-1] = pywt.threshold(coeffs[-1], 0.1, mode="soft")
        denoised = pywt.waverec(coeffs, "db1")[: len(signal)]

        self.assertEqual(len(denoised), len(signal))


class SciPySphHarmShimTest(unittest.TestCase):
    def test_shim_restores_sph_harm_with_correct_values(self) -> None:
        from stable_grn_inference._compat import ensure_scipy_sph_harm

        ensure_scipy_sph_harm()
        import scipy.special as special

        self.assertTrue(hasattr(special, "sph_harm"))
        azimuth, polar = 0.7, 1.1
        # Y_1^0 = 0.5*sqrt(3/pi)*cos(polar); legacy signature sph_harm(m, n, theta=azimuth, phi=polar)
        analytic = 0.5 * np.sqrt(3 / np.pi) * np.cos(polar)
        self.assertAlmostEqual(complex(special.sph_harm(0, 1, azimuth, polar)).real, analytic, places=6)

    def test_shim_is_idempotent(self) -> None:
        from stable_grn_inference._compat import ensure_scipy_sph_harm

        ensure_scipy_sph_harm()
        # second call should be a no-op now that sph_harm exists
        self.assertFalse(ensure_scipy_sph_harm())


class KymatioScatteringTest(unittest.TestCase):
    def test_public_numpy_scattering1d_runs(self) -> None:
        from stable_grn_inference._compat import ensure_scipy_sph_harm

        ensure_scipy_sph_harm()
        from kymatio.numpy import Scattering1D

        scattering = Scattering1D(J=4, shape=256, Q=8)
        coefficients = scattering(np.cumsum(np.random.default_rng(0).standard_normal(256)).astype("float32"))

        self.assertEqual(coefficients.ndim, 2)
        self.assertEqual(coefficients.shape[1], 256 // (2 ** 4))


if __name__ == "__main__":
    unittest.main()
