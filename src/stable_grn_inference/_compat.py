"""Small third-party compatibility shims.

Currently this restores ``scipy.special.sph_harm``, which SciPy removed in 1.15+
(superseded by ``sph_harm_y`` with a swapped argument convention). Kymatio 0.3.0
still imports ``sph_harm`` at module load time in its 3D frontend, so importing
the public ``kymatio.numpy`` API crashes on modern SciPy even when only the 1D
scattering transform is needed. Calling :func:`ensure_scipy_sph_harm` before
importing kymatio repairs this without downgrading SciPy.
"""

from __future__ import annotations


def ensure_scipy_sph_harm() -> bool:
    """Alias the removed ``scipy.special.sph_harm`` to ``sph_harm_y`` if needed.

    The legacy signature is ``sph_harm(m, n, theta, phi)`` with ``theta`` the
    azimuthal angle and ``phi`` the polar angle. The replacement is
    ``sph_harm_y(n, m, polar, azimuth)``, so the alias maps
    ``sph_harm(m, n, theta, phi) -> sph_harm_y(n, m, phi, theta)``. This mapping
    is verified against the analytic spherical harmonics (Y_0^0, Y_1^0, Y_1^1).

    Returns
    -------
    bool
        ``True`` if a shim was installed, ``False`` if ``sph_harm`` already
        existed (older SciPy) or could not be provided.
    """
    import scipy.special as special

    if hasattr(special, "sph_harm"):
        return False
    if not hasattr(special, "sph_harm_y"):
        return False

    sph_harm_y = special.sph_harm_y

    def sph_harm(m, n, theta, phi):  # noqa: ANN001 - mirrors the legacy SciPy API
        """Legacy ``scipy.special.sph_harm`` compatibility wrapper."""
        return sph_harm_y(n, m, phi, theta)

    sph_harm.__doc__ = (special.sph_harm_y.__doc__ or "") + "\n\nLegacy sph_harm(m, n, theta, phi) compatibility alias."
    special.sph_harm = sph_harm
    return True
