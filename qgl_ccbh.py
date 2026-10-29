#!/usr/bin/env python3
"""
qgl_ccbh.py -- a TOY of "cosmologically coupled black holes" (CCBH) as a source of dark energy.

THIS IS OUR OWN TOY.  The idea comes from Croker, Weiner, Farrah et al. and was tested against DESI DR2 by
Croker et al. (arXiv:2405.12282) and the DESI collaboration (PRL 135, 081003, "Positive Neutrino Masses with DESI
DR2 via Matter Conversion to Dark Energy").  Those papers' background equations were not available to me, so the
bookkeeping below is a minimal, self-consistent version written from the stated physics, not a reproduction.

Physics used (all stated in Ghodla, Easther, Briel & Eldridge, arXiv:2306.08199, Eqs. 1 and 5):

    m(a) = m_i (a / a_i)^k        dm/dt = k H m          (k = 3: the Farrah et al. value)

so a population of N comoving black holes has physical energy density  rho ~ a^-3 * a^k = a^(k-3):  constant for
k = 3, i.e. w = -1.

Bookkeeping.  Matter is converted into such black holes at a rate set by the cosmic star formation rate density
(Madau & Dickinson 2014 form).  With comoving mass in matter Om - C(a) and in coupled black holes B(a):

    dC/dln a = eps * psi(a) / H(a)              (psi per COMOVING volume; eps = fraction of formed stellar mass
                                                 that ends up in coupled black holes)
    dB/dln a = dC/dln a + k B
    E^2(a)   = [ (Om - C) + B ] / a^3 + OL ,     OL fixed by E(1) = 1 (iterated)

eps = 0 is flat LCDM exactly.  For k = 3 the growth exactly undoes the dilution, so the physical density of the
black holes is eps * int psi dt / a'^3: every formation epoch counts at its *physical* density.  Even so,
with the star formation history above the toy needs eps ~ 3.7 (k = 3, Om = 0.3, h = 0.7) to supply
Omega_DE ~ 0.7 (see `required_efficiency`), i.e. several times all the stellar mass formed, so in this
bookkeeping it does not by itself explain dark energy, and nothing here tests the
papers' models, which must differ in detail.

Single-black-hole consequences (closed form, tested): mass growth above; spin, if growth adds no angular momentum,
a_BH ~ (a / a_i)^(-2k) (Ghodla et al.), which for k = 3 and a / a_i = 2 is 0.016.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

# 1/H0 = 9.778 Gyr / h ;  rho_crit,0 = 2.775e11 h^2 Msun/Mpc^3
_GYR_PER_INV_H0 = 9.778
_RHO_CRIT_H2 = 2.775e11


# --------------------------------------------------------------------------
#  A single coupled black hole
# --------------------------------------------------------------------------

def bh_mass(a, a_i: float, m_i: float = 1.0, k: float = 3.0):
    """m(a) = m_i (a / a_i)^k  for a >= a_i."""
    a = np.asarray(a, dtype=float)
    return m_i * (a / a_i) ** k


def bh_spin(a, a_i: float, spin_i: float = 1.0, k: float = 3.0):
    """Dimensionless spin if the mass growth adds no angular momentum: spin_i (a / a_i)^(-2k)."""
    return spin_i * (np.asarray(a, dtype=float) / a_i) ** (-2.0 * k)


def density_exponent(k: float) -> float:
    """rho_BH ~ a^(k-3): the power law of a comoving population of coupled black holes."""
    return k - 3.0


# --------------------------------------------------------------------------
#  Star formation history
# --------------------------------------------------------------------------

def sfrd_madau_dickinson(a):
    """psi(z) = 0.015 (1+z)^2.7 / (1 + ((1+z)/2.9)^5.6)  Msun / yr / Mpc^3."""
    opz = 1.0 / np.asarray(a, dtype=float)
    return 0.015 * opz ** 2.7 / (1.0 + (opz / 2.9) ** 5.6)


def sfrd_code_units(a, h: float = 0.7):
    """psi in units of rho_crit,0 H0: psi * (9.778e9 yr / h) / (2.775e11 h^2 Msun/Mpc^3)."""
    return sfrd_madau_dickinson(a) * (_GYR_PER_INV_H0 * 1e9 / h) / (_RHO_CRIT_H2 * h ** 2)


# --------------------------------------------------------------------------
#  Background with matter -> coupled-black-hole conversion
# --------------------------------------------------------------------------

@dataclass
class CCBHBackground:
    Om: float
    eps: float
    k: float
    h: float
    OL: float
    a: np.ndarray
    C: np.ndarray
    B: np.ndarray

    def _interp(self, a, y):
        return CubicSpline(np.log(self.a), y)(np.log(np.asarray(a, dtype=float)))

    def E2(self, a):
        a = np.asarray(a, dtype=float)
        return (self.Om - self._interp(a, self.C) + self._interp(a, self.B)) / a ** 3 + self.OL

    @property
    def converted(self) -> float:
        """Comoving mass converted into black holes by today, in rho_crit,0 units."""
        return float(self.C[-1])

    @property
    def omega_bh(self) -> float:
        """Coupled-black-hole density today in rho_crit,0 units."""
        return float(self.B[-1])


def ccbh_background(Om: float = 0.3, eps: float = 0.0, k: float = 3.0, h: float = 0.7,
                    a_min: float = 1e-3, n: int = 800, tol: float = 1e-13) -> CCBHBackground:
    """
    Flat background with conversion efficiency eps.  OL is iterated so that E(1) = 1 (the cosmological constant
    left over after the matter and coupled black holes are accounted for).  eps = 0 gives flat LCDM.
    """
    grid = np.linspace(np.log(a_min), 0.0, n)
    OL = 1.0 - Om
    for _ in range(60):
        def rhs(x, y, OL=OL):
            a = np.exp(x)
            C, B = y
            E2 = (Om - C + B) / a ** 3 + OL
            if E2 <= 0.0:
                raise FloatingPointError("E^2 <= 0 during integration")
            dC = eps * sfrd_code_units(a, h) / np.sqrt(E2)
            return [dC, dC + k * B]
        sol = solve_ivp(rhs, [grid[0], 0.0], [0.0, 0.0], t_eval=grid, method="DOP853", rtol=1e-11, atol=1e-16)
        if not sol.success:
            raise RuntimeError(sol.message)
        C, B = sol.y
        OL_new = 1.0 - (Om - C[-1] + B[-1])
        if abs(OL_new - OL) < tol:
            OL = OL_new
            break
        OL = OL_new
    else:
        raise RuntimeError("OL iteration did not converge")
    return CCBHBackground(Om, eps, k, h, OL, np.exp(grid), C, B)


def required_efficiency(Om: float = 0.3, k: float = 3.0, h: float = 0.7, target: Optional[float] = None) -> float:
    """
    The eps that makes the coupled black holes supply `target` of the critical density today (default: all of
    1 - Om, i.e. OL = 0).  An answer above 1 means more than all the stellar mass formed would have to be coupled.
    """
    target = 1.0 - Om if target is None else target
    f = lambda e: ccbh_background(Om, e, k, h).omega_bh - target
    hi = 1.0
    while f(hi) < 0.0 and hi < 1e6:
        hi *= 2.0
    return brentq(f, 0.0, hi, xtol=1e-10, rtol=1e-10)


def main() -> None:
    import qgl_flrw as fl
    print("stellar mass formed by today in rho_crit units (eps = 1, k = 3):",
          f"{ccbh_background(0.3, 1.0, 3.0).converted:.4f}")
    for eps in (0.0, 0.5, 1.0):
        bg = ccbh_background(0.3, eps, 3.0)
        z = np.linspace(0.02, 1.5, 30)
        dmu = fl.distance_modulus(bg.E2, z) - fl.distance_modulus(lambda x: fl.lcdm_E2(x, 0.3), z)
        print(f"eps={eps:3.1f}: converted={bg.converted:.4f}  Omega_bh(today)={bg.omega_bh:.4f}  OL={bg.OL:.4f}"
              f"  rms dmu vs LCDM = {np.sqrt(np.mean(dmu ** 2)):.4f}")
    print(f"efficiency needed for Omega_bh = 0.7 with k = 3: {required_efficiency():.1f}")
    print(f"spin after a/a_i = 2 (k=3): {float(bh_spin(2.0, 1.0)):.4f}")


if __name__ == "__main__":
    main()
