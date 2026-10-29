#!/usr/bin/env python3
"""
qgl_jc_lindblad.py -- positivity-preserving decay for the QGL Jaynes-Cummings sensor.

`JCQuantumEngine._apply_decays` in unified_sandbox_core.py moves populations
along the diagonal of the density matrix (photon loss at kappa n, atomic decay
at gamma) but never damps the coherences.  The resulting matrix is not a valid
density matrix: after a few time units its smallest eigenvalue is about -0.012
and <sigma_z> falls below -1 (see tests/test_tokamak.py::test_core_decay_breaks_positivity).

`LindbladJCEngine` replaces only that step with the Kraus form of the two
Lindblad channels,

    L_a = sqrt(kappa) a            (cavity photon loss)
    L_s = sqrt(gamma) sigma_minus  (atomic decay)

    rho -> K0 rho K0^+ + sum_j K_j rho K_j^+,  K_j = sqrt(dt) L_j,
                                               K0  = 1 - (dt/2) sum_j L_j^+ L_j,

which is a sum of congruence transforms and so positive by construction.  The
Hamiltonian step is inherited unchanged.  The core module is left as it is.
"""
import numpy as np

from unified_sandbox_core import Config, JCQuantumEngine


class LindbladJCEngine(JCQuantumEngine):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        n, d = self.n_cavity, self.dim
        self._a = np.zeros((d, d))
        self._sm = np.zeros((d, d))
        for k in range(n):
            g, e = 2 * k, 2 * k + 1
            if e < d:
                self._sm[g, e] = 1.0                  # |k,g><k,e|
            if k >= 1:
                self._a[2 * (k - 1), g] = np.sqrt(k)  # |k-1,g><k,g|
                if e < d:
                    self._a[2 * (k - 1) + 1, e] = np.sqrt(k)
        self._aa = self._a.T @ self._a
        self._ss = self._sm.T @ self._sm
        self._I = np.eye(d)

    def _apply_decays(self, params, dt):
        kappa, gamma = float(params["kappa"]), float(params["gamma"])
        if dt <= 0.0:
            return
        K0 = self._I - 0.5 * dt * (kappa * self._aa + gamma * self._ss)
        rho = K0 @ self.rho @ K0.conj().T
        Ka = np.sqrt(kappa * dt) * self._a
        Ks = np.sqrt(gamma * dt) * self._sm
        rho += Ka @ self.rho @ Ka.conj().T + Ks @ self.rho @ Ks.conj().T
        self.rho = rho
