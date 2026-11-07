# QUANTGRAV

Working notes for the Quantum Gravity Lab fork: the E8 → quasicrystal reference thread and a set of 2026 quantum-gravity papers and news items.
Status key: **read** = full text or abstract retrieved; **news only** = press coverage read, paper not; **unread** = could not be retrieved.
Nothing here is a claim that QGL implements or tests any of these theories. Links to QGL code are modelling suggestions.

---

## 1. The E8 / quasicrystal thread

Chain described by Quantum Gravity Research (QGR):

```
E8 lattice (8D, 240 roots, Gosset 4_21 polytope)
   |  cut-and-project  (or Hopf mapping)
   v
Elser-Sloane quasicrystal (4D; two 600-cells, sizes in golden ratio)
   |  five 3D "tetragrid" subspaces, rotated 15.522 deg relative to each other
   v
Compound Quasicrystal (3D)   -- stated to be an exact subspace of -->
Quasicrystalline Spin Network (QSN): FCC lattice with Fibonacci spacing, filled with up/down tetrahedra
```

The QGR page frames the QSN as a Planck-scale "pixel" toy model, not established physics.

### References

| Ref | Status | Notes |
|---|---|---|
| Fang Fang (2017), "The Quasicrystalline Spin Network", video: https://www.youtube.com/watch?v=_g2JWp6F_xQ | unread | YouTube fetch failed. Needs a transcript or description. |
| QGR, "A deep link between 3D and 8D": https://quantumgravityresearch.org/portfolio/a-deep-link-between-3d-and-8d/ | read | Source of the chain above. Cites no peer-reviewed papers by name and does not mention Fang Fang. |
| J.F. Sadoc, R. Mosseri (1993), "The E8 lattice and quasicrystals: geometry, number theory and quasicrystals": https://hal.science/hal-04898979v1 | unread | HAL returned a bot-protection page. **Citation check:** a MaRDI record lists this title as *J. Phys. A* 26 (1993), DOI 10.1088/0305-4470/26/8/009, whereas the README DOI 10.1016/0022-3093(93)90351-W is a *J. Non-Cryst. Solids* prefix (Elsevier page not reachable, unverified). |
| Amaral, Clawson, Irwin, "Quasicrystalline Spin Foam with Matter: Definitions and Examples": https://arxiv.org/abs/2306.01964 | read (partial) | Cut-and-project model sets, SU(2) coherent states, area/volume spectra, restricted EPRL-style amplitude. Explicit E8 projection and window not in the retrieved text; the authors point to a Wolfram notebook. |

### Cut-and-project, as defined in 2306.01964

Mother lattice L in R^d x R^d', projections pi (physical) and pi_perp (internal), compact window K in R^d'.
Model set: `{ pi(y) : y in L, pi_perp(y) in lambda*K + xi_perp }`.

---

## 2. 2026 quantum-gravity references

### 2.1 Time and causality in quantum gravity — **read (abstract)**
Charis Anastopoulos, arXiv:2609.22890 (submitted 19 Sep 2026). https://arxiv.org/abs/2609.22890
Survey. Argues the problem of time is not just a by-product of quantizing GR: tensions among causal structure, clocks, observables and measurement already exist in classical mechanics, relativity, QM and QFT. Splits "time" into causal order, temporal duration and the present, then classifies quantum-gravity programmes (Wheeler-DeWitt, histories, background-dependent, causal sets, twistors) by how they treat it. Final section: linearized gravity and weak-gravity quantum systems, where the problem persists away from the Planck scale.
*QGL angle:* QGL runs on an external orbit time, so it sits in the "keeps an external time" class. A short note on that choice would fit in `QGL.md`.

### 2.2 Relativity of spacetime superpositions — **read (ScienceDaily summary)**
Foo, Suryaatmadja, Mann, Zych, "Relativity and decoherence of spacetime superpositions", *npj Quantum Information* 12(1), 2026 (journal reference as given by ScienceDaily; DOI not independently checked).
https://www.sciencedaily.com/releases/2026/08/260828005226.htm
Theory. Some setups described as a superposition of gravitational fields can equally be described as quantum particles in superposition on a classical spacetime; both descriptions are valid. So some proposed signatures of quantum gravity are ambiguous, and the framework identifies which would genuinely need quantum gravity. It does not show gravity is classical.
*QGL angle:* directly relevant to how far the Jaynes-Cummings sensor coupling can be read as "quantum gravity" versus quantum matter on a classical background. QGL's README already says it is not a quantum-gravity theory; this gives a citation for why.

### 2.3 Quantum phase of a freely falling atom — **news only**
Oxford press release: https://www.ox.ac.uk/news/2026-08-28-scientists-observe-einsteins-gravity-in-the-quantum-world (page failed to load; content below is from [The Quantum Insider](https://thequantuminsider.com/2026/09/02/scientists-observe-einsteins-gravity-in-the-quantum-world/)).
Reported: first measurement of the predicted quantum phase of a freely falling atom, using cold rubidium atoms on an atom chip in a "Quantum Galileo Interferometer" at Ben-Gurion University, with Ulm and Oxford (co-authors include Penrose and Vedral). One arm is held against gravity by magnetic fields, the other launched and allowed to fall; the interference phase is read out. Consistent with the equivalence principle under the conditions tested; does not show gravity is quantum. Published in *Science Advances* (2 Sep 2026 per the article; no DOI or numbers given there).
Note: this is a different story from 2.2 even though the dates are close. The ScienceDaily link for 2.2 (260828005226) covers the Foo et al. theory paper, not this experiment.
*QGL angle:* an atom-interferometer phase is a clean example of a "geometry to quantum phase" observable, the same kind of mapping `unified_*` and the JC sensor model.

### 2.4 Quadratic gravity and the Big Bang — **read (arXiv abstract); mapping to the phys.org link is inferred**
phys.org: https://phys.org/news/2026-03-quadratic-gravity-theory-reshapes-quantum.html (page failed to load). A search matched the headline to K. Sravan Kumar and J. Marto, "Quantum (quadratic) gravity: replacing the massive tensor ghost with an inverted harmonic oscillator-like instability", arXiv:2603.07150 (https://arxiv.org/abs/2603.07150). I have not confirmed that the phys.org article cites that exact paper.
Content of the arXiv paper: Stelle (1977) showed quadratic gravity is the unique renormalizable quantum gravity in 4D, at the cost of a massive spin-2 ghost. The authors propose the mode is instead an inverted-harmonic-oscillator-like instability: quantized in direct-sum QFT, it has no particle interpretation, stays off-shell, and decouples from matter. Claimed predictions: new bounds on the primordial tensor-to-scalar ratio and parity-asymmetric features at large angular scales.
*QGL angle:* the repo already has `qgl_pancake.py`, `qgl_bkl.py`, `qgl_kasner.py` and `qgl_flrw.py` for anisotropic and singular early-universe geometry; this is the natural comparison for a quadratic-gravity (R + R^2 + Weyl^2) extension.

### 2.5 Thermodynamics of Gravity from Entropy — **news only**
Ginestra Bianconi, "Thermodynamics of the gravity from entropy theory", *Physical Review D* 114(2), 2026 (reference as given by ScienceDaily; DOI not independently checked).
https://www.sciencedaily.com/releases/2026/07/260719035947.htm
Theory. Applies Gravity from Entropy (GfE) to FRW cosmologies. Total entropy may rise with expansion while entropy per volume falls. Under weak curvature GfE reproduces GR; otherwise it yields a dynamical dark-energy term, treated as internal energy, with Quantum Geometric Relative Entropy as local entropy density, and effective temperature and pressure. The article gives no numbers.
*QGL angle:* `qgl_flrw.py` and `qgl_horizon.py` (apparent-horizon thermodynamics, cf. the Capozziello et al. reference already in the README) are where a dynamical dark-energy term of this kind would be compared.

### 2.6 Physical Review D, DOI 10.1103/kbkv-vs5t — **unread**
https://journals.aps.org/prd/abstract/10.1103/kbkv-vs5t
The APS page failed to load and a search on the DOI returned nothing relevant. Title, authors and abstract are unknown. If you paste the abstract I'll add it.

---

## 3. Open items

1. Transcript or description of the Fang Fang video.
2. Sadoc-Mosseri abstract, and which DOI is correct (see table in section 1).
3. Abstract for the PRD paper (2.6).
4. Confirm the phys.org to arXiv:2603.07150 mapping (2.4).
5. Code: `qgl_e8.py` (E8 roots, Coxeter-plane H4 projection, 600-cell shells), see the module docstring for what is and is not checked.
