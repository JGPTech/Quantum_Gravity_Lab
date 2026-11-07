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

### Fibonacci icosagrid sources (found while building the 3D step)

| Ref | Status | Notes |
|---|---|---|
| F. Fang, K. Irwin, "An Icosahedral Quasicrystal and E8 derived quasicrystals", arXiv:1511.07786 | read (formulas partly garbled in extraction) | Main source for the construction: Fibonacci spacing formula, golden rotation arccos((3 tau - 1)/4), 20G with 61 vertices, tetragrids, type I / type II cross-sections of the Elser-Sloane quasicrystal. The icosian-to-E8 table and the matrix H of the projection were not readable. |
| F. Fang, K. Irwin, "From the Fibonacci Icosagrid to E8", Part I (Crystals 14, 152, 2024) and Part II (Crystals 14, 194, 2024) | read (partial) | Part I: multigrid method, 20G from a tetragrid. Part II: Hopf-fibration view of E8 (240 roots as ten 24-cells), compound of five 24-cells, left/right isoclinic rotations, cuboctahedra, C5C, 4G and 20G. Several sections and all angle values were missing from the retrieved text. |

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

## 3. Code: what is built and what it shows

Modules: `qgl_e8.py` (E8, H4 projection), `qgl_cqc.py` (4D side: 600-cell, 24-cells, isoclinic rotations), `qgl_qsn.py` (3D side: tetragrids, 20G, Fibonacci icosagrid). Tests: `tests/test_e8.py`, `tests/test_cqc.py`, `tests/test_qsn.py`.

### Results

| Check | Result |
|---|---|
| 240 E8 roots project (Coxeter-plane H4 subspace) to two shells of 120 with radii in the golden ratio; outer shell has 12 neighbours per vertex | holds (qgl_e8) |
| Inscribed 24-cells of the 600-cell | 25; ten partitions into five; five of those are cycled by a left-type 5-fold isoclinic symmetry and five by a right-type one |
| Twelve cuboctahedral equators per 24-cell | holds |
| Rotate one equator of each of the five 24-cells onto a fixed hyperplane by the unique isoclinic rotation of the same side as the 5-fold symmetry | all five land on one cuboctahedron (12 distinct vertices), as the Part II text says for one chirality |
| Same, with the opposite side | 60 distinct vertices whose pairwise-distance multiset equals that of the compound of five cuboctahedra built directly from the icosahedral axes (both chiralities of the 5-fold symmetry) |
| 20G (five 4Gs) | 61 distinct vertices, matching the count in arXiv:1511.07786 |
| Fibonacci chain x_N = T(N + floor(N/tau + 1/2)/tau) | symmetric about 0, gaps {T, tau T}, first gap long, and the gap word is a factor of the L->LS, S->L substitution word |
| Uniform spacing | each tetragrid's vertices form an FCC lattice (every triple point carries the fourth plane) |
| 20G inside the Fibonacci icosagrid vertex set | all 61 points, at the scale where the first planes carry the cuboctahedron vertices; not at 0.9x, 1.1x, sqrt(2)x or 1/tau of that scale; also at tau x and tau^2 x (self-similarity, an observation) |
| Relative rotation of two of the five tetragrids | 72 degrees as sets; 44.4775 degrees modulo the full cube group, and 60 - 44.4775 = 15.5225 degrees = the golden rotation, to 4e-14, about a 3-fold axis |
| Tetrahedral cells of one tetragrid (regular tetrahedra bounded by one plane per family) | uniform spacing: one size, edge = FCC nearest-neighbour distance. Fibonacci spacing: four sizes, edge lengths in the ratio tau between neighbours, largest = the core scale of the 20G |
| Edge graph of the union of the five tetragrids' cells (region radius 12) | exactly 30 edge directions (5 tetragrids x 6), the origin vertex has degree 60, as reported in arXiv:1511.07786 |
| Vertex degrees in that graph | all cell sizes together: 6 to 60. Largest cells only: **3 to 60**, the range reported in the paper. Degree 3 needs a large enough region (it does not appear at radius 9, where the minimum is 5) |

### Inferred, not from a source

- Fibonacci offsets use alpha = 0 and beta = 1/2 (needed for the symmetry about the origin; the source's values were not readable).
- The five tetragrids are the five tetrahedra of one chirality in the compound inscribed in the dodecahedron.
- "Vertex" of the icosagrid: the union over the five tetragrids of points lying on at least 4 planes of one tetragrid (`min_planes=4`); `min_planes=3` (any triple point) is also computed. The core containment result holds for both.
- The 15.5225 degree angle appears here as 60 degrees minus the smallest relative rotation modulo the cube group. That this is what the source means by "golden rotation" is my reading, supported by the exact numerical identity, not by the text.
- Which cells make up "the icosagrid": the paper's degree range of 3 to 60 is reproduced only by the largest tetrahedra (edge = the core scale). I take that to be the structure meant by "unit-length" edges, but the text I could read does not say so; with all four sizes the minimum degree is 6.
- Pairwise-distance equality is a strong necessary test for congruence of the two 60-point sets, not a proof; I did not solve for the explicit 3D rotation.
- Chirality labels are +1 / -1 by the sign of the Pfaffian; which is "left" or "right" in the papers is not fixed.

### Not built or not tested

- The full 3D Compound Quasicrystal as an infinite structure, and the Elser-Sloane quasicrystal with its exact window (`model_set` in `qgl_e8.py` uses a spherical window and is not Elser-Sloane).
- The claim that the whole Compound Quasicrystal lies in the QSN: only the central 20G cluster (61 points) was tested.
- Type II cross-sections, "enrichment", the E8 to QSN mapping (icosian table), diffraction patterns, and the claim that vertices lie in the integer space a + b tau (not checked).
- Spin-network structure (nodes = tetrahedra, edges = triangles, spins and normals) is not constructed, so nothing here connects to the spin-foam amplitudes of arXiv:2306.01964. The tetrahedral cells and their edge graph now exist (`tetra_cells`, `icosagrid_graph`), which is the input that step would need.

## 4. Open items

1. Transcript or description of the Fang Fang video.
2. Sadoc-Mosseri abstract, and which DOI is correct (see table in section 1).
3. Abstract for the PRD paper (2.6).
4. Confirm the phys.org to arXiv:2603.07150 mapping (2.4).
5. Next code step if wanted: build the spin-network layer on the largest-cell graph (nodes = tetrahedra, edges = shared triangles) and compare with the definitions in arXiv:2306.01964; or check the a + b tau coordinate claim.
