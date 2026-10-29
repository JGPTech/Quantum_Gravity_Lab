#!/usr/bin/env python3
"""
qgl_ensemble.py -- run many QGL orbits and tabulate how well they reproduce GR.

For each (p, e) in a grid (or a seeded random draw) it measures the perihelion
advance per orbit with qgl_precession.measure_precession and reports the ratio
to the 1PN value 6 pi G M / (c^2 p).  Optionally (--solve) it first solves for
the complexity_power that makes the ratio 1 and reports the residual.

Runs on CPU with multiprocessing; results are written to CSV and, with --plot,
to a PNG heat map.  Output is deterministic for a given seed.

Examples
--------
    python qgl_ensemble.py --p 60 100 200 --e 0.1 0.3 0.6 --out grid.csv
    python qgl_ensemble.py --random 40 --seed 1 --solve --plot ensemble.png
"""

import argparse
import itertools
import os
import sys
import warnings

import numpy as np
import pandas as pd

from unified_sandbox_core import Config
from qgl_precession import configure_orbit, measure_precession, solve_complexity_power

warnings.filterwarnings("ignore")


def run_one(job):
    """One orbit.  `job` is a dict so it pickles cleanly for multiprocessing."""
    cfg = configure_orbit(Config(), job["p"], job["e"])
    if job.get("complexity_power") is not None:
        cfg.complexity_power = float(job["complexity_power"])
    row = dict(p=job["p"], e=job["e"], k_rq=cfg.k_rq_ratio)
    if job.get("solve"):
        row["complexity_power"] = solve_complexity_power(cfg, n_orbits=job["n_orbits"])
        if row["complexity_power"] is None:
            row["complexity_power"] = np.nan
            cfg = configure_orbit(Config(), job["p"], job["e"])
    else:
        row["complexity_power"] = cfg.complexity_power
    m = measure_precession(cfg, n_orbits=job["n_orbits"])
    row.update(shift=m["shift"], scatter=m["scatter"], theory=m["theory"],
               ratio=m["ratio"], ok=m["ok"])
    return row


def make_jobs(args):
    if args.random:
        rng = np.random.default_rng(args.seed)
        ps = rng.uniform(args.p_range[0], args.p_range[1], args.random)
        es = rng.uniform(args.e_range[0], args.e_range[1], args.random)
        pairs = list(zip(ps.round(3), es.round(4)))
    else:
        pairs = list(itertools.product(args.p, args.e))
    return [dict(p=float(p), e=float(e), n_orbits=args.n_orbits, solve=args.solve,
                 complexity_power=args.complexity_power) for p, e in pairs]


def run_ensemble(jobs, workers=1):
    if workers > 1:
        import multiprocessing as mp
        with mp.Pool(workers) as pool:
            rows = pool.map(run_one, jobs)
    else:
        rows = [run_one(j) for j in jobs]
    return pd.DataFrame(rows).sort_values(["p", "e"]).reset_index(drop=True)


def summarise(df):
    ok = df[df["ok"]]
    out = dict(n=len(df), n_ok=len(ok))
    if len(ok):
        out.update(ratio_mean=ok["ratio"].mean(), ratio_min=ok["ratio"].min(),
                   ratio_max=ok["ratio"].max(),
                   worst_abs_err=(ok["ratio"] - 1).abs().max())
    return out


def plot(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ok = df[df["ok"]]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sc = ax.scatter(ok["p"], ok["e"], c=ok["ratio"], cmap="coolwarm", vmin=0.5, vmax=1.5,
                    s=70, edgecolor="k", lw=0.4)
    fig.colorbar(sc, label="measured / 1PN precession")
    ax.set_xlabel("semi-latus rectum p")
    ax.set_ylabel("eccentricity e")
    ax.set_title("QGL ensemble: precession vs GR (1 = exact)")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--p", type=float, nargs="+", default=[60, 100, 200])
    ap.add_argument("--e", type=float, nargs="+", default=[0.1, 0.3, 0.6])
    ap.add_argument("--random", type=int, default=0, help="N random (p, e) draws instead of a grid")
    ap.add_argument("--p-range", type=float, nargs=2, default=[50, 250])
    ap.add_argument("--e-range", type=float, nargs=2, default=[0.05, 0.7])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-orbits", type=int, default=4)
    ap.add_argument("--complexity-power", type=float, default=None)
    ap.add_argument("--solve", action="store_true", help="solve complexity_power per orbit")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1)))
    ap.add_argument("--out", default=None, help="CSV path")
    ap.add_argument("--plot", default=None, help="PNG path")
    args = ap.parse_args(argv)

    df = run_ensemble(make_jobs(args), workers=args.workers)
    pd.set_option("display.width", 140)
    print(df.to_string(index=False, float_format=lambda x: "%.5g" % x))
    print(summarise(df))
    if args.out:
        df.to_csv(args.out, index=False)
    if args.plot:
        plot(df, args.plot)
    return df


if __name__ == "__main__":
    main()
