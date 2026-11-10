"""Merge the per-worker results of the full U_t diagnostic (diag/ut-consistency/full/w*/<run>.pkl) into
full/merged/<run>.pkl and print one table per model plus a cross-model summary.  File operations only."""
import pickle
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ut_diag_lib as L
import ut_consistency as UC

import numpy as np

FULL = L.ROOT / "outputs/ut-consistency/full"
GRID = [i / 8 for i in range(9)]
ROLLING = {  # 20k-50k rolling training estimate (dUdt_hist), block SE
    "egnn-seed0": (-14.415, 0.368),
    "egnn-dsm-smid0.005-seed0": (-17.620, 0.133),
    "egnn-dsm-smid0.01-seed0": (-20.643, 0.209),
    "egnn-dsm-smid0.01-seed1": (-20.546, 0.176),
    "egnn-dsm-smid0.02-seed0": (-41.730, 1.849),
}
TI_REF = -20.073  # standard TI, 65 slices, trapezoid

out = FULL / "merged"
out.mkdir(parents=True, exist_ok=True)
summary = []
for run in ROLLING:
    parts = sorted(FULL.glob(f"w*/{run}.pkl"))
    merged = None
    for p in parts:
        with open(p, "rb") as fh:
            r = pickle.load(fh)
        if merged is None:
            merged = dict(info=r["info"], settings=r["settings"], t={}, sources={})
        assert r["settings"] == merged["settings"] and r["info"] == merged["info"], p
        for t, v in r["t"].items():
            assert t not in merged["t"], (run, t, p)
            merged["t"][t] = v
            merged["sources"][t] = str(p)
    if merged is None:
        print(f"\n== {run}: no results yet")
        continue
    with open(out / f"{run}.pkl", "wb") as fh:
        pickle.dump(merged, fh)
    UC.print_table(merged)
    done = sorted(merged["t"])
    missing = [t for t in GRID if t not in merged["t"]]
    if not missing:
        I = L.trapezoid(GRID, [merged["t"][t]["I_mean"] for t in GRID])
        E = L.trapezoid(GRID, [merged["t"][t]["E"][0] for t in GRID])
        w = np.full(len(GRID), 1 / 8)
        w[[0, -1]] = 1 / 16
        seI = float(np.sqrt(sum((wi * merged["t"][t]["I_se"]) ** 2 for wi, t in zip(w, GRID))))
        seE = float(np.sqrt(sum((wi * merged["t"][t]["E"][1]) ** 2 for wi, t in zip(w, GRID))))
        summary.append((run, I, seI, E, seE))
    else:
        print(f"   missing t: {missing}")

if summary:
    print(f"\n== summary (trapezoid over t = 0, 1/8, ..., 1; SE from per-t SEs, no grid error; standard TI {TI_REF})")
    print(f"{'run':28s} {'rolling (train)':>18s} {'int I dt (final)':>18s} {'int E dt (learned path)':>24s} {'I - E':>8s}")
    for run, I, seI, E, seE in summary:
        ro = ROLLING[run]
        print(f"{run:28s} {ro[0]:9.2f} +- {ro[1]:4.2f} {I:9.2f} +- {seI:4.2f} {E:15.2f} +- {seE:4.2f} {I - E:8.2f}")
