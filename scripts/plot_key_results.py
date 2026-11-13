"""Key-results figure for the MACE-OFF methane runs and the LJ local map (results/figures/key_results.png).
Reads results/data/{methane_switching,methane_hysteresis,lj_local_map_K114}.json only; no model calls."""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "data"


def load(name):
    with open(DATA / name) as fh:
        return json.load(fh)


def main():
    pil = load("methane_switching.json")["tiers"]
    hy = load("methane_hysteresis.json")["rows"]
    v2 = load("lj_local_map_K114.json")["arms"]

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0), dpi=150)
    C0, C1 = "#1f1f1f", "#c0392b"

    # (a) methane in MACE-OFF water: dissipation vs switching time
    a = ax[0]
    T = [pil[k]["switching_time_ps"] for k in "ABC"]
    y = [pil[k]["mean_dissipation"] for k in "ABC"]
    e = [pil[k]["mean_dissipation_boot_se"] for k in "ABC"]
    a.errorbar(T, y, yerr=e, fmt="o-", color=C0, capsize=3, ms=7)
    a.set_xscale("log"); a.set_xticks(T); a.set_xticklabels([str(t) for t in T]); a.minorticks_off()
    a.set_ylim(0, 44); a.set_xlabel("Switching time (ps)"); a.set_ylabel("Wasted work (kBT)")
    a.set_title("Methane in MACE-OFF water:\nno learning, plain switching", fontsize=11)
    for t, v in zip(T, y):
        a.annotate(f"{v:.1f}", (t, v), textcoords="offset points", xytext=(8, 6), fontsize=10)

    # (b) what lags: nearest water distance, forward vs reverse
    b = ax[1]
    lam = [p["lam"] for p in hy]
    f = np.array([[p["removing_mean"], p["removing_se"]] for p in hy])
    r = np.array([[p["inserting_mean"], p["inserting_se"]] for p in hy])
    b.errorbar(lam, f[:, 0], yerr=f[:, 1], fmt="o-", color=C0, ms=4, capsize=2, label="removing methane")
    b.errorbar(lam, r[:, 0], yerr=r[:, 1], fmt="s-", color=C1, ms=4, capsize=2, label="inserting methane")
    b.fill_between(lam, f[:, 0], r[:, 0], color=C1, alpha=0.12)
    b.set_xlabel("λ  (0 = methane fully present, 1 = absent)"); b.set_ylabel("Nearest water to methane (Å)")
    b.set_title("What cannot keep up:\nwater does not make room in time", fontsize=11)
    b.legend(frameon=False, fontsize=9, loc="upper right")

    # (c) LJ: wasted work by method at equal cost
    c = ax[2]
    names = ["Plain\nswitching", "Redistribute\nλ steps", "Learned\nlocal map", "Map +\nredistribute"]
    keys = ["baseline-K114", "diag_densify_only-K114", "learned-BDHD-K114", "map-densify-K114"]
    vals = [v2[k]["mean_dissipation"] for k in keys]
    errs = [v2[k]["mean_dissipation_se"] for k in keys]
    cols = [C0, "#7f8c8d", C1, C1]
    c.bar(range(4), vals, yerr=errs, color=cols, capsize=3, width=0.62)
    for i, v in enumerate(vals):
        c.text(i, v + 0.22, f"{v:.2f}\n({v / vals[0]:.2f}×)" if i else f"{v:.2f}", ha="center", fontsize=9.5)
    c.set_xticks(range(4)); c.set_xticklabels(names, fontsize=9.5); c.set_ylim(0, 5.8)
    c.set_ylabel("Wasted work (kBT)")
    c.set_title("LJ solvation, same number of force calls:\na local learned map saves about 1/3", fontsize=11)

    for x in ax:
        x.spines["top"].set_visible(False); x.spines["right"].set_visible(False)
    fig.tight_layout()
    out = ROOT / "results" / "figures" / "key_results.png"
    fig.savefig(out)
    print(out.relative_to(ROOT), [round(v, 3) for v in vals], [round(v, 3) for v in errs], [round(v, 3) for v in y])


if __name__ == "__main__":
    main()
