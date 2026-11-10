"""Plot the solvation free-energy estimates of the main methods against the reference (results/figures)."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "data"


def rows(name):
    with open(DATA / name) as fh:
        return {r.get("model") or r.get("run") or r.get("name"): r for r in csv.DictReader(fh)}


def main():
    est, two, ref = rows("neuralti_estimates.csv"), rows("two_sided_estimates.csv"), rows("references.csv")
    seeds = ("DSM-0.01-s0", "DSM-0.01-s1")
    panel = [
        ("NeuralTI, running average\n(as in the paper)", [(est[k]["rolling"], est[k]["rolling_se"]) for k in seeds]),
        ("NeuralTI, final trained model\n(2 seeds)", [(est[k]["fixed"], est[k]["fixed_se"]) for k in seeds]),
        ("Two-sided NES,\nalchemical path", [(two["base-2.4e5"]["dF"], two["base-2.4e5"]["se"])]),
        ("Two-sided, FEAT-style\nlearned transport", [(two["P1-K400"]["dF"], two["P1-K400"]["se"])]),
    ]
    reference = float(ref["base-BAR"]["value"])
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.axvline(reference, color="0.55", ls="--", lw=1.2)
    n = len(panel)
    for i, (_, pts) in enumerate(panel):
        for v, se in pts:
            ax.errorbar(float(v), n - 1 - i, xerr=float(se), fmt="o", ms=7, color="black", capsize=3, lw=1.3)
    ax.set_yticks(range(n))
    ax.set_yticklabels([lab for lab, _ in panel][::-1])
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlim(-23, -12)
    ax.set_xlabel("Solvation free energy (kBT)")
    ax.text(reference + 0.15, n - 0.55, "reference", color="0.4", fontsize=10.5, va="bottom")
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    out = ROOT / "results" / "figures" / "free_energy_estimates.png"
    fig.savefig(out, dpi=220, facecolor="white")
    print(out)


if __name__ == "__main__":
    main()
