"""Summary table of diag/replay/<run>.json (nti_replay_dsm.py) next to the NES TSM replay (read only).  Numpy only.

Per run: logged vs replayed over the last 1000 training steps (same batches, t, z; final parameters), the same
split by k = number of Adam updates between the logged step and the final parameters, R0 step-0 check, and R2
(stratified t on the training pairs, final parameters) next to the fine-grid MC estimate on other pairs.
"""
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "outputs"
RUNS = ["egnn-seed0", "egnn-dsm-smid0.01-seed0", "egnn-dsm-smid0.01-seed1", "egnn-dsm-smid0.02-seed0", "egnn-dsm-smid0.005-seed0"]
FINE = OUT / "ut-consistency/fine/summary.json"
BINS = [(1, 20), (21, 100), (101, 300), (301, 1000)]


def f(m, s, w=7, p=2):
    return f"{m:{w}.{p}f} +- {s:{p + 3}.{p}f}" if s is not None else f"{m:{w}.{p}f}"


def main():
    fine = json.loads(FINE.read_text()) if FINE.exists() else {}
    rows = []
    for run in RUNS:
        p = OUT / "replay" / f"{run}.json"
        if not p.exists():
            print(f"{run}: no result")
            continue
        r = json.loads(p.read_text())
        rows.append((run, r))
    print("\n== last 1000 training steps: same batches, t and z; logged (parameters of that step) vs replayed "
          "(final parameters)")
    print(f"{'run':36s} {'logged':>16s} {'replayed':>16s} {'replayed-logged':>17s} {'rms':>7s} {'corr':>6s}  "
          f"{'R0 step0 rel':>12s}  {'R2 (train pairs)':>17s}  {'fine-grid MC':>16s}")
    for run, r in rows:
        s = r.get("R1", {}).get("stats")
        if s is None:
            print(f"{run:36s} R1 missing: {r.get('R1', {}).get('error')}")
            continue
        r0 = r.get("R0", {})
        rel = r0.get("step0", {}).get("rel_diff")
        if rel is None and "step0" in r0:
            st0 = r0["step0"]
            rel = abs(st0["replayed"] - st0["logged"]) / max(abs(st0["logged"]), 1e-3)
        r2 = r.get("R2", {})
        mc = fine.get(run.split(" ")[0], {}).get("mc", {}).get("total")
        print(f"{run:36s} {f(s['logged_mean'], s['logged_se'])} {f(s['replayed_mean'], s['replayed_se'])} "
              f"{f(s['diff_mean'], s['diff_se'])} {s['diff_rms']:7.1f} {s['corr']:6.3f}  "
              f"{rel if rel is not None else float('nan'):12.2e}  "
              f"{f(r2.get('dF_estimate', np.nan), r2.get('se'))}  {f(*mc) if mc else '':>16s}")
    print("\n== the same split by k = Adam updates between the logged step and the final parameters "
          "(diff = replayed - logged; logged sd = step-to-step spread of the logged values in that bin)")
    for run, r in rows:
        r1 = r.get("R1", {})
        if "k_binned" in r1:
            parts = [f"k {b['k_lo']}-{b['k_hi']}: diff {f(b['diff_mean'], b['diff_se'], 6, 1)} rms {b['diff_rms']:5.1f}"
                     f" corr {b['corr']:.2f} (logged sd {b['logged_sd']:.0f})" for b in r1["k_binned"]]
        elif "per_row" in r1:  # NES json: per-row rms and corr only; row i of the last 50 has k = 20*(49-i)+1 ...
            pr = r1["per_row"]
            n = len(pr)
            parts = []
            for lo, hi in BINS:
                sel = [x for i, x in enumerate(pr) if lo <= 20 * (n - 1 - i) + 1 <= hi]
                if sel:
                    rms = float(np.sqrt(np.mean([x["diff_rms"] ** 2 for x in sel])))
                    parts.append(f"k {lo}-{hi}: rms {rms:5.1f} mean row corr {np.mean([x['corr'] for x in sel]):.2f}")
            lr = r1.get("last_row", {}).get("stats")
            if lr:
                parts.insert(0, f"last row diff {f(lr['diff_mean'], lr['diff_se'], 6, 1)} corr {lr['corr']:.2f}")
        else:
            parts = ["no R1"]
        print(f"{run:36s} " + " | ".join(parts))


if __name__ == "__main__":
    main()
