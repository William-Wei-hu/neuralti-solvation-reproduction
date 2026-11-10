"""summaries of fine_grid.py results (numpy only, no model evaluation).

  python3 fine_analysis.py --dir DIR [--full-merged DIR] [--json FILE]

For each run in DIR:
  - int_0^1 I dt of the final parameters (mc part, stratified uniform t as in training) and its split over
    [0,1/8], [1/8,7/8], [7/8,1]; the same from per-pair trapezoid on the fine grid
  - near each end: int I dt over [0,s] ([1-s,1]) from the grid, the exact free-energy change F(s)-F(0)
    (F(1)-F(1-s)) of the learned path by reweighting endpoint frames, their difference B (= neural-TI bias
    accumulated in that window, since int E dt = F difference for any parameters), ESS, and the reweighted E(s)
  - total bias of the final parameters: int I dt (mc) - reference
  - the grid I values at t = k/8 against the 9-point run (--full-merged), which used the same pairs and z
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np

REF = (-20.365, 0.013)  # NES, BAR along the base path (no network); standard TI 65 slices: -20.073
ROLLING = {  # 20k-50k rolling training estimate (dUdt_hist), block SE
    "egnn-seed0": (-14.415, 0.368),
    "egnn-dsm-smid0.005-seed0": (-17.620, 0.133),
    "egnn-dsm-smid0.01-seed0": (-20.643, 0.209),
    "egnn-dsm-smid0.01-seed1": (-20.546, 0.176),
    "egnn-dsm-smid0.02-seed0": (-41.730, 1.849),
}


def ms(x):
    x = np.asarray(x, np.float64)
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))


def nonfinite(part):
    if isinstance(part, dict) and 0 in part:  # rw
        return sum(nonfinite(v) for v in part.values())
    return int(sum((~np.isfinite(np.asarray(v, np.float64))).sum() for k, v in part.items()
                   if k in ("dUdt", "dU", "score", "U_end", "dUdt_end")))


# ----------------------------------------------------------------------------- mc
def mc_summary(r):
    d = np.asarray(r["dUdt"], np.float64)  # [K, n]
    K = len(d)
    k = np.arange(K)
    regions = dict(left=k < K // 8, mid=(k >= K // 8) & (k < 7 * K // 8), right=k >= 7 * K // 8)
    out = dict(total=ms(d.mean(0)), n_pairs=d.shape[1], strata=K)
    for name, sel in regions.items():
        out[name] = ms(d[sel].sum(0) / K)
    return out


# ----------------------------------------------------------------------------- grid
def per_pair_trapezoid(t, d, lo, hi):
    """int_lo^hi of per-pair integrands d[nt, n] over the grid nodes t (lo, hi must be nodes)."""
    sel = (t >= lo) & (t <= hi)
    tt, dd = t[sel], np.asarray(d, np.float64)[sel]
    assert len(tt) >= 2 and tt[0] == lo and tt[-1] == hi, (lo, hi, tt[:2], tt[-2:])
    return ((dd[1:] + dd[:-1]) * np.diff(tt)[:, None]).sum(0) / 2


def grid_summary(r):
    t = np.asarray(r["t32"])
    d = r["dUdt"]
    lo, hi = float(np.float32(0.125)), float(np.float32(0.875))
    return dict(total=ms(per_pair_trapezoid(t, d, 0.0, 1.0)), left=ms(per_pair_trapezoid(t, d, 0.0, lo)),
                mid=ms(per_pair_trapezoid(t, d, lo, hi)), right=ms(per_pair_trapezoid(t, d, hi, 1.0)),
                n_pairs=np.asarray(d).shape[1], n_t=len(t))


# ----------------------------------------------------------------------------- rw
def fep(dU, sign):
    """sign=-1: F(t)-F(end) = -log<exp(-dU)>; sign=+1: F(end)-F(t) = +log<exp(-dU)>.  Returns value, SE (delta
    method, frames treated as independent), ESS, normalised weights."""
    lw = -np.asarray(dU, np.float64)
    m = lw.max()
    w = np.exp(lw - m)
    mw = w.mean()
    val = sign * (m + np.log(mw))  # m + log(mw) = log<exp(-dU)>
    se = float(np.sqrt(w.var(ddof=1) / len(w)) / mw)
    ess = float(w.sum() ** 2 / (w**2).sum())
    return float(val), se, ess, w / w.sum()


def rw_summary(r, side):
    rows = []
    for j, s in enumerate(r["s"]):
        dF, se, ess, wn = fep(r["dU"][j], -1 if side == 0 else +1)
        rows.append(dict(s=float(s), t32=float(r["t32"][j]), dF=dF, dF_se=se, ess=ess,
                         E_rw=float((wn * np.asarray(r["dUdt"][j], np.float64)).sum()),
                         E_unweighted=float(np.mean(r["dUdt"][j]))))
    return dict(rows=rows, E_end=ms(r["dUdt_end"]), n_frames=len(r["U_end"]),
                max_abs_U_end_minus_target=r["max_abs_U_end_minus_target"])


def window_bias(res):
    """For each s: grid int I over [0,s] / [1-s,1] minus the FEP free-energy change over the same window."""
    t = np.asarray(res["grid"]["t32"])
    d = res["grid"]["dUdt"]
    out = {}
    for side in (0, 1):
        rs = rw_summary(res["rw"][side], side)
        rows = []
        for row in rs["rows"]:
            ts = row["t32"]
            Ii = per_pair_trapezoid(t, d, 0.0, ts) if side == 0 else per_pair_trapezoid(t, d, ts, 1.0)
            I = ms(Ii)
            rows.append(dict(row, I_int=I, B=(I[0] - row["dF"], float(np.hypot(I[1], row["dF_se"])))))
        out[side] = dict(rs, rows=rows)
    return out


# ----------------------------------------------------------------------------- printing
def f(x, w=8, p=2):
    return f"{x[0]:{w}.{p}f} +- {x[1]:5.{p}f}"


def print_part(res, part):
    run = res["info"]["run"]
    if part == "mc":
        m = mc_summary(res["mc"])
        print(f"  [{run} mc] int_0^1 I dt = {f(m['total'])}   ({m['n_pairs']} pairs x {m['strata']} strata)"
              f"   [0,1/8] {f(m['left'])}   [1/8,7/8] {f(m['mid'])}   [7/8,1] {f(m['right'])}", flush=True)
    elif part == "grid":
        g = grid_summary(res["grid"])
        r = res["grid"]
        print(f"  [{run} grid] trapezoid on {g['n_t']} t x {g['n_pairs']} pairs: int I dt = {f(g['total'])}"
              f"   [0,1/8] {f(g['left'])}   [1/8,7/8] {f(g['mid'])}   [7/8,1] {f(g['right'])}", flush=True)
        print("        t        I(t)            score", flush=True)
        for i, t in enumerate(r["t32"]):
            print(f"  {t:12.7f} {f(ms(r['dUdt'][i]), 9, 2)}   {np.mean(r['score'][i]):8.4f}", flush=True)
    elif part == "rw":
        for side in (0, 1):
            rs = rw_summary(res["rw"][side], side)
            lab = "F(t)-F(0)" if side == 0 else "F(1)-F(t)"
            print(f"  [{run} rw side {side}] {rs['n_frames']} frames, <dU/dt> at the end {f(rs['E_end'])}, "
                  f"max|U_end - U_target| = {rs['max_abs_U_end_minus_target']:.3g}", flush=True)
            print(f"        s            t     {lab:>18s}      ESS      E_rw(t)   <dU/dt> unweighted", flush=True)
            for row in rs["rows"]:
                print(f"  {row['s']:10.3g} {row['t32']:12.7f} {f((row['dF'], row['dF_se']), 9, 3)} {row['ess']:8.1f}"
                      f" {row['E_rw']:12.3f} {row['E_unweighted']:12.3f}", flush=True)


def compare_full(res, full_dir):
    """max |grid I - 9-point-run I| per pair at the shared nodes (same pairs and z: should be ~0)."""
    p = Path(full_dir) / f"{res['info']['run']}.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as fh:
        full = pickle.load(fh)
    t = list(res["grid"]["t"])
    diffs = {}
    for tk, v in full["t"].items():
        if tk not in t:
            continue
        ref = v["I_vals_pairs"] if v.get("I_source") == "endpoint frames" else v["I_vals"]
        g = np.asarray(res["grid"]["dUdt"][t.index(tk)], np.float64)
        if len(ref) != len(g):
            return None
        diffs[tk] = float(np.abs(g - np.asarray(ref, np.float64)).max())
    return diffs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--full-merged", type=Path, default=None)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--quiet-parts", action="store_true", help="skip the per-part tables")
    args = ap.parse_args(argv)
    summary = {}
    for p in sorted(args.dir.glob("*.pkl")):
        with open(p, "rb") as fh:
            res = pickle.load(fh)
        run = res["info"]["run"]
        print(f"\n######## {run}  ({res['info']['loss_type']}, sigma_mid={res['info']['sigma_mid']})", flush=True)
        if not args.quiet_parts:
            for part in ("mc", "grid", "rw"):
                if part in res:
                    print_part(res, part)
        s = summary[run] = {}
        if "mc" in res:
            s["mc"] = mc_summary(res["mc"])
            s["bias_total"] = (s["mc"]["total"][0] - REF[0], float(np.hypot(s["mc"]["total"][1], REF[1])))
        if "grid" in res:
            s["grid"] = grid_summary(res["grid"])
            if args.full_merged is not None:
                s["grid_vs_9pt_max_abs_diff"] = compare_full(res, args.full_merged)
        if "grid" in res and "rw" in res:
            wb = window_bias(res)
            s["window"] = {str(k): v for k, v in wb.items()}
            for side in (0, 1):
                lab = "[0,s]" if side == 0 else "[1-s,1]"
                print(f"  window {lab}:        s    ESS      int I dt (grid)      F change (FEP)        B = I - F", flush=True)
                for row in wb[side]["rows"]:
                    print(f"  {'':16s} {row['s']:8.3g} {row['ess']:7.1f}  {f(row['I_int'], 9, 3)}  "
                          f"{f((row['dF'], row['dF_se']), 9, 3)}  {f(row['B'], 9, 3)}", flush=True)
        if "mc" in s:
            print(f"  total: int I dt (mc) = {f(s['mc']['total'])}, bias vs {REF[0]} = {f(s['bias_total'])}; "
                  f"rolling (training) {ROLLING.get(run, (float('nan'), 0))[0]:.2f}", flush=True)
        if s.get("grid_vs_9pt_max_abs_diff") is not None:
            print(f"  grid vs 9-point run, max |dI| per pair at shared t: "
                  f"{max(s['grid_vs_9pt_max_abs_diff'].values()):.3g} over {len(s['grid_vs_9pt_max_abs_diff'])} t",
                  flush=True)
    if args.json is not None:
        with open(args.json, "w") as fh:
            json.dump(summary, fh, indent=1, default=float)
        print(f"\nwrote {args.json}", flush=True)
    return summary


if __name__ == "__main__":
    main()
