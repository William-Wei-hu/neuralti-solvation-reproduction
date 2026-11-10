"""fine-t follow-up of the U_t consistency diagnostic, no MCMC (I(t), E(t): see ut_diag_lib.py).

  python3 fine_grid.py --out DIR --runs RUN [RUN ...] [--chunk 32] [--seed 0]
          [--grid-pairs-cache FILE] [--grid-pairs 512]
          [--mc-pairs-cache FILE] [--mc-pairs 2048] [--mc-pairs-seed 1] [--mc-strata 16]
          [--rw-frames 8000] [--rw-seed 2]

Per run, three parts, each written to DIR/<run>.pkl as soon as it is done (a rerun skips finished parts; a result
file with other settings is refused):
  mc    int_0^1 I dt of the final parameters, sampled as in training: each of the mc-pairs pairs gets one t in every
        stratum [k/K, (k+1)/K) (uniform inside) and its own z (DSM noise exactly as loss_fn_dsm, via
        make_interp_eval).  Unbiased; SE over pairs.  Partial integrals over [0,1/8], [1/8,7/8], [7/8,1] from the
        strata.
  grid  I(t) on GRID: log-spaced from 1e-6 to 1e-1 towards both ends, steps of 1/32 in [1/8, 7/8].  Same pairs and
        z (ut_consistency.keys_for) as the 9-point run, so at t = k/8 it reproduces that run's I values.
  rw    endpoint data frames reweighted to U_t: frames of rho_A for t = s, of rho_B for t = 1 - s, s in RW_S.
        exp(-U_0) = rho_A and exp(-U_1) = rho_B exactly, so with w = exp(-(U_t - U_end)):
          F(s) - F(0)   = -log <w>_A   (= int_0^s E dt, FEP)      E(s) = <w dU/dt>_A / <w>_A
          F(1) - F(1-s) = +log <w>_B   (= int_(1-s)^1 E dt)       ESS = (sum w)^2 / sum w^2
        valid while ESS is large.
Model inputs t are float32; the nodes stored (t32) are the float32 values actually used.
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ut_diag_lib as L
import ut_consistency as UC

import jax
import jax.numpy as jnp
import numpy as np

LOG_S = [10 ** (-6 + k / 4) for k in range(21)]  # 1e-6 ... 1e-1, 4 per decade
EDGE_S = LOG_S + [0.0625, 0.09375]
INNER = [0.125 + j / 32 for j in range(25)]  # 1/8 ... 7/8
GRID = sorted({0.0, 1.0} | set(EDGE_S) | {1 - s for s in EDGE_S} | set(INNER))
RW_S = [LOG_S[k] for k in (0, 4, 8, 12, 14, 16, 18)] + [0.0625, 0.125]  # 1e-6 1e-5 1e-4 1e-3 10^-2.5 1e-2 10^-1.5
MC_KEY = 2**21 + 1  # fold_in index outside the keys_for range [0, 2^20]


def t_of(side, s):
    return s if side == 0 else 1 - s


def settings_of(args):
    return dict(chunk=args.chunk, seed=args.seed, grid_pairs_cache=str(args.grid_pairs_cache),
                grid_pairs=args.grid_pairs, mc_pairs_cache=str(args.mc_pairs_cache), mc_pairs=args.mc_pairs,
                mc_pairs_seed=args.mc_pairs_seed, mc_strata=args.mc_strata, rw_frames=args.rw_frames,
                rw_seed=args.rw_seed, GRID=GRID, RW_S=RW_S)


def eval_interp(interp, params, b0, b1, t32, z, chunk):
    """dU/dt and DSM score loss on interpolant points; t32 [n] float32, z [n, N, 3]."""
    n = len(b0)
    assert n % chunk == 0, (n, chunk)
    ds, scs = [], []
    for i in range(0, n, chunk):
        _, d, _, sc = interp(params, jnp.asarray(b0[i:i + chunk]), jnp.asarray(b1[i:i + chunk]),
                             jnp.asarray(t32[i:i + chunk]).reshape(-1, 1), z[i:i + chunk])
        ds.append(np.asarray(d))
        scs.append(np.asarray(sc))
    return np.concatenate(ds), np.concatenate(scs)


def part_mc(si, info, args, b0, b1):
    interp = L.make_interp_eval(si, info["loss_type"])
    K, n = args.mc_strata, len(b0)
    u = np.random.default_rng([args.seed, MC_KEY]).uniform(size=(K, n))
    t32 = ((np.arange(K)[:, None] + u) / K).astype(np.float32)
    t32 = np.minimum(t32, np.nextafter(np.float32(1), np.float32(0)))
    kz = jax.random.fold_in(jax.random.PRNGKey(args.seed), MC_KEY)
    d, sc = np.zeros((K, n), np.float32), np.zeros((K, n), np.float32)
    for k in range(K):
        z = jax.random.normal(jax.random.fold_in(kz, k), b0.shape)
        d[k], sc[k] = eval_interp(interp, si.params, b0, b1, t32[k], z, args.chunk)
    return dict(t32=t32.astype(np.float64), dUdt=d, score=sc)


def part_grid(si, info, args, b0, b1):
    interp = L.make_interp_eval(si, info["loss_type"])
    d, sc = [], []
    for t in GRID:
        kz, _ = UC.keys_for(args.seed, t)
        z = jax.random.normal(kz, b0.shape)
        di, si_ = eval_interp(interp, si.params, b0, b1, np.full(len(b0), t, np.float32), z, args.chunk)
        d.append(di)
        sc.append(si_)
    return dict(t=np.asarray(GRID), t32=np.asarray(GRID, np.float32).astype(np.float64), dUdt=np.stack(d),
                score=np.stack(sc))


def part_rw(si, ts, args):
    vg = jax.jit(jax.vmap(jax.value_and_grad(lambda p, x, t: si.U(p, x, t), argnums=2), in_axes=(None, 0, None)))
    U_ref = {0: jax.jit(jax.vmap(ts.U0)), 1: jax.jit(jax.vmap(ts.U1))}

    def on_frames(X, t):
        Us, ds = [], []
        t1 = jnp.array([t], jnp.float32)
        for i in range(0, len(X), args.chunk):
            U, d = vg(si.params, jnp.asarray(X[i:i + args.chunk]), t1)
            Us.append(np.asarray(U))
            ds.append(np.asarray(d).reshape(-1))
        return np.concatenate(Us), np.concatenate(ds)

    out = {}
    for side in (0, 1):
        X = L.endpoint_frames(float(side), args.rw_frames, args.rw_seed)
        U_end, d_end = on_frames(X, float(side))
        Ur = np.concatenate([np.asarray(U_ref[side](jnp.asarray(X[i:i + args.chunk])))
                             for i in range(0, len(X), args.chunk)])
        r = dict(U_end=U_end, dUdt_end=d_end, U_target=Ur, max_abs_U_end_minus_target=float(np.abs(U_end - Ur).max()),
                 s=np.asarray(RW_S), t32=[], dU=[], dUdt=[])
        for s in RW_S:
            t = t_of(side, s)
            U, d = on_frames(X, t)
            r["t32"].append(float(np.float32(t)))
            r["dU"].append(U - U_end)
            r["dUdt"].append(d)
        r["t32"], r["dU"], r["dUdt"] = np.asarray(r["t32"]), np.stack(r["dU"]), np.stack(r["dUdt"])
        out[side] = r
    return out


def save(res, path):
    with open(path.with_suffix(".tmp"), "wb") as fh:
        pickle.dump(res, fh)
    path.with_suffix(".tmp").replace(path)


def run_one(run, args, pairs_grid, pairs_mc, out_dir):
    import fine_analysis as FA

    path = out_dir / f"{run}.pkl"
    settings = settings_of(args)
    si, ts, info = L.load_model(run)
    bad = L.check_info(info)
    print(f"\n#### {run}: {info}", flush=True)
    if bad:
        raise SystemExit(f"checkpoint of {run} does not match expectations: {bad}")
    if path.exists():
        with open(path, "rb") as fh:
            res = pickle.load(fh)
        if res["settings"] != settings:
            raise SystemExit(f"{path} exists with different settings {res['settings']} (not overwriting)")
    else:
        res = dict(info=info, settings=settings)
    for part in ("mc", "grid", "rw"):
        if part in res:
            print(f"{run} {part}: already done, skipping", flush=True)
            continue
        t0 = time.time()
        if part == "mc":
            res[part] = part_mc(si, info, args, *pairs_mc)
        elif part == "grid":
            res[part] = part_grid(si, info, args, *pairs_grid)
        else:
            res[part] = part_rw(si, ts, args)
        res.setdefault("seconds", {})[part] = time.time() - t0
        bad_vals = FA.nonfinite(res[part])
        save(res, path)
        print(f"{run} {part}: done in {time.time() - t0:.0f}s, non-finite values {bad_vals} -> {path}", flush=True)
        FA.print_part(res, part)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grid-pairs-cache", type=Path, default=L.ROOT / "outputs/ut-consistency/full/pairs_seed0_n512.npz")
    ap.add_argument("--grid-pairs", type=int, default=512)
    ap.add_argument("--mc-pairs-cache", type=Path, default=None)
    ap.add_argument("--mc-pairs", type=int, default=2048)
    ap.add_argument("--mc-pairs-seed", type=int, default=1)
    ap.add_argument("--mc-strata", type=int, default=16)
    ap.add_argument("--rw-frames", type=int, default=8000)
    ap.add_argument("--rw-seed", type=int, default=2)
    args = ap.parse_args(argv)
    if args.mc_pairs_cache is None:
        args.mc_pairs_cache = args.out / f"pairs_seed{args.mc_pairs_seed}_n{args.mc_pairs}.npz"
    assert args.rw_frames % args.chunk == 0 and args.mc_strata % 8 == 0, (args.rw_frames, args.chunk, args.mc_strata)
    assert args.mc_pairs_seed != args.seed or args.mc_pairs != args.grid_pairs, "mc pairs must differ from grid pairs"
    L.set_jax_flags()
    print("devices:", jax.devices(), "jax", jax.__version__, flush=True)
    print("args:", vars(args), flush=True)
    print(f"GRID: {len(GRID)} t, RW_S: {len(RW_S)} s per end", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    pairs_grid = L.aligned_pairs(args.grid_pairs, args.seed, args.chunk, args.grid_pairs_cache)
    pairs_mc = L.aligned_pairs(args.mc_pairs, args.mc_pairs_seed, args.chunk, args.mc_pairs_cache)
    print(f"pairs ready: grid {len(pairs_grid[0])}, mc {len(pairs_mc[0])} ({time.time() - t0:.0f}s)", flush=True)
    for run in args.runs:
        run_one(run, args, pairs_grid, pairs_mc, args.out)


if __name__ == "__main__":
    main()
