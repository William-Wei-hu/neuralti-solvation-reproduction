"""U_t self-consistency diagnostic for trained neuralTI-solvation models (see ut_diag_lib.py).

  python3 ut_consistency.py --out DIR --runs RUN [RUN ...] [--ts 0,0.125,...,1] [--n-pairs 512] [--chunk 32]
          [--walkers 32] [--steps 100000] [--record-every 1000] [--seed 0] [--pairs-cache FILE]
          [--control-linear T ...]

For each run and t: I(t) on n_pairs interpolant points, then Metropolis MCMC under the learned U_t from the first
`walkers` of those points.  Results go to DIR/<run>.pkl after every finished t; a rerun with the same settings skips
finished t.  Interpolant points (pairs, z) and MCMC keys depend only on (seed, t), so every model sees the same
points.  --control-linear T: the same MCMC under the exact linear-path potential of experiments/standardTI.py,
started from its equilibrium slice at t=T, which must not drift (pipeline check).
At t=0 and t=1, exp(-U_t) is exactly rho_A / rho_B, so E(t) is taken from --n-endpoint-frames endpoint data frames
(no MCMC there).
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ut_diag_lib as L

import jax
import jax.numpy as jnp
import numpy as np


def keys_for(seed, t):
    k = jax.random.fold_in(jax.random.PRNGKey(seed), int(round(t * 2**20)))
    kz, kmc = jax.random.split(k)
    return kz, kmc


def summarise_chain(chain):
    n_rec = len(chain["step"]) - 1
    assert n_rec >= 4, "need at least 4 records (steps / record_every)"
    out = dict(
        E0=L.walker_stats(chain["dUdt"][0:1]),
        E_half=L.walker_stats(chain["dUdt"][1 + n_rec // 2:]),
        E_quarter=L.walker_stats(chain["dUdt"][1 + (3 * n_rec) // 4:]),
        U0=L.walker_stats(chain["U"][0:1]),
        U_half=L.walker_stats(chain["U"][1 + n_rec // 2:]),
        acceptance=float(chain["acceptance"].mean()),
        n_nonfinite_proposals=int(chain["n_nonfinite_proposals"]),
    )
    if "n_edges" in chain:
        out.update(
            max_edges=int(chain["n_edges"].max()),
            coord0=float(chain["coord"][0].mean()),
            coord_half=float(chain["coord"][1 + n_rec // 2:].mean()),
            rmin_ss_min=float(chain["rmin_ss"].min()),
            rmin_ab_min=float(chain["rmin_ab"].min()),
        )
    return out


def make_logger(tag, every_frac=0.1):
    t_start = time.time()

    def log(c, n, rec, acc):
        if c == n or c % max(1, int(n * every_frac)) == 0:
            print(f"  [{tag}] {c}/{n} records  <dU/dt>={np.mean(rec['dUdt'][-1]):9.3f}  <U>={np.mean(rec['U'][-1]):10.2f}"
                  f"  acc={np.mean(acc):.3f}  {time.time() - t_start:7.0f}s", flush=True)

    return log


def fmt(ms):
    return f"{ms[0]:9.3f} +- {ms[1]:6.3f}"


def print_table(res):
    done = sorted(res["t"])
    print(f"\n== {res['info']['run']}  ({res['info']['loss_type']}, sigma_mid={res['info']['sigma_mid']})", flush=True)
    print("     t   I(t) interpolant      E0 (step 0)         E(t) 2nd half      E(t) last 1/4   I-E(half)"
          "   acc   max_edges  coord0->half  rmin_ss  nonfinite", flush=True)
    for t in done:
        r = res["t"][t]
        if r["E_source"] == "data":
            print(f"{t:6.3f} {fmt((r['I_mean'], r['I_se']))}   (endpoint data)  {fmt(r['E'])}"
                  f"   (iid SE, {len(r['E_vals'])} frames) {r['I_mean'] - r['E'][0]:9.3f}"
                  f"   [I on {len(r['I_vals_pairs'])} pairs: {fmt(r['I_pairs'])}]", flush=True)
            continue
        m = r["mcmc"]
        print(f"{t:6.3f} {fmt((r['I_mean'], r['I_se']))} {fmt(m['E0'])} {fmt(m['E_half'])} {fmt(m['E_quarter'])}"
              f" {r['I_mean'] - m['E_half'][0]:9.3f}  {m['acceptance']:.3f}  {m['max_edges']:6d}"
              f"  {m['coord0']:5.1f}->{m['coord_half']:5.1f}  {m['rmin_ss_min']:.3f}  {m['n_nonfinite_proposals']}",
              flush=True)
    if len(done) >= 2 and done[0] == 0.0 and done[-1] == 1.0:
        I = L.trapezoid(done, [res["t"][t]["I_mean"] for t in done])
        E = L.trapezoid(done, [res["t"][t]["E"][0] for t in done])
        print(f"trapezoid over {len(done)} t:  int I dt (neural TI, final params) = {I:.3f}   "
              f"int E dt (TI along learned path) = {E:.3f}   difference = {I - E:.3f}", flush=True)


def diagnose_run(run, args, b0, b1, out_dir):
    path = out_dir / f"{run}.pkl"
    settings = dict(n_pairs=args.n_pairs, walkers=args.walkers, steps=args.steps, record_every=args.record_every,
                    seed=args.seed, pairs_cache=str(args.pairs_cache), n_endpoint_frames=args.n_endpoint_frames)
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
        res = dict(info=info, settings=settings, t={})

    interp = L.make_interp_eval(si, info["loss_type"])
    fns = L.make_chain_fns(lambda params, x, t: si.U(params, x, t))
    structure = L.make_structure(ts, info["cutoff"])
    step_std = float(np.sqrt(2 * ts.sampling_dt))
    for t in args.ts:
        if t in res["t"]:
            print(f"t={t}: already done, skipping", flush=True)
            continue
        t0 = time.time()
        kz, kmc = keys_for(args.seed, t)
        z = jax.random.normal(kz, b0.shape)
        tt = jnp.full((args.chunk, 1), t, jnp.float32)
        xs, ds, Us, scs = [], [], [], []
        for i in range(0, args.n_pairs, args.chunk):
            xt, d, U, sc = interp(si.params, jnp.asarray(b0[i:i + args.chunk]), jnp.asarray(b1[i:i + args.chunk]),
                                  tt, z[i:i + args.chunk])
            if i < args.walkers:
                xs.append(np.asarray(xt))
            ds.append(np.asarray(d))
            Us.append(np.asarray(U))
            scs.append(np.asarray(sc))
        d, U, sc = np.concatenate(ds), np.concatenate(Us), np.concatenate(scs)
        x_init = np.concatenate(xs)[: args.walkers]
        t_interp = time.time() - t0
        print(f"t={t}: I = {d.mean():.3f} +- {d.std(ddof=1) / np.sqrt(len(d)):.3f} on {len(d)} interpolant points"
              f" (finite {np.isfinite(d).all()}), score loss {sc.mean():.4g}, {t_interp:.0f}s", flush=True)
        r = dict(I_vals=d, I_mean=float(d.mean()), I_se=float(d.std(ddof=1) / np.sqrt(len(d))), U_interp=U, score=sc)
        t1 = jnp.array([t], jnp.float32)
        if t in (0.0, 1.0) and args.n_endpoint_frames > 0:
            # exp(-U_0) = rho_A and exp(-U_1) = rho_B exactly: E(t) from the endpoint MCMC data, no new chain.
            # At t=0 (t=1) the interpolant point is x0 (x1) itself (+ sigma_min z for DSM), up to a translation and
            # a solvent permutation that U_t is invariant to, so I(t) is also evaluated on these frames (same
            # interp code with x1 = x0, i.e. dx = 0): far more points than the n_pairs pairs, whose I is kept too.
            X = L.endpoint_frames(t, args.n_endpoint_frames, args.seed)
            ze = jax.random.normal(jax.random.fold_in(kz, 1), X.shape)
            de, e = [], []
            for i in range(0, len(X), args.chunk):
                Xc = jnp.asarray(X[i:i + args.chunk])
                de.append(np.asarray(interp(si.params, Xc, Xc, tt, ze[i:i + args.chunk])[1]))
                e.append(np.asarray(fns["dudt"](si.params, Xc, t1)))
            de, e = np.concatenate(de), np.concatenate(e)
            r.update(I_source="endpoint frames", I_vals_pairs=r["I_vals"], I_pairs=(r["I_mean"], r["I_se"]),
                     I_vals=de, I_mean=float(de.mean()), I_se=float(de.std(ddof=1) / np.sqrt(len(de))),
                     E_source="data", E_vals=e, E=(float(e.mean()), float(e.std(ddof=1) / np.sqrt(len(e)))))
            print(f"t={t}: on {len(e)} endpoint frames (iid SE): I = {r['I_mean']:.3f} +- {r['I_se']:.3f}, "
                  f"E = {r['E'][0]:.3f} +- {r['E'][1]:.3f} (finite {np.isfinite(de).all() and np.isfinite(e).all()})",
                  flush=True)
        else:
            chain, _, _ = L.run_chain(si.params, fns, jnp.asarray(x_init), t1, kmc, step_std,
                                      args.steps, args.record_every, structure, make_logger(f"{run} t={t}"))
            m = summarise_chain(chain)
            r.update(E_source="mcmc", chain=chain, mcmc=m, E=m["E_half"])
        r["seconds"] = dict(interp=t_interp, total=time.time() - t0)
        res["t"][t] = r
        with open(path.with_suffix(".tmp"), "wb") as fh:
            pickle.dump(res, fh)
        path.with_suffix(".tmp").replace(path)
        print(f"t={t}: done in {time.time() - t0:.0f}s -> {path}", flush=True)
    print_table(res)


def control_linear(T, args, out_dir):
    """MCMC under the exact linear-path U_t of standardTI.py from its equilibrium slice at t=T."""
    path = out_dir / f"control_linear_t{T:.3f}.pkl"
    if path.exists():
        print(f"{path} exists, skipping control", flush=True)
        return
    from data.target_systems import LJ3D_with_solute
    from omegaconf import OmegaConf

    cutoff = float(OmegaConf.load(L.CODE / "experiments/config.yaml").U_model.cutoff)
    ts = LJ3D_with_solute()
    with open(L.RUNS / f"standardTI-seed0/data/LJ3Dwithsolute_{T:.3f}", "rb") as fh:
        X = np.asarray(pickle.load(fh))
    U_lin = lambda params, x, t: ts.Ut(t.reshape(), x)
    fns = L.make_chain_fns(U_lin)
    t1 = jnp.array([T], jnp.float32)
    ref_d = np.concatenate([np.asarray(fns["dudt"](None, jnp.asarray(c), t1)) for c in np.split(X, len(X) // 500)])
    ref_U = np.concatenate([np.asarray(fns["Uv"](None, jnp.asarray(c), t1)) for c in np.split(X, len(X) // 500)])
    print(f"control t={T}: slice reference over {len(X)} frames <dU/dt>={ref_d.mean():.3f} <U>={ref_U.mean():.2f}",
          flush=True)
    kz, kmc = keys_for(args.seed + 1000, T)
    chain, _, _ = L.run_chain(None, fns, jnp.asarray(X[: args.walkers]), t1, kmc, float(np.sqrt(2 * ts.sampling_dt)),
                              args.steps, args.record_every, L.make_structure(ts, cutoff),
                              make_logger(f"control-linear t={T}"))
    summ = summarise_chain(chain)
    with open(path, "wb") as fh:
        pickle.dump(dict(T=T, ref_dUdt=ref_d, ref_U=ref_U, chain=chain, mcmc=summ, settings=vars(args).copy()), fh)
    print(f"control t={T}: ref {ref_d.mean():.3f} +- {ref_d.std(ddof=1) / np.sqrt(len(ref_d)):.3f} (iid SE)  "
          f"E0 {fmt(summ['E0'])}  E_half {fmt(summ['E_half'])}  acc {summ['acceptance']:.3f}  "
          f"U ref {ref_U.mean():.2f} U_half {summ['U_half'][0]:.2f} +- {summ['U_half'][1]:.2f}", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--runs", nargs="*", default=[])
    p.add_argument("--ts", type=lambda s: [float(v) for v in s.split(",")], default=[i / 8 for i in range(9)])
    p.add_argument("--n-pairs", type=int, default=512)
    p.add_argument("--chunk", type=int, default=32)
    p.add_argument("--walkers", type=int, default=32)
    p.add_argument("--steps", type=int, default=100000)
    p.add_argument("--record-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pairs-cache", type=Path, default=None)
    p.add_argument("--control-linear", type=float, nargs="*", default=[])
    p.add_argument("--n-endpoint-frames", type=int, default=32000,
                   help="t=0 and t=1: I(t) and E(t) on this many endpoint data frames, no MCMC (0: pairs + MCMC)")
    args = p.parse_args(argv)
    assert args.walkers <= args.n_pairs and args.n_pairs % args.chunk == 0, "need walkers <= n_pairs, chunk | n_pairs"
    assert args.n_endpoint_frames % args.chunk == 0, "chunk must divide n_endpoint_frames"
    if args.pairs_cache is None:
        args.pairs_cache = args.out / f"pairs_seed{args.seed}_n{args.n_pairs}.npz"
    L.set_jax_flags()
    args.out.mkdir(parents=True, exist_ok=True)
    print("devices:", jax.devices(), "jax", jax.__version__, flush=True)
    print("args:", vars(args), flush=True)
    for T in args.control_linear:
        control_linear(T, args, args.out)
    if args.runs:
        t0 = time.time()
        b0, b1 = L.aligned_pairs(args.n_pairs, args.seed, args.chunk, args.pairs_cache)
        print(f"{len(b0)} aligned pairs ready ({time.time() - t0:.0f}s)", flush=True)
        for run in args.runs:
            diagnose_run(run, args, b0, b1, args.out)


if __name__ == "__main__":
    main()
