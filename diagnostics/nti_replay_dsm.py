"""replay the NeuralTI training data stream of one run exactly (DSM or TSM), final parameters.

Generalises an earlier TSM-only replay check.  Features: any run under runs/; the run's own
src/ and data/ code is imported (no NES modules); R2 uses the DSM noise of loss_fn_dsm for DSM runs; R1 also
reports the difference as a function of the number of Adam updates k between the logged step and the final
parameters; one JSON per run.

Training (runs/<run>/experiments/main.py, identical in all runs) is replayed with its own code objects:
  key = PRNGKey(cfg.PRNGKey); key1, key2, key = split(key, 3)    key1/key2 go to Canonical_Sampler.sample, which
                                                                   returns at once because the data files exist
  SI.init_params(key=key); key = split(key, 2)[0]
  one outer iteration = one dUdt_hist row:
     key1, key2 = split(key); batch = paired_dataloader.next(key1)
     20 x { key2 = split(key2)[0]; grad + aux of SI.loss_fn(params, batch, key2); Adam update }; key = key2
  (loss_fn_dsm splits key2 itself into the t and z keys, so replaying key2 replays t and z.)
  jax flags from cfg.jax_config (x64 off, rank promotion raise, debug_nans on).
  dUdt_hist row r, column j is the batch-mean dU/dt of grad step s = 20 r + j, computed with the parameters after
  s updates; the checkpoint holds the parameters after 50000 updates, so k = 50000 - s >= 1.

R0  rows 0..n_first-1 from the re-created initial parameters with the training updates (main.py's body_fun).
    Step 0 must reproduce dUdt_hist[0, 0] (same parameters, batch, t, z): checks PRNG chain, data, alignment, loss.
R1  the last n_last rows: the same batches, t and z as in training, FINAL parameters, no updates.  Per step the
    sample noise is identical, so replayed - logged is the effect of the k updates between the two parameter sets.
R2  the R1 training pairs, `strata` stratified t per pair (and fresh z for DSM), final parameters: the
    fixed-parameter estimate of int_0^1 I dt on the training pairs.
Reads runs/<run> only (cwd = its experiments/ for the dataloader's relative paths; no bytecode is written).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
import traceback
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np

RUNS = Path(os.environ.get("NTI_RUNS", Path(__file__).resolve().parents[1] / "runs"))
K_BINS = [(1, 20), (21, 100), (101, 300), (301, 1000)]


def json_safe(v):
    if isinstance(v, dict):
        return {str(k): json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [json_safe(x) for x in v]
    if isinstance(v, np.ndarray):
        return json_safe(v.tolist())
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=1) + "\n")
    os.replace(tmp, path)


def cmp(logged, replayed):
    logged, replayed = np.asarray(logged, np.float64).ravel(), np.asarray(replayed, np.float64).ravel()
    d = replayed - logged
    n = len(d)
    return dict(n_steps=n,
                logged_mean=float(logged.mean()), logged_se=float(logged.std(ddof=1) / np.sqrt(n)) if n > 1 else None,
                replayed_mean=float(replayed.mean()),
                replayed_se=float(replayed.std(ddof=1) / np.sqrt(n)) if n > 1 else None,
                diff_mean=float(d.mean()), diff_se=float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else None,
                diff_rms=float(np.sqrt((d ** 2).mean())), diff_max_abs=float(np.abs(d).max()),
                logged_sd=float(logged.std(ddof=1)) if n > 1 else None,
                corr=float(np.corrcoef(logged, replayed)[0, 1]) if n > 2 else None)


def k_binned(logged, replayed, k):
    out = []
    for lo, hi in K_BINS:
        sel = (k >= lo) & (k <= hi)
        if sel.sum() >= 3:
            out.append(dict(k_lo=lo, k_hi=hi, **cmp(logged[sel], replayed[sel])))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-first", type=int, default=2)
    ap.add_argument("--n-last", type=int, default=50)
    ap.add_argument("--strata", type=int, default=16)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--r1-steps", type=int, default=None,
                    help="CPU tests only: replay just the last r1-steps grad steps of each R1 row (default: all)")
    args = ap.parse_args(argv)
    t_start = time.time()
    run_dir = RUNS / args.run
    exp = run_dir / "experiments"
    out = args.out / f"{args.run}.json"
    for p in (str(run_dir / "src"), str(run_dir)):  # the run's own copy of the training code
        if p not in sys.path:
            sys.path.insert(0, p)

    import jax
    import jax.numpy as jnp
    import hydra
    import optax

    raw = (exp / "LJsolute.model").read_bytes()
    ck = pickle.loads(raw)
    cfg = ck["cfg"]
    for name, value in cfg.jax_config.items():  # main.py: same loop, before any computation
        jax.config.update(name, value)
    print("devices", jax.devices(), "jax", jax.__version__, "x64", jax.config.jax_enable_x64, flush=True)
    with open(exp / "dUdt_hist", "rb") as f:
        hist = np.asarray(pickle.load(f)["train"], np.float64)
    n_rows, nb = hist.shape
    assert nb == cfg.new_batch_every_n_step, (hist.shape, cfg.new_batch_every_n_step)
    final_step = int(np.asarray(ck["opt_state"][0].count))
    loss_type = str(cfg.model.get("loss_type", "tsm"))
    res = dict(args={k: str(v) for k, v in vars(args).items()}, run=args.run,
               checkpoint=str(exp / "LJsolute.model"), checkpoint_sha256=hashlib.sha256(raw).hexdigest(),
               checkpoint_step=final_step, loss_type=loss_type,
               sigma_min=float(cfg.model.sigma_min) if loss_type == "dsm" else 0.0,
               sigma_mid=float(cfg.model.sigma_mid) if loss_type == "dsm" else 0.0,
               PRNGKey=int(cfg.PRNGKey), jax_version=jax.__version__, hist_shape=list(hist.shape))
    assert final_step == n_rows * nb, (final_step, n_rows, nb)
    atomic_json(out, res)

    os.chdir(exp)  # upstream dataloader uses ../data/<name>
    from dataloader import paired_dataloader
    assert os.path.isfile(f"../data/{cfg.data0}") and os.path.isfile(f"../data/{cfg.data1}")
    dl = paired_dataloader(cfg)

    # ---- objects exactly as main.py ----
    target_system = hydra.utils.instantiate(cfg.target_system)
    SI = hydra.utils.instantiate(cfg.model)
    SI.num_features = target_system.num_dim
    SI.U_model = hydra.utils.instantiate(cfg.U_model)
    key = jax.random.PRNGKey(cfg.PRNGKey)
    key1, key2, key = jax.random.split(key, 3)  # key1, key2: Canonical_Sampler.sample -> returns (files exist)
    SI.init_params(key=key)
    key = jax.random.split(key, 2)[0]
    key_after_init = key
    optim = hydra.utils.instantiate(cfg.optim)
    params = SI.params
    opt_state = optim.init(params)
    final_params = ck["params"]

    def update_step(key, params, batch_pair, opt_state):
        grad_fn = jax.grad(SI.loss_fn, has_aux=True)
        grad, aux = grad_fn(params, batch_pair, key)
        updates, opt_state = optim.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, grad, opt_state, aux

    @jax.jit
    def body_fun(i, carry):
        params, _, opt_state, key, dUdt, _, data = carry
        key, _ = jax.random.split(key)
        params, grad, opt_state, aux = update_step(key, params, data, opt_state)
        _, _, dUdt_ = aux
        dUdt = dUdt.at[i].set(dUdt_)
        return params, grad, opt_state, key, dUdt, aux[:2], data

    # ---- R0 ----
    t0 = time.time()
    try:
        if args.n_first == 0:
            raise StopIteration("R0 skipped (--n-first 0)")
        rows = []
        for r in range(args.n_first):
            key1, key2 = jax.random.split(key)
            batch = dl.next(key1)
            carry = (params, params, opt_state, key2, jnp.zeros(nb), (0, 0), batch)
            params, grad, opt_state, key, dUdt, aux, _ = jax.lax.fori_loop(0, nb, body_fun, carry)
            rows.append(np.asarray(dUdt, np.float64))
            print(f"R0 row {r}: logged {np.round(hist[r, :5], 3)}  replayed {np.round(rows[-1][:5], 3)}", flush=True)
        rows = np.array(rows)
        res["R0"] = dict(rows=list(range(args.n_first)), logged=hist[:args.n_first], replayed=rows,
                         step0=dict(logged=float(hist[0, 0]), replayed=float(rows[0, 0]),
                                    rel_diff=float(abs(rows[0, 0] - hist[0, 0]) / max(abs(hist[0, 0]), 1e-3))),
                         stats=cmp(hist[:args.n_first], rows), seconds=time.time() - t0)
    except StopIteration as e:
        res["R0"] = dict(skipped=str(e))
    except Exception as e:  # keep going: R1/R2 are still informative, but record the failure verbatim
        res["R0"] = dict(error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
        print(res["R0"]["traceback"], flush=True)
    print("R0", {k: v for k, v in res["R0"].items() if k not in ("logged", "replayed", "traceback")}, flush=True)
    atomic_json(out, res)

    # ---- fast-forward the key chain (no data, no model) ----
    @jax.jit
    def advance(key):
        k1, k2 = jax.random.split(key)
        ks = []
        for _ in range(nb):
            k2 = jax.random.split(k2)[0]
            ks.append(k2)
        return k1, jnp.stack(ks), k2

    first_last = n_rows - args.n_last
    key = key_after_init
    keys_last = []
    for r in range(n_rows):
        k1, ks, key = advance(key)
        if r >= first_last:
            keys_last.append((r, k1, ks))

    # ---- R1: last rows, final params, same batches, t and z ----
    t0 = time.time()
    loss_aux = jax.jit(lambda p, b, k: SI.loss_fn(p, b, k)[1])
    js = np.arange(nb) if args.r1_steps is None else np.arange(nb - args.r1_steps, nb)
    rep, logged, B0, B1 = [], [], [], []
    for i, (r, k1, ks) in enumerate(keys_last):
        batch = dl.next(k1)
        B0.append(np.asarray(batch[0])), B1.append(np.asarray(batch[1]))
        rep.append([float(loss_aux(final_params, batch, ks[j])[2]) for j in js])
        logged.append(hist[r, js])
        if i % 10 == 0 or i == len(keys_last) - 1:
            print(f"R1 row {r}: logged mean {hist[r].mean():8.2f}  replayed mean {np.mean(rep[-1]):8.2f}  "
                  f"{time.time() - t0:.0f}s", flush=True)
    rep, logged = np.array(rep), np.array(logged)
    k_upd = final_step - (nb * np.array([r for r, _, _ in keys_last])[:, None] + js[None, :])
    per_row = [dict(row=r, k_min=int(k_upd[i].min()), logged_mean=float(logged[i].mean()),
                    replayed_mean=float(rep[i].mean()), diff_rms=float(np.sqrt(((rep[i] - logged[i]) ** 2).mean())),
                    corr=float(np.corrcoef(rep[i], logged[i])[0, 1]))
               for i, (r, _, _) in enumerate(keys_last)]
    res["R1"] = dict(rows=[first_last, n_rows - 1], steps_per_row=[int(js[0]), int(js[-1])], stats=cmp(logged, rep),
                     last_row=dict(logged=logged[-1], replayed=rep[-1], stats=cmp(logged[-1], rep[-1])),
                     last5_rows=cmp(logged[-5:], rep[-5:]), k_binned=k_binned(logged.ravel(), rep.ravel(),
                                                                                k_upd.ravel()),
                     last_step=dict(k=int(k_upd[-1, -1]), logged=float(logged[-1, -1]), replayed=float(rep[-1, -1])),
                     per_row=per_row, logged=logged, replayed=rep, k=k_upd, seconds=time.time() - t0)
    print("R1", res["R1"]["stats"], "\n   last row", res["R1"]["last_row"]["stats"], flush=True)
    for b in res["R1"]["k_binned"]:
        print(f"   k {b['k_lo']:4d}-{b['k_hi']:4d}: diff {b['diff_mean']:8.2f} +- {b['diff_se']:6.2f}  rms {b['diff_rms']:7.2f}"
              f"  corr {b['corr']:.3f}  logged sd {b['logged_sd']:.1f}", flush=True)
    atomic_json(out, res)

    # ---- R2: stratified t per training pair (+ DSM noise), final params ----
    t0 = time.time()
    try:
        X0, X1 = np.concatenate(B0), np.concatenate(B1)
        n, K = len(X0), args.strata
        rng = np.random.default_rng(args.seed)
        ts = ((np.arange(K)[None, :] + rng.random((n, K))) / K).astype(np.float32)  # [n, K]
        zkey = jax.random.PRNGKey(args.seed)

        def dudt_one(x0, x1, t, z):
            dx = ((x1 - x0) + 0.5) % 1 - 0.5  # loss_fn / loss_fn_dsm interpolant
            xt = x0 + t * dx
            xt = (xt + SI.sigma(t) * z) % 1 if loss_type == "dsm" else xt % 1
            return jax.grad(lambda tt: SI.U(final_params, xt, tt))(jnp.reshape(t, (1,)))[0]

        f = jax.jit(jax.vmap(dudt_one))
        P0, P1 = np.repeat(X0, K, 0), np.repeat(X1, K, 0)
        T = ts.reshape(-1)
        g, bs = [], 128
        for s in range(0, len(T), bs):
            z = jax.random.normal(jax.random.fold_in(zkey, s), P0[s:s + bs].shape)
            g.append(np.asarray(f(jnp.asarray(P0[s:s + bs]), jnp.asarray(P1[s:s + bs]), jnp.asarray(T[s:s + bs]), z),
                                np.float64))
        g = np.concatenate(g).reshape(n, K)
        per_pair = g.mean(1)
        res["R2"] = dict(n_pairs=n, strata=K, dF_estimate=float(per_pair.mean()),
                         se=float(per_pair.std(ddof=1) / np.sqrt(n)), nonfinite=int((~np.isfinite(g)).sum()),
                         noise="loss_fn_dsm (sigma_t z)" if loss_type == "dsm" else "none (tsm)",
                         seconds=time.time() - t0)
    except Exception as e:
        res["R2"] = dict(error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
        print(res["R2"]["traceback"], flush=True)
    print("R2", {k: v for k, v in res["R2"].items() if k != "traceback"}, flush=True)
    res["seconds"] = time.time() - t_start
    atomic_json(out, res)
    return res


if __name__ == "__main__":
    main()
