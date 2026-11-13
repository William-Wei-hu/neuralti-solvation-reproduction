"""Tier B of j2_switch.py re-run as is, also storing frames on a 21-point lambda grid (input of
hysteresis_analysis.py).

  python lagB_switch.py run --roots ROOTS_DIR --out OUT_DIR [--ref J2_OUT_DIR]
  python lagB_switch.py selfcheck OUT_DIR --ref J2_OUT_DIR

Physics unchanged from j2_switch.py: K2 = HMC full refresh, L = 40, dt = 0.5 fs, float64, linear schedule, T = 8 ps,
K = 400, roots at positions 32-63 of the (segment, chain) order of the j1_endpoints.py roots, stage keys
stage_keys(20261112, dir, 10 * chain + segment, K) - the same roots and keys as tier B of j2_switch.py.
Changes relative to j2_switch.py (implementation only):
  * mlip_nes.make_switch(record=True): the scan also returns x before each work jump, i.e. the state after the MH step
    at lam_k.  Frames kept: stages k = 0, 20, ..., 380 (k = 0 is the root) plus the final state, 21 frames per
    trajectory, at lam = k / K (fwd) or 1 - k / K (rev); the lam of each frame is stored explicitly.
  * 32 trajectories per device call (j2_switch.py: 16), one call per direction, written to disk when the call returns.
SELF-CHECK against tier B of j2_switch.py (fixed before the run; only if --ref is given).  record=True changes the
outputs of the scan, and the batch is 32 instead of 16, so bitwise identity is not required: chaos amplifies roundoff.
Matched by root position, in both directions (64 trajectories):
  S1 first MH step: |acc_new[:, 0] - acc_ref[:, 0]| <= 1e-6 for every trajectory (depends only on root, key, potential)
  S2 first 10 stages: |acc difference| <= 1e-6 for all; first lambda bin in travel order (20 stages, 400 fs; fwd bin 0,
     rev bin 19): |Wbin difference| <= 1e-4 kBT for all
  S3 whole trajectory (reported, not a pass criterion): max |dW|, the number of trajectories with |dW| > 1e-6 kBT, and
     per trajectory the first stage with |acc difference| > 1e-3; mean W of both runs per direction
  PASS = S1 and S2.  If every |dW| <= 1e-6 kBT: "trajectories identical"; otherwise, with PASS: "identical up to
  numerical noise, later divergence is chaotic amplification".  FAIL is reported as is.
Time guard: a chunk is not started if its projected end is later than --time-limit-min minus 8 min (default 66).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax

jax.config.update('jax_enable_x64', True)
jax.config.update('jax_default_matmul_precision', 'highest')
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mlip_nes as M  # noqa: E402

KT = M.kbt(298.15)
SEED, NBINS, L_LEAP, DT, STAGE_FS, NGRID = 20261112, 20, 40, 0.5, 20.0, 20
T_FS, SL, CHUNK = 8000.0, slice(32, 64), 32
ROOTS = REF = RUN = None                       # set from the command line
T_LIMIT = 66 * 60
K = int(round(T_FS / STAGE_FS))
STRIDE = K // NGRID
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)


def load_roots():
    out = {}
    for e in (0, 1):
        z = np.load(ROOTS / f'roots_end{e}.npz')
        o = np.lexsort((z['chain_id'], z['segment']))
        r = {k: z[k] for k in z.files}
        for k in ('x', 'chain_id', 'segment', 'kernel'):
            r[k] = r[k][o]
        out[e] = r
    return out


def setup():
    import mace_iface as MI
    roots = load_roots()
    r0 = roots[0]
    Z, L, sol = r0['Z'], float(r0['L']), r0['solute'].astype(bool)
    pad_src = json.loads((ROOTS / 'j1.json').read_text())['info']['pad']
    m = MI.MaceIface('off23s', Z, L, sol, pad=int(pad_src))
    log('devices', jax.devices(), json.dumps(m.info))
    one = m._one

    def efn(x, lam):
        _, eint, F, ne = one(x, lam)
        return eint, F, jnp.asarray(ne).astype(jnp.int32)
    mass = jnp.asarray(m.masses)
    pairs = M.oh_pairs_from(Z, r0['x'][0], L, sol)
    sidx, widx = np.where(sol)[0], np.where(~sol)[0]
    base = M.make_probes(sidx, widx, pairs, L)
    C = sidx[int(np.where(Z[sol] == 6)[0][0])]
    Hm = sidx[np.where(Z[sol] == 1)[0]]

    def probes(x):
        d = base(x)
        ch = M.min_image(x[Hm] - x[C], L)
        d['max_ch'] = jnp.sqrt((ch ** 2).sum(-1).max())
        return d
    return roots, m, efn, mass, probes, L, widx, C


def run():
    RUN.mkdir(parents=True, exist_ok=False)
    roots, m, efn, mass, probes, L, widx, C = setup()
    meta = dict(info=m.info, roots=str(ROOTS), ref=str(REF), kernel='K2 hmc L=40', dt=DT,
                T_fs=T_FS, K=K, roots_pos=[SL.start, SL.stop], chunk=CHUNK, frame_stride=STRIDE, chunks=[])
    (RUN / 'meta.json').write_text(json.dumps(meta, indent=1, default=float))
    sw = M.make_switch(efn, kind='hmc', m=mass, dt=DT, L=L_LEAP, box=L, kT=KT, nbins=NBINS, record=True, probes=probes)
    swb = jax.jit(jax.vmap(sw, in_axes=(0, 0, None, 0)))
    per_call = None
    for d, e in (('fwd', 0), ('rev', 1)):
        R = roots[e]
        idx = np.arange(SL.start, SL.stop)
        lam_sched = M.schedule(K, d)
        lams = jnp.asarray(lam_sched)
        for c0 in range(0, len(idx), CHUNK):
            ci = idx[c0:c0 + CHUNK]
            calls = len(ci) * (K * (L_LEAP + 1) + 1)
            if per_call is not None and time.time() - T0 + calls * per_call > T_LIMIT - 8 * 60:
                log(f'time guard: {d} chunk {c0} not started')
                meta['time_guard'] = f'stopped before {d} chunk {c0}'
                (RUN / 'meta.json').write_text(json.dumps(meta, indent=1, default=float))
                return
            rid = 10 * R['chain_id'][ci] + R['segment'][ci]
            keys = jnp.stack([M.stage_keys(SEED, d, int(r), K) for r in rid])
            p0 = jax.vmap(lambda k: M.draw_p(k, mass, KT, jnp.float64))(keys[:, 0])
            ts = time.time()
            out = jax.block_until_ready(swb(jnp.asarray(R['x'][ci]), p0, lams, keys[:, 1:]))
            sec = time.time() - ts
            per_call = sec / calls
            xr = np.asarray(out['x_before_jump'])                     # (n, K, N, 3), stage k = state at lam_k
            x = np.asarray(out['x'])
            ks = np.arange(0, K, STRIDE)
            frames = np.concatenate([xr[:, ks], x[:, None]], 1).astype(np.float32)
            frame_lam = np.concatenate([lam_sched[ks], lam_sched[-1:]])
            dC = M.min_image(x[:, widx] - x[:, [C]], L)
            dC = np.sqrt((dC ** 2).sum(-1)).min(1)
            rec = dict(W=np.asarray(out['W']), Wbins=np.asarray(out['Wbins']), acc=np.asarray(out['acc']),
                       n_edges_max=np.asarray(out['n_edges_max']), nonfinite=np.asarray(out['nonfinite']),
                       min_ss=np.asarray(out['min_solute_solvent']).min(1),
                       max_oh=np.asarray(out['max_oh']).max(1), max_ch=np.asarray(out['max_ch']).max(1),
                       final_C_water=dC, frames=frames, frame_lam=frame_lam, frame_stage=np.append(ks, K),
                       x_final=x.astype(np.float32), x_root=np.asarray(R['x'][ci]).astype(np.float32),
                       chain_id=R['chain_id'][ci], segment=R['segment'][ci], root_pos=ci,
                       seconds=np.float64(sec), calls=np.int64(calls))
            f = RUN / f'lagB-{d}-c{c0 // CHUNK:02d}.npz'
            np.savez(f, **rec)
            meta['chunks'].append(dict(file=f.name, seconds=sec, calls=calls, ms_per_config=1e3 * sec / calls))
            (RUN / 'meta.json').write_text(json.dumps(meta, indent=1, default=float))
            log(f'{d} chunk {c0 // CHUNK}: <W>={rec["W"].mean():.3f} sd={rec["W"].std():.3f} '
                f'acc={np.nanmean(rec["acc"]):.3f} edges={int(rec["n_edges_max"].max())} '
                f'nonfinite={int(rec["nonfinite"].sum())} oh={rec["max_oh"].max():.3f} frames={frames.shape} '
                f'{sec:.0f}s {1e3 * sec / calls:.2f} ms/config')


def selfcheck(run):
    """Compare with tier B of j2_switch.py (REF) by root position (criteria in the module docstring)."""
    run = Path(run)
    out = dict(criteria='S1 |dacc[0]|<=1e-6 all; S2 |dacc[:10]|<=1e-6 all and first-bin |dWbin|<=1e-4 kBT all')
    s1 = s2 = True
    allW = []
    for d, b in (('fwd', 0), ('rev', NBINS - 1)):
        new = [dict(np.load(f)) for f in sorted(run.glob(f'lagB-{d}-c*.npz'))]
        old = [dict(np.load(f)) for f in sorted(REF.glob(f'tierB-{d}-c*.npz'))]
        cat = lambda L_, k: np.concatenate([p[k] for p in L_])   # noqa: E731
        pn, po = cat(new, 'root_pos'), cat(old, 'root_pos')
        on, oo = np.argsort(pn), np.argsort(po)
        assert np.array_equal(pn[on], po[oo])
        an, ao = cat(new, 'acc')[on], cat(old, 'acc')[oo]
        wn, wo = cat(new, 'W')[on], cat(old, 'W')[oo]
        bn, bo = cat(new, 'Wbins')[on][:, b], cat(old, 'Wbins')[oo][:, b]
        da = np.abs(an - ao)
        first = [int(np.argmax(r > 1e-3)) if (r > 1e-3).any() else None for r in da]
        r = dict(n=len(wn), S1_max_dacc0=float(da[:, 0].max()), S2_max_dacc_first10=float(da[:, :10].max()),
                 S2_max_dWbin_first=float(np.abs(bn - bo).max()), S3_max_dW=float(np.abs(wn - wo).max()),
                 S3_n_dW_gt_1e6=int((np.abs(wn - wo) > 1e-6).sum()), S3_first_stage_dacc_gt_1e3=first,
                 mean_W_new=float(wn.mean()), mean_W_ref=float(wo.mean()),
                 se_W=float(np.hypot(wn.std(ddof=1), wo.std(ddof=1)) / np.sqrt(len(wn))))
        s1 &= r['S1_max_dacc0'] <= 1e-6
        s2 &= r['S2_max_dacc_first10'] <= 1e-6 and r['S2_max_dWbin_first'] <= 1e-4
        allW.append(np.abs(wn - wo))
        out[d] = r
    ident = bool(np.concatenate(allW).max() <= 1e-6)
    out.update(S1=bool(s1), S2=bool(s2), PASS=bool(s1 and s2),
               verdict=('trajectories identical' if (s1 and s2 and ident) else
                        'identical up to numerical noise; later divergence is chaotic amplification' if (s1 and s2)
                        else 'FAIL'))
    (run / 'selfcheck.json').write_text(json.dumps(out, indent=1))
    log('SELFCHECK', json.dumps({k: v for k, v in out.items() if k not in ('fwd', 'rev')}),
        json.dumps({d: {k: v for k, v in out[d].items() if k != 'S3_first_stage_dacc_gt_1e3'} for d in ('fwd', 'rev')}))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='tier B re-run with frames on a 21-point lambda grid')
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--roots', required=True, help='output directory of j1_endpoints.py')
    r.add_argument('--out', required=True, help='new output directory (must not exist)')
    r.add_argument('--ref', default=None, help='output directory of j2_switch.py; enables the self-check')
    r.add_argument('--time-limit-min', type=float, default=66.0)
    c = sub.add_parser('selfcheck')
    c.add_argument('run_dir')
    c.add_argument('--ref', required=True, help='output directory of j2_switch.py')
    a = ap.parse_args()
    REF = Path(a.ref) if a.ref else None
    if a.cmd == 'run':
        ROOTS, RUN, T_LIMIT = Path(a.roots), Path(a.out), 60 * a.time_limit_min
        run()
        if REF is not None and len(list(RUN.glob('lagB-rev-c*.npz'))):
            selfcheck(RUN)
    else:
        selfcheck(a.run_dir)
