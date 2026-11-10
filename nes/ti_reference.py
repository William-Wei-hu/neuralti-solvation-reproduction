"""Conventional-TI reference parsed from the log of upstream run_standardTI.sh.

upstream experiments/standardTI.py prints, per slice t, running means "Computing dudt for t=X: V"
over 64 chunks of 500 frames; the last line per t is the slice mean <dU_t/dt>_t on the base path
U_t = target_system.Ut.  Slices: 0, 1, then dyadic refinement to 65 equispaced points (k/64).
Only the log file is read.

Estimates reported (no sampling error is available from the log; spread between quadratures is a
rough discretisation indicator only):
  upstream_uniform_mean   mean over all slices (what standardTI.py prints)
  trapezoid_65 / trapezoid_33 / simpson_65
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

DEFAULT_LOG = Path(__file__).resolve().parents[1] / 'outputs' / 'standardTI.log'  # stdout of upstream run_standardTI.sh
PAT = re.compile(r'Computing dudt for t=([0-9.]+): (-?[0-9.]+(?:e[-+]?\d+)?)')
EST = re.compile(r'TI estimate with (\d+) slices: (-?[0-9.]+)')


def parse(log=DEFAULT_LOG):
    text = Path(log).read_text(errors='replace').replace('\r', '\n')
    last, chunks = {}, {}
    for m in PAT.finditer(text):
        k = int(round(float(m.group(1)) * 64))
        last[k] = float(m.group(2))
        chunks[k] = chunks.get(k, 0) + 1
    printed = [(int(n), float(v)) for n, v in EST.findall(text)]
    return last, chunks, printed


def _trap(y, dt):
    return float(dt * (y[0] / 2 + y[1:-1].sum() + y[-1] / 2))


def _simpson(y, dt):
    return float(dt / 3 * (y[0] + y[-1] + 4 * y[1:-1:2].sum() + 2 * y[2:-1:2].sum()))


def reference(log=DEFAULT_LOG):
    last, chunks, printed = parse(log)
    complete = {k: v for k, v in last.items() if chunks.get(k, 0) >= 64}
    out = dict(log=str(log), slices_found=len(last), slices_complete=len(complete),
               upstream_printed=printed,
               slices={f'{k / 64:.6f}': v for k, v in sorted(complete.items())})
    for level, n in ((6, 65), (5, 33), (4, 17), (3, 9)):
        step = 64 // (n - 1)
        ks = list(range(0, 65, step))
        if all(k in complete for k in ks):
            y = np.array([complete[k] for k in ks])
            dt = 1.0 / (n - 1)
            out[f'trapezoid_{n}'] = _trap(y, dt)
            out[f'simpson_{n}'] = _simpson(y, dt)
            out[f'upstream_uniform_mean_{n}'] = float(y.mean())
    best = next((n for n in (65, 33, 17, 9) if f'trapezoid_{n}' in out), None)
    out['best_n'] = best
    out['complete'] = best == 65
    if best:
        out['TI'] = out[f'trapezoid_{best}']
        vals = [out[k] for k in (f'trapezoid_{best}', f'simpson_{best}', f'upstream_uniform_mean_{best}')]
        coarse = 33 if best == 65 else None
        if coarse and f'trapezoid_{coarse}' in out:
            vals.append(out[f'trapezoid_{coarse}'])
        out['quadrature_spread'] = float(max(vals) - min(vals))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', default=str(DEFAULT_LOG))
    a = ap.parse_args()
    import json
    print(json.dumps(reference(a.log), indent=1))
