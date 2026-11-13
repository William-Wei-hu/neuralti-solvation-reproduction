"""Convert a MACE-OFF Torch model to a mace-jax bundle read by mace_iface.py.

  python convert_mace_off.py MACE-OFF23_small.model bundles/off23s

Writes OUT_DIR/{params.msgpack, config.json, convert-info.json} (float64 parameters as converted by mace-jax).
Needs mace-torch (0.3.17 was used) and mace-jax (commit e8a6f2d was used); runs on CPU.
The MACE-OFF weights (https://github.com/ACEsuit/mace-off) are distributed under the Academic Software Licence
(non-commercial); they and the converted bundle are not part of this repository.

This is the conversion step of our pre-measurement script, extracted unchanged except for the input / output paths;
it has not been rerun in this standalone form.  mace_iface.MaceIface loads the bundle with
mace_jax.tools.bundle.load_model_bundle (backend 'jax') or, for backend 'cueq', with the same config.json plus
equivariance_config = dict(backend='cueq', layout='mul_ir', group='O3', optimize_all=True, conv_fusion=True).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def log(*a):
    print(*a, flush=True)


def convert(model_path, out_dir):
    import jax
    jax.config.update('jax_enable_x64', True)
    import torch
    from flax import serialization
    from mace_jax.cli.mace_jax_from_torch import _serialize_for_json, convert_model
    from mace_jax.nnx_utils import state_to_serializable_dict
    from mace_jax.tools.import_from_torch import extract_config_mace_model
    log('devices', jax.devices(), 'torch', torch.__version__)
    t0 = time.time()
    tm = torch.load(str(model_path), map_location='cpu', weights_only=False)
    tm = tm['model'] if isinstance(tm, dict) and 'model' in tm else tm
    tm.eval()
    tdtype = str(next(tm.parameters()).dtype)
    config = extract_config_mace_model(tm)
    if 'error' in config:
        raise RuntimeError(config['error'])
    config['torch_model_class'] = tm.__class__.__name__
    _, state, _ = convert_model(tm, config)
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / 'params.msgpack').write_bytes(serialization.to_bytes(state_to_serializable_dict(state)))
    (d / 'config.json').write_text(json.dumps(_serialize_for_json(config), indent=2))
    info = dict(model=Path(model_path).name, torch_class=tm.__class__.__name__, torch_param_dtype=tdtype,
                r_max=float(config['r_max']), atomic_numbers=[int(z) for z in config['atomic_numbers']],
                num_interactions=config.get('num_interactions'), hidden_irreps=str(config.get('hidden_irreps')),
                n_params_torch=int(sum(p.numel() for p in tm.parameters())), seconds=time.time() - t0)
    (d / 'convert-info.json').write_text(json.dumps(info, indent=1))
    log('CONVERT', json.dumps(info))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='MACE-OFF Torch model -> mace-jax bundle')
    ap.add_argument('model', help='MACE-OFF .model file (mace-torch)')
    ap.add_argument('out_dir', help='bundle directory, e.g. bundles/off23s')
    a = ap.parse_args()
    convert(a.model, a.out_dir)
