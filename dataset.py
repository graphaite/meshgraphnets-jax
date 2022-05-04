"""Utility functions for reading the datasets."""

import jax.numpy as jnp
import numpy as onp


def _parse(ds, cfg):
    """Parse raw dataset."""
    traj_length = ds[cfg.TARGET_FIELDS[0]].shape[0]
    for key, value in ds.items():
        if key not in cfg.TARGET_FIELDS:
            ds[key] = value.tile([traj_length, 1, 1])
    return ds


def load_dataset(cfg):
    """Load dataset."""
    data = onp.load(cfg.DATASET_PATH + "train.npz")
    ds = {k: jnp.array(v) for k, v in data.items()}
    ds = _parse(ds, cfg)
    return ds


def add_targets(ds, cfg):
    """Adds target and optionally history fields to dataframe."""
    ds_out = {}
    for key, val in ds.items():
        ds_out[key] = val[1:-1]
        if key in cfg.TARGET_FIELDS:
            if cfg.ADD_HISTORY:
                ds_out['prev|' + key] = val[0:-2]
            ds_out['target|' + key] = val[2:]
    return ds_out
