"""Shared configuration for the experiment scripts.

Every script accepts ``--quick``, which reduces dimensions, ensemble sizes and
grid resolutions so that the whole suite runs in a couple of minutes.  The
numbers quoted in the paper are those produced by the default (full) settings;
``--quick`` reproduces them only qualitatively and is intended for checking that
the pipeline runs.

Random seeds are fixed and reported, so every figure is reproducible bit for
bit on a given NumPy version.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Parameters of the benchmark law, fixed across Sections 10.1--10.5.
A0 = 1.6  # atoms of the two-atom law at +- a0
BETA = 1.0  # constant schedule, so Lambda(t) = t and alpha_t = exp(-t)
SEED = 20240517

RESULTS = os.environ.get("FREEDDPM_RESULTS", "results")


def parse(description: str, **extra):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--quick", action="store_true", help="small, fast configuration")
    p.add_argument("--seed", type=int, default=SEED)
    for name, kw in extra.items():
        p.add_argument(f"--{name.replace('_', '-')}", **kw)
    return p.parse_args()


def rng(args):
    return np.random.default_rng(args.seed)


def cfg(args, full, quick):
    """Pick the full or quick value of a parameter."""
    return quick if args.quick else full


def report(name: str, values: dict):
    """Print a result block and append it to ``results/<name>.json``."""
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, f"{name}.json")
    payload = {"experiment": name, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), **values}
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=float)
    width = max(len(k) for k in values) if values else 0
    print(f"\n[{name}]")
    for k, v in values.items():
        if isinstance(v, float):
            print(f"  {k:<{width}} = {v:.6g}")
        else:
            print(f"  {k:<{width}} = {v}")
    print(f"  (written to {path})")
    return payload

# ---- resumable runs: a time budget and a pickle cache of finished units ---------------------------
import pickle
import time as _time

_T0 = _time.time()


def cache_load(name):
    """Return the dict of finished units cached for ``name`` (empty if none)."""
    path = os.path.join(RESULTS, "cache", f"{name}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    return {}


def cache_save(name, units):
    os.makedirs(os.path.join(RESULTS, "cache"), exist_ok=True)
    with open(os.path.join(RESULTS, "cache", f"{name}.pkl"), "wb") as fh:
        pickle.dump(units, fh)


def out_of_time(args):
    """True once ``--budget`` seconds have elapsed (0 means no limit).  Scripts that use it save their
    finished units, exit with status 3, and are simply run again to continue."""
    return getattr(args, "budget", 0.0) > 0 and (_time.time() - _T0) > args.budget
