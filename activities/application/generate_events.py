"""Generate only the signal and background samples used in these exercises.

The Gaussian mixtures, detector smearing and expected yields are unchanged
from the ML4HEP-TIFR tutorial. Sample counts control Monte Carlo precision,
not the physical yields. See LICENSE.txt for the upstream license.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils import background_components, signal_components, smearing_parameters

N_BKG = N_SIG = 20_000
SEED = 42
LAM_BKG = 1_000_000.0
LAM_SIG = 1_100.0


def sample_mixture(components, n, rng):
    fractions = np.array([c[0] for c in components], dtype=float)
    counts = rng.multinomial(n, fractions / fractions.sum())
    chunks = [
        rng.multivariate_normal(mean, cov, size=count)
        for (_, mean, cov), count in zip(components, counts)
        if count
    ]
    values = np.concatenate(chunks, axis=0)
    rng.shuffle(values)
    return values


def generate_events(n_bkg=N_BKG, n_sig=N_SIG, seed=SEED, output_dir="dataframes"):
    """Write reproducible samples with truth/reconstructed features and weights."""
    if min(n_bkg, n_sig) < 2:
        raise ValueError("Each sample needs at least two events.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    scale, resolution = smearing_parameters()
    for name, components, count, expected_yield in [
        ("background", background_components(), n_bkg, LAM_BKG),
        ("signal", signal_components(), n_sig, LAM_SIG),
    ]:
        truth = sample_mixture(components, count, rng)
        reco = rng.normal(truth * scale, resolution)
        frame = pd.DataFrame(truth, columns=[f"z{i}" for i in range(1, 6)])
        frame[[f"x{i}" for i in range(1, 6)]] = reco
        frame["weight"] = expected_yield / count
        frame.to_parquet(output_dir / f"{name}.parquet", index=False)
    print(f"Generated {n_bkg:,} background + {n_sig:,} signal events in {output_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_bkg", type=int, default=N_BKG)
    parser.add_argument("--n_sig", type=int, default=N_SIG)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate_events(args.n_bkg, args.n_sig, args.seed)
