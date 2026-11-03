"""Shared distributions and train/inference split for the Mandi exercises.

Adapted from iris-hep/nsbi-lhc-toolkit; see LICENSE.txt and README.md.
"""

import numpy as np

def split_train_inference(df, train_fraction=0.5, n_train=None, seed=0):
    """Deterministically partition ``df`` into disjoint (train, inference) sets.

    Density ratios must be *trained* and the likelihood *evaluated* on
    different events -- otherwise the fit inherits the classifier's overfitting.
    Call this with **identical** ``train_fraction``/``n_train`` and ``seed`` in
    the training notebooks (2a, 2b) and the fitting notebook (3): the shared
    seed fixes the permutation, so the "train" half in one notebook and the
    "inference" half in the other are guaranteed non-overlapping.

    Parameters
    ----------
    df : pandas.DataFrame
        Sample to split (e.g. background or signal).
    train_fraction : float
        Fraction of ``df`` assigned to the training subset. Ignored if
        ``n_train`` is given.
    n_train : int or None
        Absolute number of events for the training subset. Takes priority over
        ``train_fraction`` when not None.
    seed : int
        Seed for the permutation. Must match across notebooks.

    Returns
    -------
    (train, inference) : tuple[pandas.DataFrame, pandas.DataFrame]
        Disjoint subsets. If a ``weight`` column is present, each subset's
        weights are rescaled so the subset still sums to the full expected
        yield (the per-event weights represent an expected yield, which the
        physical measurement expects regardless of how many MC events we use).
    """
    n = len(df)
    k = int(n_train) if n_train is not None else int(round(train_fraction * n))
    k = max(0, min(n, k))
    perm = np.random.default_rng(seed).permutation(n)
    train = df.iloc[np.sort(perm[:k])].reset_index(drop=True)
    inference = df.iloc[np.sort(perm[k:])].reset_index(drop=True)
    if "weight" in df.columns:
        total = df["weight"].sum()
        for sub in (train, inference):
            s = sub["weight"].sum()
            if s > 0:
                sub["weight"] = sub["weight"] * (total / s)
    return train, inference


# ---------------------------------------------------------------------------
# Gaussian-mixture model definition
# ---------------------------------------------------------------------------
# This is the SINGLE SOURCE OF TRUTH for the toy distributions.  Both
# ``generate_events.py`` (which samples from these mixtures) and the
# parameter-fitting notebook (which evaluates the *analytic* density ratios to
# separate training quality from inference-statistics effects) import these
# definitions, so the "truth" ratios are guaranteed to match the data.


def build_cov(sigmas, correlations=None):
    """Full covariance from per-feature sigmas and a sparse dict of pairwise
    correlations ``{(i, j): rho}``.  Off-diagonal correlations "rotate" the
    components; mixing rotated components gives the curved density contours."""
    sigmas = np.asarray(sigmas, dtype=float)
    corr = np.eye(len(sigmas))
    if correlations:
        for (i, j), rho in correlations.items():
            corr[i, j] = corr[j, i] = rho
    return np.outer(sigmas, sigmas) * corr


# Shared broad component -- present in EVERY sample.  Guarantees overlapping
# support so trained (and truth) density ratios stay bounded everywhere.
BASE_MEAN = np.array([2.5, 2.0, 3.0, 1.5, 2.5])
BASE_SIGMA = np.array([2.5, 2.4, 2.5, 2.4, 2.5])
BASE_COV = build_cov(BASE_SIGMA)
BASE_FRAC = 0.20


def background_components():
    """Bimodal, correlated background mixture: two modes plus the base."""
    bg1 = (
        0.45,
        np.array([2.0, 1.0, 2.5, 0.8, 2.0]),
        build_cov([1.0, 0.9, 1.2, 0.9, 1.0], {(0, 1): 0.6, (3, 4): 0.5}),
    )
    bg2 = (
        0.35,
        np.array([4.2, 3.1, 4.6, 2.6, 3.6]),
        build_cov([0.9, 1.0, 1.0, 0.9, 1.1], {(0, 1): -0.5, (2, 3): 0.5}),
    )
    base = (BASE_FRAC, BASE_MEAN, BASE_COV)
    return [bg1, bg2, base]


def signal_components(v=10):
    """Signal mixture at parameter value ``v`` (default ``v=10``, the signal
    used in the measurement).  Two modes whose means slide with ``t = v / 10``
    (so v=0 sits close to the background and v=10 is the most displaced), plus
    the shared base component."""
    t = v / 10.0
    sig_a = (
        0.50,
        np.array([2.2, 1.3, 2.8, 1.0, 2.2]) + t * np.array([2.3, 1.9, 1.5, 1.3, 1.8]),
        build_cov([0.9, 0.8, 1.0, 0.8, 0.9], {(0, 2): 0.5, (1, 4): 0.4}),
    )
    sig_b = (
        0.30,
        np.array([3.5, 2.5, 3.8, 2.0, 3.0]) + t * np.array([0.8, 0.7, 1.2, 0.6, 1.0]),
        build_cov([0.8, 0.9, 0.9, 0.8, 1.0], {(0, 1): 0.5, (2, 4): -0.4}),
    )
    base = (BASE_FRAC, BASE_MEAN, BASE_COV)
    return [sig_a, sig_b, base]


def mixture_density(X, components):
    """Evaluate the mixture pdf at points ``X`` (shape ``(n, n_dim)``).

    ``components`` is a list of ``(fraction, mean, cov)``; fractions are
    renormalised to sum to 1.  Returns an array of shape ``(n,)``.
    """
    from scipy.stats import multivariate_normal

    X = np.asarray(X, dtype=float)
    fracs = np.array([c[0] for c in components], dtype=float)
    fracs = fracs / fracs.sum()
    p = np.zeros(len(X))
    for (_, mean, cov), f in zip(components, fracs):
        p += f * multivariate_normal(mean, cov).pdf(X)
    return p

def smearing_parameters():
    scale = np.array([1.2, 1.1, 0.99, 0.96, 1.01])
    resolution = np.array([1.0, 0.1, 0.9, 1.3, 0.2])
    return [scale, resolution]

