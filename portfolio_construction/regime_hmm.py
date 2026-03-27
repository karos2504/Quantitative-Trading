"""
Hidden Markov Model (HMM) Regime Detection

Identifies latent market states from log-returns and rolling realised
volatility.  State 0 is always the lowest-volatility (bull) regime,
regardless of HMM initialisation order.
"""

import warnings
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

_VOL_WINDOW = 21   # rolling window for realised vol feature (trading days)
_MIN_ROWS   = 30   # absolute floor for a meaningful HMM fit


def fit_hmm_regimes(prices: pd.Series, vix_prices: pd.Series = None, n_components: int = 3) -> pd.Series:
    """
    Fit a Gaussian HMM to a price series and return a regime label series.

    Features used: daily log-returns and rolling realised volatility
    (window = ``_VOL_WINDOW`` days, min_periods = ``_VOL_WINDOW``).
    Using ``min_periods = window`` (not 2) avoids near-zero vol estimates
    during the warm-up period that would distort state assignments.

    States are renumbered so that State 0 always corresponds to the
    lowest mean realised volatility.  The caller maps {0 -> bull, 1 -> bear}.

    Parameters
    ----------
    prices : pd.Series
        Time-indexed daily price series.
    n_components : int
        Number of latent regimes (default: 2).

    Returns
    -------
    pd.Series
        Integer regime labels (dtype float so NaN is representable),
        aligned to ``prices.index``.  Observations before the rolling
        warm-up window are left as ``NaN`` and are *not* forward-filled,
        so callers can detect and handle the unobserved prefix explicitly.
        Within the observed region, labels are forward-filled to handle any
        internal gaps produced by ``predict``.

    Raises
    ------
    ValueError
        If there are fewer than ``max(_MIN_ROWS, n_components * 10)`` usable
        observations after feature construction.
    """
    returns    = np.log(prices / prices.shift(1))
    volatility = returns.rolling(window=_VOL_WINDOW, min_periods=_VOL_WINDOW).std()

    features = pd.concat([returns, volatility], axis=1).dropna()
    features.columns = ["ret", "vol"]

    # Add trend feature (price relative to 200-day MA)
    ma200 = prices.rolling(window=200, min_periods=200).mean()
    trend = (prices / ma200 - 1).dropna()
    
    # Re-align features
    features = pd.concat([features, trend], axis=1).dropna()
    features.columns = ["ret", "vol", "trend"]

    if vix_prices is not None:
        vix_aligned = vix_prices.reindex(features.index).ffill().dropna()
        features = pd.concat([features, vix_aligned], axis=1).dropna()
        features.columns = ["ret", "vol", "trend", "vix"]

    min_required = max(_MIN_ROWS, n_components * 15)
    if len(features) < min_required:
        raise ValueError(
            f"Insufficient data: {len(features)} usable rows after feature "
            f"construction (need at least {min_required})."
        )

    X_scaled = StandardScaler().fit_transform(features.values)

    hmm_model = GaussianHMM(
        n_components=n_components,
        covariance_type="full",
        n_iter=200, # Increased iterations for better convergence
        random_state=42,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        hmm_model.fit(X_scaled)

    for w in caught:
        # Filter out common convergence warnings if they aren't critical
        if "ConvergenceWarning" in str(w.category):
            continue
        print(f"HMM warning: {w.category.__name__}: {w.message}")

    raw_states = hmm_model.predict(X_scaled)

    # Reorder states by ascending mean realised volatility (original scale).
    state_vol = np.array([
        features["vol"].values[raw_states == s].mean()
        for s in range(n_components)
    ])
    order  = np.argsort(state_vol)          # ascending: lowest vol = index 0
    remap  = np.empty(n_components, dtype=int)
    for new_id, old_id in enumerate(order):
        remap[old_id] = new_id
    ordered_states = remap[raw_states]

    # Build output aligned to the original prices index.
    # Rows before the warm-up window stay NaN (no forward-fill into the
    # unobserved prefix — that would fabricate look-ahead-free labels from
    # future-derived state assignments).
    regime = pd.Series(np.nan, index=prices.index, dtype=float)
    regime.loc[features.index] = ordered_states.astype(float)

    # Forward-fill only within the observed region.
    first_obs = features.index[0]
    regime.loc[first_obs:] = regime.loc[first_obs:].ffill()

    return regime

def _smooth_regime(regime: pd.Series, min_duration: int = 5) -> pd.Series:
    """Suppress regime flips shorter than min_duration days."""
    smoothed = regime.copy()
    state    = regime.iloc[0]
    count    = 0
    pending  = None
    for i, val in enumerate(regime):
        if val == state:
            count  += 1
            pending = None
        else:
            if pending is None:
                pending       = val
                pending_count = 1
            elif val == pending:
                pending_count += 1
                if pending_count >= min_duration:
                    state   = pending
                    pending = None
            else:
                pending       = val
                pending_count = 1
        smoothed.iloc[i] = state
    return smoothed
