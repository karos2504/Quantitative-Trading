"""
Generic Weight Allocators for Portfolio Construction
"""
import numpy as np
import pandas as pd
from collections import defaultdict
from pypfopt import EfficientFrontier, risk_models

def downside_adjusted_scores(candidates: list,
                             momentum_scores: dict,
                             returns_window: pd.DataFrame,
                             downside_vol_mult: float = 3.0) -> dict:
    if returns_window.empty or len(returns_window) < 6:
        return momentum_scores

    semi_devs = {}
    for t in candidates:
        if t not in returns_window.columns:
            continue
        r   = returns_window[t].dropna()
        neg = r[r < 0]
        semi_devs[t] = np.sqrt(np.mean(neg ** 2)) if len(neg) >= 3 else 0.0

    if not semi_devs:
        return momentum_scores

    median_sd = np.median(list(semi_devs.values()))
    adjusted  = {}
    for t in candidates:
        base = momentum_scores.get(t, 0.0)
        sd   = semi_devs.get(t, 0.0)
        if median_sd > 0 and sd > downside_vol_mult * median_sd:
            adjusted[t] = base * (median_sd / sd)
        else:
            adjusted[t] = base
    return adjusted

def risk_parity_momentum_weights(candidates: list,
                                 momentum_scores: dict,
                                 returns_window: pd.DataFrame,
                                 min_weight: float = 0.0,
                                 max_weight: float = 1.0) -> dict:
    """
    Allocates weights based on (Momentum Score / Volatility).
    Targets higher Sharpe by penalizing high-vol momentum stocks.
    """
    if returns_window.empty or len(candidates) < 2:
        return momentum_proportional_weights(candidates, momentum_scores, min_weight, max_weight)

    # Use annualized volatility (sqrt(12) for monthly data)
    vols = returns_window[candidates].std() * np.sqrt(12)
    vols = vols.replace(0, np.nan).fillna(vols.median())
    
    # Calculate Risk-Adjusted Momentum Scores
    ra_scores = {
        t: max(momentum_scores.get(t, 0.0), 0.0) / vols[t]
        for t in candidates if t in vols.index
    }
    
    return momentum_proportional_weights(candidates, ra_scores, min_weight, max_weight)


def momentum_proportional_weights(candidates: list,
                                  adj_scores: dict,
                                  min_weight: float = 0.0,
                                  max_weight: float = 1.0) -> dict:
    raw   = {t: max(adj_scores.get(t, 0.0), 0.0) for t in candidates}
    total = sum(raw.values())
    if total == 0:
        eq = 1.0 / len(candidates)
        return {t: eq for t in candidates}
    w = {t: v / total for t, v in raw.items()}
    for _ in range(3):
        w = {t: min(max(v, min_weight), max_weight) for t, v in w.items()}
        s = sum(w.values())
        if s == 0: break
        w = {t: v / s for t, v in w.items()}
    return w

def get_dynamic_sector_caps(candidates: list,
                            momentum_scores: dict,
                            universe_with_sectors: dict,
                            max_sector_w_top: float,
                            max_sector_w_bull: float) -> dict:
    sec_scores = defaultdict(list)
    for t in candidates:
        sec = universe_with_sectors.get(t, "OTHER")
        sec_scores[sec].append(momentum_scores.get(t, 0.0))
    sec_avg = {s: np.mean(v) for s, v in sec_scores.items()}
    top2    = sorted(sec_avg, key=lambda s: sec_avg[s], reverse=True)[:2]
    return {s: (max_sector_w_top if s in top2 else max_sector_w_bull)
            for s in sec_avg}

def markowitz_weights(candidates: list, 
                      returns_hist: pd.DataFrame, 
                      momentum_scores: dict,
                      objective: str, 
                      is_bull: bool,
                      universe_with_sectors: dict,
                      min_weight: float = 0.0,
                      max_weight: float = 1.0,
                      max_sector_w_top: float = 0.30,
                      max_sector_w_bull: float = 0.30,
                      max_sector_w_bear: float = 0.20,
                      target_vol: float = 0.20,
                      risk_free_rate: float = 0.04) -> dict:
    hist = returns_hist[candidates].dropna()
    if len(hist) < 12 or len(candidates) < 3:
        eq = 1.0 / len(candidates)
        return {t: eq for t in candidates}

    dyn_caps   = get_dynamic_sector_caps(
        candidates, momentum_scores, universe_with_sectors, 
        max_sector_w_top, max_sector_w_bull
    )
    sector_cap = max_sector_w_bear if not is_bull else max_sector_w_bull

    try:
        S  = risk_models.CovarianceShrinkage(
                hist, returns_data=True, frequency=12).ledoit_wolf()
        mu = pd.Series({t: momentum_scores.get(t, 0.0) for t in candidates})
        ef = EfficientFrontier(mu, S,
                               weight_bounds=(min_weight, max_weight),
                               solver="CLARABEL")

        for sec in set(universe_with_sectors.get(t, "OTHER") for t in candidates):
            cap  = dyn_caps.get(sec, sector_cap) if is_bull else max_sector_w_bear
            mask = [
                1.0 if universe_with_sectors.get(t, "OTHER") == sec else 0.0
                for t in candidates
            ]
            if sum(mask) > 1:
                ef.add_constraint(
                    lambda w, m=mask, c=cap:
                        sum(w[i] * m[i] for i in range(len(m))) <= c
                )

        if objective == "min_vol":
            ef.min_volatility()
        elif objective == "efficient_risk":
            ef.efficient_risk(target_volatility=target_vol)
        else:
            ef.max_sharpe(risk_free_rate=risk_free_rate)

        cleaned = ef.clean_weights(cutoff=min_weight, rounding=4)
        return {t: w for t, w in cleaned.items() if w > 0.0}

    except Exception:
        vols = returns_hist[candidates].std()
        inv  = {t: 1.0 / vols[t] if vols[t] > 0 else 1.0 for t in candidates}
        tot  = sum(inv.values())
        raw  = {t: v / tot for t, v in inv.items()}
        cap  = {t: min(w, max_weight) for t, w in raw.items()}
        tot2 = sum(cap.values())
        if tot2 == 0:
            return {t: 1.0 / len(candidates) for t in candidates}
        return {t: w / tot2 for t, w in cap.items()}
