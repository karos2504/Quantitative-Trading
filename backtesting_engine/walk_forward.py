import numpy as np
import pandas as pd
from datetime import timedelta
from backtesting import Backtest

from config.settings import CASH, COMMISSION, WF_TRAIN_MONTHS, WF_TEST_MONTHS

def run_walk_forward(ticker, df, strategy_class, best_params, precompute_fn=lambda x: x, min_trades_valid=1):
    records = []

    if hasattr(df.index, 'tz') and df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)

    start = df.index[0]
    end   = df.index[-1]

    train_td = timedelta(days=WF_TRAIN_MONTHS * 30)
    test_td  = timedelta(days=WF_TEST_MONTHS  * 30)

    window_start = start
    window_id    = 1

    while True:
        train_end  = window_start + train_td
        test_start = train_end
        test_end   = test_start + test_td
        
        if test_end > end: break

        full_slice = df.loc[window_start:test_end]
        if len(full_slice) < 50:
            window_start = test_start
            window_id   += 1
            continue

        try:
            proc = precompute_fn(full_slice) if precompute_fn else full_slice
            if hasattr(proc.index, 'tz') and proc.index.tz is not None:
                proc.index = proc.index.tz_localize(None)
            
            proc_test = proc.loc[test_start:]
            # Provide some safety handling for very small test periods
            if len(proc_test) < 50:
                raise ValueError("too few test bars")

            bt = Backtest(proc_test, strategy_class, cash=CASH, commission=COMMISSION, trade_on_close=False, finalize_trades=True)
            stats = bt.run(**best_params)
            
            n_t   = int(stats["# Trades"])
            wr    = round(float(stats.get("Win Rate [%]") or 0), 1)
            valid = n_t >= min_trades_valid

            records.append({
                "window":       window_id,
                "train_end":    train_end.date(),
                "test_start":   test_start.date(),
                "test_end":     test_end.date(),
                "return_%":     round(stats["Return [%]"], 2) if valid else np.nan,
                "sharpe":       round(stats["Sharpe Ratio"] or 0, 3) if valid else np.nan,
                "max_dd_%":     round(stats["Max. Drawdown [%]"], 2) if valid else np.nan,
                "win_rate_%":   wr if valid else np.nan,
                "n_trades":     n_t,
                "valid":        valid,
            })
        except Exception as e:
            records.append({
                "window":     window_id, "train_end":  train_end.date(), "test_start": test_start.date(),
                "test_end":   test_end.date(), "return_%":   np.nan, "sharpe":     np.nan, "max_dd_%":   np.nan,
                "win_rate_%": np.nan, "n_trades":   0, "valid":      False, "error":      str(e),
            })

        window_start = test_start
        window_id   += 1

    wf_df = pd.DataFrame(records)
    if wf_df.empty: return wf_df

    print(f"\n{'─'*64}")
    print(f"  Walk-Forward OOS — {ticker}  (params: {best_params})")
    print(f"{'─'*64}")
    display_cols = ["window","train_end","test_start","test_end",
            "return_%","sharpe","max_dd_%","win_rate_%","n_trades", "valid"]
    print(wf_df[[c for c in display_cols if c in wf_df.columns]].to_string(index=False))
    
    valid_df = wf_df[wf_df.get("valid", wf_df["n_trades"] > 0)]
    if not valid_df.empty:
        print(f"\n  Avg OOS Return : {valid_df['return_%'].mean():.2f}%")
        print(f"  Avg OOS Sharpe : {valid_df['sharpe'].mean():.3f}")
        print(f"  Avg Win Rate   : {valid_df['win_rate_%'].mean():.1f}%")
        print(f"  Win windows    : {(valid_df['return_%'] > 0).sum()}/{len(valid_df)}")
    else:
        print(f"\n  ⚠️  No valid windows (all had < {min_trades_valid} trades)")
        
    return wf_df
