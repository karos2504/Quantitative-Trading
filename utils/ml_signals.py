"""
ML/DL/RL Signal Enhancement Layer

Provides three model types for enhancing trading strategy signals:
  1. XGBSignalFilter  — XGBoost gradient-boosted classifier
  2. LSTMSignalFilter — PyTorch LSTM sequence classifier
  3. PPOTradingAgent  — Stable-Baselines3 PPO reinforcement learning agent

Usage:
    from utils.ml_signals import run_ml_comparison
    run_ml_comparison(df, entries, exits, ticker, freq='D')
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# SHARED FEATURE ENGINEERING
# ============================================================================

def build_features(df, lookback=20):
    """
    Build ML feature matrix from OHLCV data.

    Features:
        - RSI(14), MACD histogram, ATR ratio, Bollinger Band %B
        - 1/5/10-bar momentum
        - Rolling volatility (20-bar std of returns)
        - Relative volume (volume / 20-bar avg)
        - High-low range ratio
    """
    feat = pd.DataFrame(index=df.index)
    close = df['Close'] if 'Close' in df.columns else df['Adj Close']

    # Price returns at multiple horizons
    feat['ret_1'] = close.pct_change(1)
    feat['ret_5'] = close.pct_change(5)
    feat['ret_10'] = close.pct_change(10)

    # RSI(14)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    feat['rsi'] = 100 - 100 / (1 + rs)

    # MACD histogram
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    feat['macd_hist'] = macd - signal

    # ATR ratio (ATR / close)
    if 'ATR' in df.columns:
        feat['atr_ratio'] = df['ATR'] / close
    else:
        high_low = df['High'] - df['Low']
        feat['atr_ratio'] = high_low.rolling(14).mean() / close

    # Bollinger Band %B
    sma20 = close.rolling(lookback).mean()
    std20 = close.rolling(lookback).std()
    feat['bb_pctb'] = (close - (sma20 - 2 * std20)) / (4 * std20).replace(0, np.nan)

    # Volatility (rolling 20-bar std of returns)
    feat['volatility'] = feat['ret_1'].rolling(lookback).std()

    # Relative volume
    if 'Volume' in df.columns:
        avg_vol = df['Volume'].rolling(lookback).mean()
        feat['rel_volume'] = df['Volume'] / avg_vol.replace(0, np.nan)
    else:
        feat['rel_volume'] = 1.0

    # High-low range ratio
    if 'High' in df.columns and 'Low' in df.columns:
        feat['hl_range'] = (df['High'] - df['Low']) / close

    feat.replace([np.inf, -np.inf], np.nan, inplace=True)
    feat.fillna(0, inplace=True)
    return feat


def build_labels(df, forward_bars=10, threshold=0.005):
    """
    Build binary labels: 1 if forward return > threshold, 0 otherwise.
    Used to train classifiers on "is this a good entry point?"
    """
    close = df['Close'] if 'Close' in df.columns else df['Adj Close']
    fwd_ret = close.shift(-forward_bars) / close - 1
    labels = (fwd_ret > threshold).astype(int)
    labels.iloc[-forward_bars:] = 0  # Can't know future for last N bars
    return labels


# ============================================================================
# 1. XGBoost Signal Filter (ML)
# ============================================================================

class XGBSignalFilter:
    """Gradient-boosted tree classifier for signal filtering."""

    def __init__(self, **kwargs):
        import xgboost as xgb
        self.model = xgb.XGBClassifier(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', 4),
            learning_rate=kwargs.get('learning_rate', 0.1),
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',
            verbosity=0,
            use_label_encoder=False,
        )
        self.fitted = False

    def fit(self, features, labels):
        """Train on feature matrix and binary labels."""
        mask = ~(features.isna().any(axis=1) | labels.isna())
        X, y = features[mask].values, labels[mask].values
        if len(np.unique(y)) < 2:
            self.fitted = False
            return self
        self.model.fit(X, y)
        self.fitted = True
        return self

    def predict_proba(self, features):
        """Return probability of class 1 (good entry)."""
        if not self.fitted:
            return pd.Series(0.5, index=features.index)
        proba = self.model.predict_proba(features.fillna(0).values)
        return pd.Series(proba[:, 1], index=features.index)


# ============================================================================
# 2. LSTM Signal Filter (DL)
# ============================================================================

class LSTMSignalFilter:
    """PyTorch LSTM sequence classifier for signal filtering."""

    def __init__(self, seq_len=10, hidden_size=16, epochs=10, lr=0.003, batch_size=64):
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.model = None
        self.fitted = False

    def _build_sequences(self, features, labels=None):
        """Convert feature matrix into overlapping sequences."""
        import torch
        X_vals = features.fillna(0).values.astype(np.float32)
        seqs = []
        labs = []
        for i in range(self.seq_len, len(X_vals)):
            seqs.append(X_vals[i - self.seq_len:i])
            if labels is not None:
                labs.append(labels.iloc[i])

        X = torch.FloatTensor(np.array(seqs))
        if labels is not None:
            y = torch.FloatTensor(np.array(labs, dtype=np.float32))
            return X, y
        return X

    def fit(self, features, labels):
        """Train LSTM on sequential features with mini-batch training."""
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        if len(features) < self.seq_len + 10:
            self.fitted = False
            return self

        X, y = self._build_sequences(features, labels)
        if len(torch.unique(y)) < 2:
            self.fitted = False
            return self

        n_features = X.shape[2]

        class LSTMNet(nn.Module):
            def __init__(self, input_size, hidden_size):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
                self.fc = nn.Linear(hidden_size, 1)

            def forward(self, x):
                _, (h, _) = self.lstm(x)
                return torch.sigmoid(self.fc(h[-1]))

        self.model = LSTMNet(n_features, self.hidden_size)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCELoss()

        # Mini-batch DataLoader
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        best_loss = float('inf')
        patience_counter = 0
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                pred = self.model(X_batch).squeeze()
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            # Early stopping
            if epoch_loss < best_loss * 0.999:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 3:
                    break

        self.fitted = True
        return self

    def predict_proba(self, features):
        """Return probability of class 1 from LSTM."""
        import torch
        if not self.fitted or self.model is None:
            return pd.Series(0.5, index=features.index)

        X = self._build_sequences(features)
        self.model.eval()
        with torch.no_grad():
            proba = self.model(X).squeeze().numpy()

        # Pad the first seq_len entries with 0.5
        full = np.full(len(features), 0.5)
        full[self.seq_len:] = proba
        return pd.Series(full, index=features.index)


# ============================================================================
# 3. PPO Trading Agent (RL)
# ============================================================================

class PPOTradingAgent:
    """
    Proximal Policy Optimization agent for learning entry/exit timing.
    Uses a custom Gymnasium environment wrapping the price data.
    """

    def __init__(self, total_timesteps=10_000):
        self.total_timesteps = total_timesteps
        self.model = None
        self.fitted = False

    def fit(self, features, labels, close_prices=None):
        """Train PPO agent in a trading environment."""
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logging
        os.environ.setdefault('SB3_TENSORBOARD', '0')

        import gymnasium as gym
        from gymnasium import spaces
        from stable_baselines3 import PPO

        if close_prices is None or len(close_prices) < 50:
            self.fitted = False
            return self

        feat_values = features.fillna(0).values
        close_values = close_prices.values.flatten()

        class TradingEnv(gym.Env):
            """Minimal trading environment for RL."""
            metadata = {'render_modes': []}

            def __init__(self, features, closes):
                super().__init__()
                self.features = features
                self.closes = closes
                self.n = len(features)
                self.observation_space = spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(features.shape[1] + 2,), dtype=np.float32
                )
                self.action_space = spaces.Discrete(3)  # 0=hold, 1=buy, 2=sell
                self.reset()

            def reset(self, seed=None, options=None):
                super().reset(seed=seed)
                self.step_idx = 0
                self.position = 0  # -1, 0, 1
                self.entry_price = 0.0
                self.total_pnl = 0.0
                return self._obs(), {}

            def _obs(self):
                feat = self.features[min(self.step_idx, self.n - 1)]
                return np.concatenate([feat, [self.position, self.total_pnl]]).astype(np.float32)

            def step(self, action):
                price = self.closes[self.step_idx]
                reward = 0.0

                if action == 1 and self.position <= 0:  # Buy
                    if self.position == -1:
                        reward = self.entry_price - price  # Close short
                    self.position = 1
                    self.entry_price = price
                elif action == 2 and self.position >= 0:  # Sell
                    if self.position == 1:
                        reward = price - self.entry_price  # Close long
                    self.position = -1
                    self.entry_price = price

                self.total_pnl += reward
                self.step_idx += 1
                done = self.step_idx >= self.n - 1

                return self._obs(), reward, done, False, {}

        env = TradingEnv(feat_values, close_values)

        # n_steps must be divisible by batch_size, and <= episode length
        ep_len = len(feat_values) - 1
        n_steps = min(128, max(32, ep_len // 2))
        # Round down to nearest multiple of 32
        n_steps = (n_steps // 32) * 32
        if n_steps < 32:
            n_steps = 32

        self.model = PPO(
            'MlpPolicy', env, verbose=0,
            n_steps=n_steps,
            batch_size=min(32, n_steps),
            n_epochs=3,
            learning_rate=3e-4,
            tensorboard_log=None,
        )

        try:
            self.model.learn(total_timesteps=self.total_timesteps)
            self.fitted = True
        except Exception:
            self.fitted = False

        self._env_class = TradingEnv
        self._feat_shape = feat_values.shape[1]
        return self

    def predict_proba(self, features, close_prices=None):
        """Use trained PPO to generate trading actions → probability-like scores."""
        if not self.fitted or self.model is None:
            return pd.Series(0.5, index=features.index)

        feat_values = features.fillna(0).values
        scores = np.full(len(features), 0.5)
        position = 0
        total_pnl = 0.0

        for i in range(len(feat_values)):
            obs = np.concatenate([feat_values[i], [position, total_pnl]]).astype(np.float32)
            action, _ = self.model.predict(obs, deterministic=True)
            if action == 1:
                scores[i] = 0.8  # Buy signal
            elif action == 2:
                scores[i] = 0.2  # Sell signal
            else:
                scores[i] = 0.5  # Hold

            # Update position tracking
            if action == 1:
                position = 1
            elif action == 2:
                position = -1

        return pd.Series(scores, index=features.index)


# ============================================================================
# ML COMPARISON RUNNER
# ============================================================================

def run_ml_comparison(df, raw_entries, raw_exits, ticker, freq='D',
                      confidence_threshold=0.6, train_ratio=0.7):
    """
    Run all 3 ML models, filter signals by confidence, and compare
    against raw signals using VBTBacktester.

    Args:
        df: DataFrame with OHLCV + indicator data.
        raw_entries: pd.Series of boolean entry signals.
        raw_exits: pd.Series of boolean exit signals.
        ticker: Ticker symbol for display.
        freq: Data frequency.
        confidence_threshold: Minimum ML confidence to keep a signal.
        train_ratio: Fraction of data used for training.
    """
    from utils.backtesting import VBTBacktester

    print(f"\n{'=' * 60}")
    print(f"  🤖 ML/DL/RL Signal Enhancement: {ticker}")
    print(f"{'=' * 60}")

    close = df['Close'] if 'Close' in df.columns else df['Adj Close']
    features = build_features(df)
    labels = build_labels(df)

    split = int(len(df) * train_ratio)
    train_feat, test_feat = features.iloc[:split], features.iloc[split:]
    train_labels = labels.iloc[:split]
    test_entries = raw_entries.iloc[split:]
    test_exits = raw_exits.iloc[split:]
    test_close = close.iloc[split:]

    if len(test_close) < 20:
        print("  ⚠️ Insufficient test data for ML comparison")
        return

    n_test_signals = int(test_entries.sum())
    if n_test_signals < 2:
        print(f"  ⚠️ Only {n_test_signals} entry signals in test set — skipping ML (need ≥2)")
        return

    # Raw baseline
    print("\n  --- Raw Signals (no ML filter) ---")
    try:
        bt_raw = VBTBacktester(test_close, test_entries, test_exits,
                               freq=freq, init_cash=100_000)
        raw_result = bt_raw.run(print_stats=False)
        print(f"  Return: {raw_result['total_return'] * 100:.2f}%  "
              f"Sharpe: {raw_result['sharpe']:.4f}  "
              f"Max DD: {raw_result['max_drawdown'] * 100:.2f}%")
    except Exception as e:
        print(f"  ❌ Raw baseline error: {e}")
        return

    models = {}
    try:
        models['🌲 XGBoost'] = XGBSignalFilter()
    except ImportError:
        print("  ⚠️ XGBoost not installed. Skipping XGB ML filter.")
    
    try:
        models['🧠 LSTM'] = LSTMSignalFilter(seq_len=min(10, split // 10), epochs=10)
    except ImportError:
        print("  ⚠️ PyTorch not installed. Skipping LSTM DL filter.")
        
    try:
        models['🎮 PPO (RL)'] = PPOTradingAgent(total_timesteps=min(3000, split))
    except ImportError:
        print("  ⚠️ Stable-Baselines3/Gymnasium not installed. Skipping RL agent.")

    if not models:
        print("  ⚠️ No ML libraries installed. Proceeding with raw signals only.")
        return

    for name, model in models.items():
        print(f"\n  --- {name} ---")
        try:
            # Train
            if isinstance(model, PPOTradingAgent):
                model.fit(train_feat, train_labels,
                          close_prices=close.iloc[:split])
            else:
                model.fit(train_feat, train_labels)

            if not model.fitted:
                print(f"    ⚠️ Model could not be fitted (insufficient data/classes)")
                continue

            # Predict confidence on test set
            if isinstance(model, PPOTradingAgent):
                proba = model.predict_proba(test_feat, close_prices=test_close)
            else:
                proba = model.predict_proba(test_feat)

            # Filter signals by confidence
            enhanced_entries = test_entries & (proba >= confidence_threshold)
            enhanced_exits = test_exits | (proba <= (1 - confidence_threshold))

            n_raw = test_entries.sum()
            n_enhanced = enhanced_entries.sum()
            print(f"    Signals: {n_raw} raw → {n_enhanced} filtered "
                  f"(kept {n_enhanced / max(1, n_raw) * 100:.0f}%)")

            if enhanced_entries.sum() == 0:
                print(f"    ⚠️ All signals filtered out — lowering threshold")
                enhanced_entries = test_entries & (proba >= 0.5)
                enhanced_exits = test_exits

            bt_ml = VBTBacktester(test_close, enhanced_entries, enhanced_exits,
                                  freq=freq, init_cash=100_000)
            ml_result = bt_ml.run(print_stats=False)

            ret_diff = ml_result['total_return'] - raw_result['total_return']
            sharpe_diff = ml_result['sharpe'] - raw_result['sharpe']
            emoji = "📈" if ret_diff > 0 else "📉"

            print(f"    Return: {ml_result['total_return'] * 100:.2f}%  "
                  f"Sharpe: {ml_result['sharpe']:.4f}  "
                  f"Max DD: {ml_result['max_drawdown'] * 100:.2f}%")
            print(f"    {emoji} Δ Return: {ret_diff * 100:+.2f}%  "
                  f"Δ Sharpe: {sharpe_diff:+.4f}")

        except Exception as e:
            print(f"    ❌ Error: {e}")
