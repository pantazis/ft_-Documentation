# user_data/documentation/backtestWinnersVsLosers.py
# -*- coding: utf-8 -*-
"""
Backtest Winners vs Losers — Single-file HTML report with embedded data and charts.

What it does (summary):
0) Verifies environment & pointers
1) Locates latest backtest using .last_result.json (falls back to newest folder/zip)
2) Unpacks results (if zipped) and loads:
   - backtest-result-YYYY-MM-DD_HH-MM-SS.json           (trades & meta)
   - backtest-result-YYYY-MM-DD_HH-MM-SS_market_change.feather (market baseline)
3) (Optional) Rebuilds entry-candle features from local candles if present
4) Discovers commonalities — Winners FIRST, then Losers
5) Emits ONE offline HTML (self-contained): tables + charts (Chart.js if available locally)

Paths (Windows-friendly; also works on Linux/Mac):
- Latest pointer: user_data/backtest_results/.last_result.json
- Results root:   user_data/backtest_results/
- Output HTML:    user_data/plot/charts_last/Backtest_WinnersVsLosers.html
- Optional candles: user_data/data/<exchange>/<quote>/<timeframe>/*.json.gz

Notes:
- No internet access required. If a local Chart.js file exists, it is embedded inline.
  Otherwise a tiny canvas fallback draws basic charts so the report still renders offline.
- Minimum-per-bin size enforced (n < 30 -> cell shown as null / greyed).
- Robustness checks (by pair, ATR regime, month) to choose Top 5 patterns per side.

Tested with backtests produced by Freqtrade 2024–2025 formats.
"""

import os
import io
import re
import gzip
import json
import math
import glob
import shutil
import zipfile
import textwrap
import importlib
import itertools
from datetime import datetime, timezone
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

# Try to enable Feather for baseline; if not present, we continue without it.
try:
    import pyarrow  # noqa: F401
    _HAVE_FEATHER = True
except Exception:
    _HAVE_FEATHER = False

# Optional: stats tests (KS). If unavailable, we skip p-values.
try:
    from scipy import stats as _scistats  # noqa: F401
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

# Optional: ML models for deeper analysis
try:
    _sklearn_linear = importlib.import_module("sklearn.linear_model")
    _sklearn_tree = importlib.import_module("sklearn.tree")
    _sklearn_preproc = importlib.import_module("sklearn.preprocessing")
    _sklearn_model_selection = importlib.import_module("sklearn.model_selection")
    LogisticRegression = getattr(_sklearn_linear, "LogisticRegression")
    DecisionTreeClassifier = getattr(_sklearn_tree, "DecisionTreeClassifier")
    StandardScaler = getattr(_sklearn_preproc, "StandardScaler")
    train_test_split = getattr(_sklearn_model_selection, "train_test_split")
    _HAVE_SKLEARN = True
except Exception:
    _HAVE_SKLEARN = False


# -----------------------------
# Configurable constants
# -----------------------------
RESULTS_ROOT = os.path.join("user_data", "backtest_results")
LAST_POINTER = os.path.join(RESULTS_ROOT, ".last_result.json")
OUTPUT_DIR   = os.path.join("user_data", "plot", "charts_last")
OUTPUT_HTML  = os.path.join(OUTPUT_DIR, "Backtest_WinnersVsLosers.html")
OUTPUT_SUMMARY_TXT = os.path.join(OUTPUT_DIR, "Backtest_WinnersVsLosers_summary.txt")

# If present, will be embedded inline into the HTML:
#  - Prefer an already-downloaded local Chart.js bundle (no internet).
# You can place a copy at e.g. user_data/documentation/chart.umd.min.js
LOCAL_CHARTJS_CANDIDATES = [
    os.path.join("user_data", "documentation", "chart.umd.min.js"),
    os.path.join("user_data", "documentation", "Chart.min.js"),
    os.path.join("user_data", "documentation", "chart.min.js"),
]

# Feature parameters
EMA_LEN = 100
ATR_LEN = 14
RSI_LEN = 14
ADX_LEN = 14
DON_LEN = 20
VOL_ZSCORE_WIN = 120

MIN_BIN_SIZE = 30

# -----------------------------
# Utilities
# -----------------------------

def _build_indicator_specs(timeframe: str) -> dict[str, dict[str, object]]:
    """Describe indicator calculations for downstream display."""
    timeframe_str = timeframe or "n/a"
    return {
        "ema_slope": {
            "label": "EMA slope",
            "timeframe": timeframe_str,
            "parameters": {"EMA length": EMA_LEN},
            "description": "Percentage slope of EMA on close prices.",
        },
        "ema_dist": {
            "label": "EMA distance",
            "timeframe": timeframe_str,
            "parameters": {"EMA length": EMA_LEN},
            "description": "Relative distance between close and EMA.",
        },
        "atr_norm": {
            "label": "ATR normalized",
            "timeframe": timeframe_str,
            "parameters": {"ATR length": ATR_LEN},
            "description": "ATR divided by close to express volatility as a fraction.",
        },
        "rsi": {
            "label": "RSI",
            "timeframe": timeframe_str,
            "parameters": {"RSI length": RSI_LEN},
            "description": "Relative Strength Index on close prices.",
        },
        "adx": {
            "label": "ADX",
            "timeframe": timeframe_str,
            "parameters": {"ADX length": ADX_LEN},
            "description": "+DI/-DI derived Average Directional Index.",
        },
        "don_pos": {
            "label": "Donchian position",
            "timeframe": timeframe_str,
            "parameters": {"Donchian length": DON_LEN},
            "description": "Close position within Donchian channel range.",
        },
        "don_width": {
            "label": "Donchian width",
            "timeframe": timeframe_str,
            "parameters": {"Donchian length": DON_LEN},
            "description": "Channel width normalized by close price.",
        },
        "vol_z": {
            "label": "Volume z-score",
            "timeframe": timeframe_str,
            "parameters": {"Rolling window": VOL_ZSCORE_WIN},
            "description": "Volume standardized by rolling mean and std.",
        },
    }

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _write_text(path: str, data: str):
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(data)

def _safe_dt(s):
    if s is None or s == "":
        return None
    try:
        return pd.to_datetime(s, utc=True)
    except Exception:
        try:
            return pd.to_datetime(s)
        except Exception:
            return None

def _coerce_dt_col(df: pd.DataFrame, col: str):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

def _find_latest_backtest():
    # 1) Try pointer file
    if os.path.exists(LAST_POINTER):
        try:
            j = _read_json(LAST_POINTER)
            latest = j.get("latest_backtest", "")
            if latest:
                cand = os.path.join(RESULTS_ROOT, latest)
                if os.path.exists(cand):
                    return cand
        except Exception:
            pass

    # 2) Fallback: newest folder or zip in RESULTS_ROOT
    entries = []
    for p in glob.glob(os.path.join(RESULTS_ROOT, "backtest-result-*")):
        try:
            mtime = os.path.getmtime(p)
            entries.append((mtime, p))
        except Exception:
            continue
    if not entries:
        raise FileNotFoundError("No backtest results found under user_data/backtest_results/")
    entries.sort(reverse=True)
    return entries[0][1]

def _unpack_if_needed(latest_path: str) -> str:
    """
    Returns folder containing the result files.
    Accepts either .../backtest-result-YYYY-mm-dd_HH-MM-SS.zip or a folder path.
    """
    if latest_path.lower().endswith(".zip"):
        folder = latest_path[:-4]
        if os.path.exists(folder) and os.path.isdir(folder):
            return folder
        # unzip
        with zipfile.ZipFile(latest_path, "r") as zf:
            zf.extractall(folder)
        return folder
    # Already a folder
    return latest_path

def _match_result_files(folder: str):
    """
    Find the two files inside the folder:
    - backtest-result-YYYY-mm-dd_HH-MM-SS.json
    - backtest-result-YYYY-mm-dd_HH-MM-SS_market_change.feather
    """
    js = glob.glob(os.path.join(folder, "backtest-result-*_*.json"))
    js = [p for p in js if "_market_change" not in p and not p.endswith("_config.json")]
    js.sort()
    if not js:
        raise FileNotFoundError("Could not find backtest-result-*.json in " + folder)
    base_candidates = [p for p in js if re.search(r"backtest-result-\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.json$", os.path.basename(p))]
    trade_json = base_candidates[0] if base_candidates else js[0]

    feather = glob.glob(os.path.join(folder, "*_market_change.feather"))
    feather.sort()
    market_feather = feather[-1] if feather else None
    return trade_json, market_feather

def _load_trades_json(path: str) -> pd.DataFrame:
    j = _read_json(path)
    # Freqtrade stores trades either under "trades" (legacy flat) or within
    # the per-strategy block introduced in newer releases.
    trades = j.get("trades") or j.get("results") or []
    meta_source = j

    if not trades and isinstance(j.get("strategy"), dict):
        for _, strat_data in j["strategy"].items():
            if not isinstance(strat_data, dict):
                continue
            candidate = strat_data.get("trades") or strat_data.get("results") or []
            if candidate:
                trades = candidate
                meta_source = strat_data
                break
        else:
            # Fall back to the first strategy block even if it has no trades,
            # so downstream metadata (like timeframe) stays available.
            first_val = next(iter(j["strategy"].values()), {})
            if isinstance(first_val, dict):
                meta_source = first_val

    df = pd.DataFrame(trades)
    if df.empty:
        raise ValueError("No trades found in backtest JSON; cannot build Winners vs Losers report.")
    # Common columns
    for col in ("open_date", "close_date"):
        _coerce_dt_col(df, col)
    # profit_ratio may be "profit_ratio" or derive from "profit_abs" / stake_amount
    if "profit_ratio" not in df.columns:
        if {"profit_abs", "stake_amount"} <= set(df.columns):
            with np.errstate(divide="ignore", invalid="ignore"):
                df["profit_ratio"] = np.where(df["stake_amount"] != 0,
                                              df["profit_abs"] / df["stake_amount"],
                                              np.nan)
        elif "profit_percent" in df.columns:
            # Sometimes stored as percentage (e.g., 2.0 -> 0.02)
            df["profit_ratio"] = df["profit_percent"] / 100.0
        else:
            raise ValueError("Cannot find or derive profit_ratio.")
    if "pair" not in df.columns:
        raise ValueError("Missing 'pair' in trades json.")
    # Holding time
    if "close_date" in df.columns and "open_date" in df.columns:
        df["holding_s"] = (df["close_date"] - df["open_date"]).dt.total_seconds()
    else:
        df["holding_s"] = np.nan
    return df, meta_source  # return meta json too

def _load_market_feather(path: str | None) -> pd.DataFrame | None:
    if not path or not _HAVE_FEATHER:
        return None
    try:
        return pd.read_feather(path)
    except Exception:
        return None

# -----------------------------
# Feature engineering (optional, from candles)
# -----------------------------

def _true_range(df):
    # df requires 'high','low','close' columns
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def _rsi(series: pd.Series, length: int) -> pd.Series:
    delta = series.diff()
    up = (delta.clip(lower=0)).ewm(alpha=1/length, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1/length, adjust=False).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _adx(df: pd.DataFrame, length: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute ADX and DI series using Wilder's smoothing with index preserved."""
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    up_move = high.diff()
    down_move = low.diff() * -1

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    tr = _true_range(df)
    atr = tr.ewm(alpha=1 / length, adjust=False, ignore_na=True).mean()

    plus_di = 100 * (plus_dm.ewm(alpha=1 / length, adjust=False, ignore_na=True).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / length, adjust=False, ignore_na=True).mean() / atr)

    denom = (plus_di + minus_di).replace(0, np.nan)
    dx = (100 * (plus_di.subtract(minus_di).abs() / denom)).replace([np.inf, -np.inf], np.nan)
    adx = dx.ewm(alpha=1 / length, adjust=False, ignore_na=True).mean()

    plus_di = plus_di.fillna(0.0).rename("plus_di")
    minus_di = minus_di.fillna(0.0).rename("minus_di")
    adx = adx.fillna(0.0).rename("adx")

    return plus_di, minus_di, adx

def _donchian(df, length=20):
    highest = df["high"].rolling(length).max()
    lowest  = df["low"].rolling(length).min()
    width = (highest - lowest).replace(0, np.nan)
    pos = (df["close"] - lowest) / width
    return highest, lowest, width, pos

def _volume_z(df, win=120):
    v = df.get("volume")
    if v is None:
        return pd.Series(np.nan, index=df.index)
    mean = v.rolling(win).mean()
    std = v.rolling(win).std(ddof=0)
    z = (v - mean) / std.replace(0, np.nan)
    return z

def _pair_to_slug(pair: str) -> str:
    """Normalize pair into slug used in filenames (e.g. BTC/USDC -> BTC_USDC)."""
    if not pair:
        return ""
    base = str(pair).split(":")[0]
    slug = base.replace("/", "_").replace("-", "_")
    slug = re.sub(r"__+", "_", slug).strip("_")
    return slug


def _read_candles_from_path(path: str) -> pd.DataFrame:
    """Read candle data supporting feather, json, json.gz."""
    if path.lower().endswith(".feather"):
        if not _HAVE_FEATHER:
            raise RuntimeError("pyarrow required to read feather files")
        return pd.read_feather(path)
    if path.lower().endswith(".json.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            payload = json.load(fh)
    else:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    if isinstance(payload, dict) and "data" in payload:
        payload = payload["data"]
    df = pd.DataFrame(payload)
    # Raw Binance downloads may store candles as positional arrays.
    positional_cols = {
        6: ["date", "open", "high", "low", "close", "volume"],
        7: ["date", "open", "high", "low", "close", "volume", "quote_volume"],
        8: ["date", "open", "high", "low", "close", "volume", "quote_volume", "ignore"],
    }
    if all(isinstance(col, int) for col in df.columns) and len(df.columns) in positional_cols:
        df.columns = positional_cols[len(df.columns)]
    return df


def _load_all_candle_files(timeframe: str, pairs_needed: list[str]) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Load candles for requested pairs across json/json.gz/feather storage."""
    root = os.path.join("user_data", "data")
    if not os.path.isdir(root):
        return {}, pairs_needed

    pairs: dict[str, pd.DataFrame] = {}
    missing: list[str] = []

    for pair in pairs_needed:
        slug = _pair_to_slug(pair)
        if not slug:
            missing.append(pair)
            continue

        # Prefer JSON sources when pyarrow is unavailable so we still load candles.
        base_patterns = [
            os.path.join(root, "**", f"{slug}-{timeframe}.feather"),
            os.path.join(root, "**", f"{slug}_{timeframe}.feather"),
            os.path.join(root, "**", f"{slug}-{timeframe}.json.gz"),
            os.path.join(root, "**", f"{slug}_{timeframe}.json.gz"),
            os.path.join(root, "**", f"{slug}-{timeframe}.json"),
            os.path.join(root, "**", f"{slug}_{timeframe}.json"),
            os.path.join(root, "**", slug, timeframe, "*.json.gz"),
            os.path.join(root, "**", slug, timeframe, "*.json"),
        ]
        if not _HAVE_FEATHER:
            # JSON first when we can't read feather files.
            json_first = [
                base_patterns[2],
                base_patterns[3],
                base_patterns[4],
                base_patterns[5],
                base_patterns[6],
                base_patterns[7],
                base_patterns[0],
                base_patterns[1],
            ]
            search_patterns = json_first
        else:
            search_patterns = base_patterns

        matches: list[str] = []
        for pattern in search_patterns:
            matches = glob.glob(pattern, recursive=True)
            if matches:
                break
        if not matches:
            missing.append(pair)
            continue

        matches.sort(key=lambda p: os.path.getmtime(p))

        df = None
        while matches:
            path = matches.pop()
            try:
                df = _read_candles_from_path(path)
                break
            except RuntimeError as exc:
                if "pyarrow" in str(exc).lower() and matches:
                    continue
                if matches:
                    continue
                df = None
            except Exception:
                if matches:
                    continue
                df = None

        if df is None or df.empty:
            missing.append(pair)
            continue

        # Normalize schema
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
            df = df.sort_values("date").set_index("date")
        elif "open_time" in df.columns:
            df["open_time"] = pd.to_datetime(df["open_time"], utc=True, errors="coerce")
            df = df.sort_values("open_time").set_index("open_time")
        elif isinstance(df.index, pd.DatetimeIndex):
            df = df.sort_index()
        else:
            missing.append(pair)
            continue

        # Require price columns to compute indicators
        if not {"close", "high", "low"} <= set(df.columns):
            missing.append(pair)
            continue

        df["high"] = df["high"].fillna(df["close"])
        df["low"] = df["low"].fillna(df["close"])

        df["ema"] = _ema(df["close"], EMA_LEN)
        df["ema_slope"] = df["ema"].pct_change()
        df["ema_dist"] = (df["close"] - df["ema"]) / df["ema"]
        tr = _true_range(df)
        df["atr"] = tr.ewm(alpha=1/ATR_LEN, adjust=False).mean()
        df["atr_norm"] = df["atr"] / df["close"]
        df["rsi"] = _rsi(df["close"], RSI_LEN)
        plus_di, minus_di, adx = _adx(df, ADX_LEN)
        df["plus_di"], df["minus_di"], df["adx"] = plus_di, minus_di, adx
        hi, lo, width, pos = _donchian(df, DON_LEN)
        df["don_high"], df["don_low"], df["don_width"], df["don_pos"] = hi, lo, width / df["close"], pos
        df["vol_z"] = _volume_z(df, VOL_ZSCORE_WIN)

        idx = df.index.tz_convert("UTC") if df.index.tz is not None else df.index
        df["hour"] = idx.hour
        df["weekday"] = idx.weekday

        pairs[pair] = df

    return pairs, missing

def _align_trade_features(trades: pd.DataFrame, candles_by_pair: dict[str, pd.DataFrame]) -> pd.DataFrame:
    feats = []
    for idx, row in trades.iterrows():
        pair = row.get("pair")
        open_dt = row.get("open_date")
        if pair not in candles_by_pair or pd.isna(open_dt):
            feats.append({})  # missing
            continue
        cdf = candles_by_pair[pair]
        # pick the candle at or immediately BEFORE open_dt
        try:
            loc = cdf.index.get_loc(open_dt, method="ffill")
        except Exception:
            # If open_dt is before the first candle, skip
            loc = cdf.index.searchsorted(open_dt) - 1
            if loc < 0:
                feats.append({})
                continue
        # row at loc
        feat_row = cdf.iloc[loc]
        out = {
            "ema": float(feat_row.get("ema", np.nan)),
            "ema_slope": float(feat_row.get("ema_slope", np.nan)),
            "ema_dist": float(feat_row.get("ema_dist", np.nan)),
            "atr_norm": float(feat_row.get("atr_norm", np.nan)),
            "rsi": float(feat_row.get("rsi", np.nan)),
            "adx": float(feat_row.get("adx", np.nan)),
            "don_pos": float(feat_row.get("don_pos", np.nan)),
            "don_width": float(feat_row.get("don_width", np.nan)),
            "vol_z": float(feat_row.get("vol_z", np.nan)),
            "hour": int(feat_row.get("hour", np.nan)) if not pd.isna(feat_row.get("hour", np.nan)) else np.nan,
            "weekday": int(feat_row.get("weekday", np.nan)) if not pd.isna(feat_row.get("weekday", np.nan)) else np.nan,
        }
        feats.append(out)
    fdf = pd.DataFrame(feats, index=trades.index)
    return fdf

# -----------------------------
# Analysis
# -----------------------------

def _binned_winrate(series: pd.Series, wins: pd.Series, labels) -> tuple[list, list, list]:
    """
    Given a numeric series and boolean winners mask, compute winrate per bin label.
    labels: Either explicit edges or explicit text labels used with pd.cut.
    We will build 5 quantile bins for continuous features by default.
    """
    # Build numeric bins using quantiles if labels is "quantile_5"
    if labels == "quantile_5":
        q = series.quantile([0, .2, .4, .6, .8, 1.0]).values
        edges = np.unique(q)
        if len(edges) < 2:
            return [], [], []
        cats = pd.cut(series, bins=edges, include_lowest=True, duplicates="drop")
        labs = [str(c) for c in cats.cat.categories]
    elif isinstance(labels, (list, tuple)) and all(isinstance(x, (int, float)) for x in labels):
        # treat as edges
        cats = pd.cut(series, bins=labels, include_lowest=True, duplicates="drop")
        labs = [str(c) for c in cats.cat.categories]
    else:
        # assume labels are category names provided by caller
        cats = series
        labs = sorted(list(pd.Series(labels).astype(str).unique()))

    wr, counts = [], []
    for cat in (cats.cat.categories if hasattr(cats, "cat") else labs):
        mask = (cats == cat) if not hasattr(cats, "cat") else (cats == cat)
        n = int(mask.sum())
        if n < MIN_BIN_SIZE:
            wr.append(None)  # insufficient sample
            counts.append(n)
            continue
        wr.append(float(100.0 * wins[mask].mean()))
        counts.append(n)
    return [str(x) for x in (cats.cat.categories if hasattr(cats, "cat") else labs)], wr, counts

def _distribution(series: pd.Series, wmask: pd.Series, num=40):
    """Simple histogram (equal-width) for winners & losers."""
    data = series.replace([np.inf, -np.inf], np.nan).dropna()
    if data.empty:
        return [], [], []
    hist_all, edges = np.histogram(data, bins=num)
    # winners
    data_w = series[wmask].replace([np.inf, -np.inf], np.nan).dropna()
    data_l = series[~wmask].replace([np.inf, -np.inf], np.nan).dropna()
    hw, _ = np.histogram(data_w, bins=edges)
    hl, _ = np.histogram(data_l, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers.round(6).tolist(), hw.astype(int).tolist(), hl.astype(int).tolist()

def _effect_size(a: pd.Series, b: pd.Series) -> float:
    """Cohen's d (pooled)."""
    a = a.dropna(); b = b.dropna()
    if len(a) < 3 or len(b) < 3:
        return np.nan
    mean_diff = a.mean() - b.mean()
    var = (((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) / (len(a)+len(b)-2))
    if var <= 0 or np.isnan(var):
        return np.nan
    d = mean_diff / math.sqrt(var)
    return float(d)

def _ks_pvalue(a: pd.Series, b: pd.Series) -> float | None:
    if not _HAVE_SCIPY:
        return None
    a = a.dropna()
    b = b.dropna()
    if len(a) < 10 or len(b) < 10:
        return None
    try:
        stat, p = _scistats.ks_2samp(a, b, alternative='two-sided', method='auto')
        return float(p)
    except Exception:
        return None

def _winrate_by(series: pd.Series, wins: pd.Series, labels_sorted) -> list[dict]:
    out = []
    for lab in labels_sorted:
        mask = (series == lab)
        n = int(mask.sum())
        wr = float(100.0 * wins[mask].mean()) if n >= MIN_BIN_SIZE else None
        out.append({"bucket": lab, "wr_pct": wr, "n": n})
    return out

def _to_month(ts: pd.Series) -> pd.Series:
    dates = pd.to_datetime(ts, utc=True, errors="coerce")
    dates = dates.dt.tz_convert(None)
    return dates.dt.to_period("M").astype(str)

def _top_patterns(df: pd.DataFrame, wins_mask: pd.Series, baseline_wr: float, want="winners") -> list[dict]:
    """
    Heuristic mining of Top 5 patterns:
    - Evaluate bins on key features
    - Score by lift * support, with minimal n and replication across pair/month/ATR regime
    """
    patterns = []

    # Candidate binnings
    cand_defs = [
        ("ema_slope",  "quantile_5"),
        ("don_pos",    [0.0, .2, .4, .6, .8, 1.0]),
        ("atr_norm",   "quantile_5"),
        ("rsi",        [0,30,40,50,60,70,100]),
        ("adx",        [0,10,20,30,40,60,100]),
    ]
    # Auxiliary strata
    by_month = _to_month(df["open_date"]) if "open_date" in df else pd.Series(index=df.index, dtype=str)
    by_pair  = df["pair"].astype(str)
    if "atr_norm" in df and df["atr_norm"].notna().sum() >= MIN_BIN_SIZE:
        atr_reg = pd.qcut(df["atr_norm"], q=3, duplicates="drop", labels=["Low","Medium","High"])
    else:
        atr_reg = pd.Series(["?" for _ in df.index])

    for feat, binspec in cand_defs:
        if feat not in df.columns:
            continue
        labels, wr, counts = _binned_winrate(df[feat], wins_mask, binspec)
        for lab, wrv, n in zip(labels, wr, counts):
            if wrv is None or n < MIN_BIN_SIZE:
                continue
            lift = wrv - baseline_wr
            # For losers list, invert lift meaning (we want high loss-rate lift)
            score = lift * n if want == "winners" else (-lift) * n

            # Replication checks
            mask = pd.cut(df[feat], bins=labels) if binspec != "quantile_5" and isinstance(labels, list) and "[" in labels[0] else (df[feat].astype(str) == lab)
            # If labels are Interval, rebuild mask robustly:
            if isinstance(labels[0], str) and labels[0].startswith("(") or labels[0].startswith("["):
                # parse interval
                try:
                    interval = pd.Interval(left=float(lab.split(",")[0][1:]),
                                           right=float(lab.split(",")[1][:-1]),
                                           closed="right" if lab.startswith("(") else "both")
                    mask = df[feat].between(interval.left, interval.right, inclusive="both" if interval.closed == "both" else "right")
                except Exception:
                    mask = (df[feat].astype(str) == lab)

            # Stability across strata:
            # - by pair
            pairs_wr = []
            for p, sub in df[mask].groupby(by_pair[mask], observed=False):
                if len(sub) >= MIN_BIN_SIZE:
                    pairs_wr.append(sub["profit_ratio"].gt(0).mean())
            # - by month
            months_wr = []
            for m, sub in df[mask].groupby(by_month[mask], observed=False):
                if len(sub) >= MIN_BIN_SIZE:
                    months_wr.append(sub["profit_ratio"].gt(0).mean())
            # - by atr regime
            atr_wr = []
            if "atr_norm" in df:
                for a, sub in df[mask].groupby(atr_reg[mask], observed=False):
                    if len(sub) >= MIN_BIN_SIZE:
                        atr_wr.append(sub["profit_ratio"].gt(0).mean())

            rep_count = sum([len(pairs_wr) >= 2, len(months_wr) >= 2, len(atr_wr) >= 2])
            # require replication in at least 2 strata to be considered "stable"
            if rep_count >= 2:
                patterns.append({
                    "rule": f"{feat} in {lab}",
                    "n": int(n),
                    "winrate_pct": float(wrv) if want == "winners" else None,
                    "lossrate_pct": float(100.0 - wrv) if want == "losers" else None,
                    "lift_pp": float(lift) if want == "winners" else float((100.0 - wrv) - (100.0 - baseline_wr)),
                    "notes": f"Stable across {rep_count}/3 strata; pairs={len(pairs_wr)} months={len(months_wr)} atr={len(atr_wr)}",
                    "_score": float(score),
                })

    # Sort and take top 5
    patterns.sort(key=lambda x: x["_score"], reverse=True)
    # Clean internal score
    for p in patterns:
        p.pop("_score", None)
    return patterns[:5]

# -----------------------------
# HTML rendering
# -----------------------------

def _read_local_chartjs() -> str | None:
    for cand in LOCAL_CHARTJS_CANDIDATES:
        if os.path.exists(cand):
            try:
                with open(cand, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception:
                continue
    return None

def _build_html(data_obj: dict) -> str:
    # Try to embed Chart.js locally (inline). If not found, add minimal no-lib fallback.
    chartjs_code = _read_local_chartjs()
    has_chartjs = chartjs_code is not None

    # Precompute template inserts to avoid inline expressions that upset older Python f-strings
    chart_locations_html = "<br/>".join(LOCAL_CHARTJS_CANDIDATES)
    if has_chartjs:
        chartjs_tag = "<script>\n// Chart.js embedded inline\n" + chartjs_code + "\n</script>"
        chart_note = ""
    else:
        cdn_url = "https://cdn.jsdelivr.net/npm/chart.js"
        chartjs_tag = f"<script src=\"{cdn_url}\"></script>"
        chart_note = (
            "<div id=\"no-chartjs-note\" class=\"warn\" style=\"display:none\">"
            "Attempted to load Chart.js from the public CDN at "
            f"<code>{cdn_url}</code> but it was unavailable."
            " The report reverted to the minimal built-in plotter. "
            "Place a local copy at:<br/><code>"
            f"{chart_locations_html}</code><br/>to embed it directly.</div>"
        )
    generated_on = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Basic CSS + layout
    css = """
    body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Inter,Arial,sans-serif;margin:20px;color:#222;background:#fff}
    h1,h2,h3{margin:0 0 .5rem}
    .muted{color:#666}
    .grid{display:grid;gap:18px}
    .cols-2{grid-template-columns:1fr 1fr}
    .card{border:1px solid #e5e7eb;border-radius:12px;padding:16px;box-shadow:0 1px 2px rgba(0,0,0,.03)}
    table{border-collapse:collapse;width:100%}
    th,td{border:1px solid #eee;padding:8px;text-align:left;font-size:14px}
    th{background:#f9fafb}
    .badge{display:inline-block;padding:2px 8px;border-radius:999px;background:#eef2ff;color:#3730a3;font-size:12px}
    canvas{max-width:100%;height:360px}
    .warn{color:#b45309;background:#fff7ed;border:1px solid #fde68a;padding:8px;border-radius:8px;margin:8px 0}
    .good{color:#065f46;background:#ecfdf5;border:1px solid #a7f3d0;padding:8px;border-radius:8px;margin:8px 0}
    """

    # Embed the data
    data_json = json.dumps(data_obj, ensure_ascii=False)

    # Minimal fallback plotter (draws lines/bars on canvas) if Chart.js missing
    fallback_js = """
    // Tiny no-lib plotter used only if Chart.js isn't embedded.
    function fallbackLineChart(canvasId, xs, ys, label, xLabel, yLabel){
        const c = document.getElementById(canvasId).getContext('2d');
        const W=c.canvas.width, H=c.canvas.height;
        c.clearRect(0,0,W,H);
        c.strokeStyle='#555'; c.lineWidth=1;
        c.strokeRect(0,0,W,H);
        const n=xs.length;
        if(n===0){ c.fillText('No data',10,20); return; }
        const minY = Math.min(...ys.filter(v=>v!=null)), maxY = Math.max(...ys.filter(v=>v!=null));
        const pad=30;
        c.strokeStyle='#999';
        c.beginPath(); c.moveTo(pad,H-pad); c.lineTo(W-pad,H-pad); c.stroke();
        c.beginPath(); c.moveTo(pad,pad); c.lineTo(pad,H-pad); c.stroke();
        c.strokeStyle='#3b82f6'; c.lineWidth=2; c.beginPath();
        for(let i=0;i<n;i++){
            if(ys[i]==null) continue;
            const x = pad + (W-2*pad)*(i/(n-1));
            const y = pad + (H-2*pad)*(1 - (ys[i]-minY)/(maxY-minY+1e-9));
            if(i===0) c.moveTo(x,y); else c.lineTo(x,y);
        }
        c.stroke();
        c.fillStyle='#111'; c.fillText(label||'', pad+6, pad+14);
        if(xLabel){
            c.fillText(xLabel, W/2 - c.measureText(xLabel).width/2, H - 6);
        }
        if(yLabel){
            c.save();
            c.translate(12, H/2 + c.measureText(yLabel).width/2);
            c.rotate(-Math.PI/2);
            c.fillText(yLabel, 0, 0);
            c.restore();
        }
    }
    function fallbackBarChart(canvasId, labels, values, label, xLabel, yLabel){
        const c = document.getElementById(canvasId).getContext('2d');
        const W=c.canvas.width, H=c.canvas.height;
        c.clearRect(0,0,W,H);
        c.strokeStyle='#555'; c.lineWidth=1;
        c.strokeRect(0,0,W,H);
        const pad=30, n=labels.length;
        const maxV = Math.max(...values.filter(v=>v!=null), 0);
        const bw = (W-2*pad)/Math.max(n,1);
        c.fillStyle='#3b82f6';
        for(let i=0;i<n;i++){
            const v = values[i]==null?0:values[i];
            const x = pad + i*bw + 2, y = pad + (H-2*pad)*(1 - (v/(maxV||1)));
            const h = (H-2*pad) - (H-2*pad)*(1 - (v/(maxV||1)));
            c.fillRect(x, y, Math.max(2, bw-4), h);
        }
        c.fillStyle='#111'; c.fillText(label||'', pad+6, pad+14);
        if(xLabel){
            c.fillText(xLabel, W/2 - c.measureText(xLabel).width/2, H - 6);
        }
        if(yLabel){
            c.save();
            c.translate(12, H/2 + c.measureText(yLabel).width/2);
            c.rotate(-Math.PI/2);
            c.fillText(yLabel, 0, 0);
            c.restore();
        }
    }
    """

    # Chart init JS
    chart_init = """
    const DATA = __DATA_JSON__;
    const hasChartJs = !!window.Chart;

    const HEADER_LABELS = {
        feature: "Feature",
        group: "Group",
        mean: "Mean",
        median: "Median",
        std: "Std Dev",
        effect: "Effect Size",
        n: "Trades",
        rule: "Pattern Rule",
        winrate_pct: "Win Rate %",
        lossrate_pct: "Loss Rate %",
        lift_pp: "Lift (pp)",
        notes: "Notes",
        bucket: "Bucket",
        wr_pct: "Win Rate %",
        counts: "Count",
        hour: "Hour",
        weekday: "Weekday",
        indicator: "Indicator",
        min: "Min",
        max: "Max",
        profit: "Δ Profit",
        wins_kept_pct: "Wins Kept",
        losses_cut_pct: "Losses Cut",
        wins_lost: "Wins Lost",
        losses_cut_trades: "Losses Cut (trades)",
        timeframe: "Timeframe",
        parameters: "Parameters",
        description: "Description",
        corr_profit: "Corr vs Profit",
        corr_abs: "Abs Corr",
        metric: "Metric",
        value: "Value",
        coef: "Coefficient",
        odds_ratio: "Odds Ratio",
        importance: "Importance",
        accuracy: "Accuracy",
        train_size: "Train Samples",
        test_size: "Test Samples",
        n_samples: "Samples",
        depth: "Depth",
        n_leaves: "Leaves",
        intercept: "Intercept",
        features: "Features",
        lift: "Lift",
        baseline: "Baseline",
        samples: "Samples",
        coeffs: "Coefficients"
    };

    function tableFromArray(rows, columns, id){
        const el = document.getElementById(id);
        const thead = `<thead><tr>${columns.map(c=>`<th>${HEADER_LABELS[c] || c}</th>`).join('')}</tr></thead>`;
        const tbody = `<tbody>${rows.map(r=>`<tr>${columns.map(c=>`<td>${(r[c]===null||r[c]===undefined)?'':r[c]}</td>`).join('')}</tr>`).join('')}</tbody>`;
        el.innerHTML = `<table>${thead}${tbody}</table>`;
    }

    function formatBound(value){
        if(!Number.isFinite(value)) return "n/a";
        const absVal = Math.abs(value);
        if(absVal >= 1) return value.toFixed(3);
        if(absVal >= 0.01) return value.toFixed(4);
        return value.toExponential(2);
    }

    function featureSummaryTable(){
        const cols = ["feature","group","mean","median","std","effect","n"];
        tableFromArray(DATA.feature_summary, cols, "feature-summary");
    }

    function strategySettings(){
        const settings = DATA.meta.strategy_settings || {};
        const container = document.getElementById("strategy-settings");
        const entries = Object.entries(settings);
        if (!container) return;
        if(!entries.length){
            container.innerHTML = '<div class="muted">No settings provided.</div>';
            return;
        }
        const rows = entries.map(([key, value]) => ({ key, value }));
        const thead = "<thead><tr><th>Setting</th><th>Value</th></tr></thead>";
        const tbody = `<tbody>${rows.map(r=>`<tr><td>${r.key}</td><td>${r.value}</td></tr>`).join('')}</tbody>`;
        container.innerHTML = `<table>${thead}${tbody}</table>`;
    }

    function patternTables(){
        const wcols = ["rule","n","winrate_pct","lift_pp","notes"];
        const lcols = ["rule","n","lossrate_pct","lift_pp","notes"];
        tableFromArray(DATA.winners_patterns, wcols, "winners-patterns");
        tableFromArray(DATA.losers_patterns,  lcols, "losers-patterns");
    }

    function metaSummary(){
        const m = DATA.meta;
                const summary =
                    `<div class="good"><b>Total trades:</b> ${m.total_trades} &nbsp; ` +
                    `<b>Winners:</b> ${m.winners} &nbsp; <b>Losers:</b> ${m.losers} &nbsp; ` +
                    `<b>Baseline Win-rate:</b> ${m.baseline_winrate_pct.toFixed(2)}% &nbsp; ` +
                    `<span class="muted">Date span:</span> ${m.date_span}</div>`;
                let warning = "";
                if (m.missing_indicator_pairs && m.missing_indicator_pairs.length){
                        const preview = m.missing_indicator_pairs.slice(0, 10).join(", ");
                        const suffix = m.missing_indicator_pairs.length > 10 ? "…" : "";
                        warning = `<div class="warn">Missing entry-candle indicators for ${m.missing_indicator_pairs.length} ` +
                                            `pair(s): ${preview}${suffix}</div>`;
                }
        let stratBlock = "";
        if (m.strategy_settings){
            const s = m.strategy_settings;
            const label = (key) => (key in s && s[key] !== null && s[key] !== "") ? s[key] : "n/a";
            stratBlock = `<div class="card" style="margin-top:12px;">` +
                         `<div><b>Strategy:</b> ${label('strategy_name')}</div>` +
                         `<div><b>Timeframe:</b> ${label('timeframe')}</div>` +
                         `<div><b>Stake Currency:</b> ${label('stake_currency')}</div>` +
                         `<div><b>Stake Amount:</b> ${label('stake_amount')}</div>` +
                         `<div><b>Stoploss:</b> ${label('stoploss')}</div>` +
                         `<div><b>Trailing Stop:</b> ${label('trailing_stop')}</div>` +
                         `</div>`;
        }
        document.getElementById("meta").innerHTML = summary + warning + stratBlock;
    }

    function timeTables(){
        const hours = DATA.time_winrate.hour;
        const wdays = DATA.time_winrate.weekday;
        tableFromArray(hours, ["bucket","wr_pct","n"], "time-hour");
        tableFromArray(wdays, ["bucket","wr_pct","n"], "time-weekday");
    }

    function indicatorBaseTitle(indicator, fallbackLabel, suffix){
        const specs = DATA.indicator_specs || {};
        const spec = specs[indicator] || null;
        const label = (spec && spec.label) ? spec.label : fallbackLabel;
        const parts = [];
        if (spec && spec.parameters){
            const paramEntries = Object.entries(spec.parameters).filter(([_, v]) => v !== null && v !== undefined);
            if (paramEntries.length){
                const formatted = paramEntries.map(([k, v]) => `${k}: ${v}`).join(' · ');
                parts.push(formatted);
            }
        }
        if (spec && spec.timeframe){
            parts.push(`TF ${spec.timeframe}`);
        }
        const suffixText = suffix ? ` — ${suffix}` : '';
        const meta = parts.length ? ` (${parts.join(' | ')})` : '';
        return `${label}${meta}${suffixText}`;
    }

    function thresholdTitle(indicator, baseTitle){
        const thresholds = DATA.indicator_thresholds || {};
        const th = thresholds[indicator];
        if(!th) return baseTitle;
        const profitDelta = Number.isFinite(th.profit_change) ? th.profit_change.toFixed(4) : "0.0000";
        const winKeepPct = Number.isFinite(th.win_keep_pct) ? Math.round(th.win_keep_pct * 100) : null;
        const lossCutPct = Number.isFinite(th.loss_cut_pct) ? Math.round(th.loss_cut_pct * 100) : null;
        const parts = [
            `Min ${formatBound(th.min)}`,
            `Max ${formatBound(th.max)}`,
            `ΔProfit ${profitDelta}`,
            lossCutPct !== null ? `Losses cut ${lossCutPct}% (${th.losses_cut})` : null,
            winKeepPct !== null ? `Wins kept ${winKeepPct}% (lost ${th.wins_lost})` : null,
        ].filter(Boolean);
        if(!parts.length){
            return baseTitle;
        }
        return `${baseTitle} | ${parts.join(' · ')}`;
    }

    function indicatorTitle(indicator, fallbackLabel, suffix){
        const base = indicatorBaseTitle(indicator, fallbackLabel, suffix);
        return thresholdTitle(indicator, base);
    }

    function setIndicatorHeading(elementId, indicator, suffix){
        const el = document.getElementById(elementId);
        if(!el) return;
        const baseLabel = el.dataset && el.dataset.label ? el.dataset.label : indicator;
        el.textContent = indicatorBaseTitle(indicator, baseLabel, suffix);
    }

    function indicatorThresholdTable(){
        const container = document.getElementById("indicator-thresholds");
        if(!container) return;
        const thresholds = DATA.indicator_thresholds || {};
        const entries = Object.entries(thresholds);
        if(!entries.length){
            container.innerHTML = '<div class="muted">No indicator filters calculated.</div>';
            return;
        }
        const specsLookup = DATA.indicator_specs || {};
        const rows = entries.map(([indicator, stats]) => {
            const profitValue = Number(stats.profit_change) || 0;
            const spec = specsLookup[indicator] || {};
            const indicatorLabel = indicatorBaseTitle(indicator, spec.label || indicator, "");
            return {
                indicator: indicatorLabel,
                min: formatBound(stats.min),
                max: formatBound(stats.max),
                profit: profitValue.toFixed(4),
                wins_kept_pct: Number.isFinite(stats.win_keep_pct) ? `${Math.round(stats.win_keep_pct * 100)}%` : 'n/a',
                losses_cut_pct: Number.isFinite(stats.loss_cut_pct) ? `${Math.round(stats.loss_cut_pct * 100)}%` : 'n/a',
                wins_lost: stats.wins_lost ?? 'n/a',
                losses_cut_trades: stats.losses_cut ?? 'n/a',
                _profitSort: profitValue,
            };
        }).sort((a, b) => (b._profitSort - a._profitSort)).map(row => {
            delete row._profitSort;
            return row;
        });

        const cols = ["indicator","min","max","profit","wins_kept_pct","losses_cut_pct","wins_lost","losses_cut_trades"];
        tableFromArray(rows, cols, "indicator-thresholds");
    }

    function indicatorSpecsTable(){
        const container = document.getElementById("indicator-specs");
        if(!container) return;
        const specs = DATA.indicator_specs || {};
        const entries = Object.entries(specs);
        if(!entries.length){
            container.innerHTML = '<div class="muted">No indicator metadata available.</div>';
            return;
        }
        const rows = entries.map(([key, spec]) => {
            const paramEntries = Object.entries(spec.parameters || {});
            const params = paramEntries.length ? paramEntries.map(([k, v]) => `${k}: ${v}`).join('; ') : 'n/a';
            return {
                indicator: spec.label || key,
                timeframe: spec.timeframe || 'n/a',
                parameters: params,
                description: spec.description || ''
            };
        });
        tableFromArray(rows, ["indicator","timeframe","parameters","description"], "indicator-specs");
    }

    function correlationProfitTable(){
        const container = document.getElementById("corr-indicators");
        if(!container) return;
        const ml = DATA.ml || {};
        const rows = ml.correlation_profit || [];
        if(!rows.length){
            container.innerHTML = '<div class="muted">Not enough overlapping data to compute correlations.</div>';
            return;
        }
        tableFromArray(rows, ["feature","corr_profit","corr_abs"], "corr-indicators");
    }

    function correlationMatrixTable(){
        const container = document.getElementById("corr-matrix");
        if(!container) return;
        const ml = DATA.ml || {};
        const matrix = ml.correlation_matrix;
        if(!matrix || !matrix.features || !matrix.matrix || !matrix.features.length){
            container.innerHTML = '<div class="muted">Correlation matrix unavailable.</div>';
            return;
        }
        const features = matrix.features;
        const header = `<thead><tr><th>Feature</th>${features.map(f => `<th>${f}</th>`).join('')}</tr></thead>`;

        const colorFor = (value) => {
            const v = Number(value);
            if(!Number.isFinite(v)) return '';
            const intensity = Math.min(1, Math.abs(v));
            const hue = v >= 0 ? 140 : 0; // green vs red
            const lightness = 60 - intensity * 25; // darker with higher magnitude
            const textColor = intensity > 0.55 ? '#fff' : '#111';
            return `background-color:hsl(${hue},70%,${lightness}%);color:${textColor};`;
        };

        const body = matrix.matrix.map((rowVals, idx) => {
            const cells = rowVals.map((val, jdx) => {
                if(!Number.isFinite(val)){
                    return `<td></td>`;
                }
                const style = colorFor(val);
                return `<td style="${style}">${Number(val).toFixed(4)}</td>`;
            }).join('');
            return `<tr><td><b>${features[idx]}</b></td>${cells}</tr>`;
        }).join('');

        container.innerHTML = `<table>${header}<tbody>${body}</tbody></table>`;
    }

    function logisticSection(){
        const summaryContainer = document.getElementById("logistic-summary");
        const coefContainer = document.getElementById("logistic-coefs");
        if(!summaryContainer || !coefContainer) return;
        const logi = (DATA.ml && DATA.ml.logistic) || null;
        if(!logi || !logi.available){
            const reason = logi && logi.reason ? logi.reason : 'Logistic regression not available (insufficient data or sklearn missing).';
            summaryContainer.innerHTML = `<div class="muted">${reason}</div>`;
            coefContainer.innerHTML = '';
            return;
        }
        const metrics = [
            { metric: 'Samples', value: logi.n_samples },
            { metric: 'Train Samples', value: logi.train_size },
            { metric: 'Test Samples', value: logi.test_size },
            { metric: 'Accuracy', value: `${(logi.accuracy * 100).toFixed(2)}%` },
            { metric: 'Intercept', value: logi.intercept }
        ];
        tableFromArray(metrics, ["metric","value"], "logistic-summary");
        tableFromArray(logi.coefficients, ["feature","coef","odds_ratio"], "logistic-coefs");
    }

    function treeSection(){
        const summaryContainer = document.getElementById("tree-summary");
        const impContainer = document.getElementById("tree-importances");
        if(!summaryContainer || !impContainer) return;
        const tree = (DATA.ml && DATA.ml.decision_tree) || null;
        if(!tree || !tree.available){
            const reason = tree && tree.reason ? tree.reason : 'Decision tree not available (insufficient data or sklearn missing).';
            summaryContainer.innerHTML = `<div class="muted">${reason}</div>`;
            impContainer.innerHTML = '';
            return;
        }
        const metrics = [
            { metric: 'Samples', value: tree.n_samples },
            { metric: 'Train Samples', value: tree.train_size },
            { metric: 'Test Samples', value: tree.test_size },
            { metric: 'Accuracy', value: `${(tree.accuracy * 100).toFixed(2)}%` },
            { metric: 'Depth', value: tree.depth },
            { metric: 'Leaves', value: tree.n_leaves }
        ];
        tableFromArray(metrics, ["metric","value"], "tree-summary");
        tableFromArray(tree.feature_importances, ["feature","importance"], "tree-importances");
    }

    function combinationTables(){
        const ml = DATA.ml || {};
        const combos = ml.combinations || {};
        const pairContainer = document.getElementById("combo-pairs");
        const tripContainer = document.getElementById("combo-triplets");

        const renderTable = (rowsRaw, containerId) => {
            const el = document.getElementById(containerId);
            if(!el) return;
            if(!rowsRaw || !rowsRaw.length){
                el.innerHTML = '<div class="muted">No combinations cleared the minimum sample threshold.</div>';
                return;
            }
            const rows = rowsRaw.map(row => {
                const feats = row.features || [];
                const coeffs = Array.isArray(row.coefficients)
                    ? row.coefficients.map((coef, idx) => {
                        const feat = feats[idx] || idx;
                        return `${feat}:${Number(coef).toFixed(4)}`;
                    }).join(' | ')
                    : '';
                return {
                    features: feats.join(', '),
                    accuracy: `${Number(row.accuracy * 100).toFixed(2)}%`,
                    lift: `${Number(row.lift * 100).toFixed(2)}pp`,
                    baseline: `${Number(row.baseline * 100).toFixed(2)}%`,
                    samples: row.samples,
                    train_size: row.train_size,
                    test_size: row.test_size,
                    coeffs: coeffs,
                };
            });
            tableFromArray(rows, ["features","accuracy","lift","baseline","samples","train_size","test_size","coeffs"], containerId);
        };

        renderTable(combos.pairs || [], "combo-pairs");
        renderTable(combos.triplets || [], "combo-triplets");
    }

    function indicatorBucketLabel(indicator, fallback){
        const base = indicatorBaseTitle(indicator, fallback, "");
        return base ? `${base} bucket` : 'Bucket';
    }

    const AXIS_LABELS = {
        counts: 'Trades (count)',
        winrate: 'Win-rate (%)',
        hour: 'Hour of day',
        weekday: 'Day of week'
    };

    function initCharts(){
        const dist = DATA.distributions;
        const wb = DATA.winrate_bins;

        setIndicatorHeading("heading-ema_slope", "ema_slope", "W vs L");
        setIndicatorHeading("heading-don_pos", "don_pos", "W vs L");
        setIndicatorHeading("heading-atr_norm", "atr_norm", "W vs L");
        setIndicatorHeading("heading-rsi", "rsi", "W vs L");
        setIndicatorHeading("heading-adx", "adx", "W vs L");
        setIndicatorHeading("heading-wr-emaslope", "ema_slope", "Win-rate");
        setIndicatorHeading("heading-wr-donpos", "don_pos", "Win-rate");
        setIndicatorHeading("heading-wr-atr", "atr_norm", "Win-rate by regime");

        // Distributions (W vs L) - we plot winners bars and losers bars with threshold details
    plotBar("chart-ema_slope", dist.ema_slope.labels, dist.ema_slope.winners, dist.ema_slope.losers, indicatorTitle("ema_slope", "EMA slope", "Histogram"), indicatorBucketLabel("ema_slope", "EMA slope"), AXIS_LABELS.counts);
    plotBar("chart-don_pos",   dist.don_pos.labels,   dist.don_pos.winners,   dist.don_pos.losers,   indicatorTitle("don_pos", "Donchian position", "Histogram"), indicatorBucketLabel("don_pos", "Donchian position"), AXIS_LABELS.counts);
    plotBar("chart-atr_norm",  dist.atr_norm.labels,  dist.atr_norm.winners,  dist.atr_norm.losers,  indicatorTitle("atr_norm", "ATR normalized", "Histogram"), indicatorBucketLabel("atr_norm", "ATR normalized"), AXIS_LABELS.counts);
    plotBar("chart-rsi",       dist.rsi.labels,       dist.rsi.winners,       dist.rsi.losers,       indicatorTitle("rsi", "RSI", "Histogram"), indicatorBucketLabel("rsi", "RSI"), AXIS_LABELS.counts);
    plotBar("chart-adx",       dist.adx.labels,       dist.adx.winners,       dist.adx.losers,       indicatorTitle("adx", "ADX", "Histogram"), indicatorBucketLabel("adx", "ADX"), AXIS_LABELS.counts);

    // Win-rate binned
    plotLine("chart-wr-emaslope", wb.ema_slope.labels, wb.ema_slope.wr_pct, indicatorBaseTitle("ema_slope", "EMA slope", "Win-rate"), indicatorBucketLabel("ema_slope", "EMA slope"), AXIS_LABELS.winrate);
    plotLine("chart-wr-donpos",   wb.don_pos.labels,   wb.don_pos.wr_pct,   indicatorBaseTitle("don_pos", "Donchian position", "Win-rate"), indicatorBucketLabel("don_pos", "Donchian position"), AXIS_LABELS.winrate);
    plotBarSingle("chart-wr-atr", wb.atr_regime.labels, wb.atr_regime.wr_pct, indicatorBaseTitle("atr_norm", "ATR normalized", "Win-rate by regime"), "ATR regime", AXIS_LABELS.winrate);

        // Time
        plotLine("chart-wr-hour",    DATA.time_winrate.hour.map(x=>x.bucket),    DATA.time_winrate.hour.map(x=>x.wr_pct), "Win-rate by Hour", AXIS_LABELS.hour, AXIS_LABELS.winrate);
        plotBarSingle("chart-wr-weekday", DATA.time_winrate.weekday.map(x=>x.bucket), DATA.time_winrate.weekday.map(x=>x.wr_pct), "Win-rate by Weekday", AXIS_LABELS.weekday, AXIS_LABELS.winrate);
    }

    function plotBar(canvasId, labels, winVals, loseVals, title, xLabel, yLabel){
                if(hasChartJs){
                        new Chart(document.getElementById(canvasId), {
                             type:'bar',
                             data:{labels:labels, datasets:[
                                 {label:'Winners (count)', data:winVals, borderWidth:1},
                                 {label:'Losers (count)',  data:loseVals, borderWidth:1}
                             ]},
                             options:{responsive:true, plugins:{title:{display:true, text:title}}, scales:{x:{title:{display:!!xLabel,text:xLabel},ticks:{autoSkip:true,maxTicksLimit:12}}, y:{title:{display:!!yLabel,text:yLabel}}}}
                        });
        }else{
            // fallback: sum winners & losers and plot as single bars
            const combined = labels.map((_,i)=> (winVals[i]||0)+(loseVals[i]||0));
            fallbackBarChart(canvasId, labels, combined, title + " (fallback)", xLabel, yLabel);
        }
    }

    function plotBarSingle(canvasId, labels, vals, title, xLabel, yLabel){
        if(hasChartJs){
            new Chart(document.getElementById(canvasId), {
               type:'bar',
               data:{labels:labels, datasets:[{label:'Win-rate %', data:vals, borderWidth:1}]},
               options:{responsive:true, plugins:{title:{display:true, text:title}}, scales:{x:{title:{display:!!xLabel,text:xLabel}}, y:{title:{display:!!yLabel,text:yLabel}}}}
            });
        }else{
            fallbackBarChart(canvasId, labels, vals, title + " (fallback)", xLabel, yLabel);
        }
    }

    function plotLine(canvasId, labels, vals, title, xLabel, yLabel){
        if(hasChartJs){
            new Chart(document.getElementById(canvasId), {
               type:'line',
               data:{labels:labels, datasets:[{label:'Win-rate %', data:vals, borderWidth:2, fill:false}]},
               options:{responsive:true, plugins:{title:{display:true, text:title}}, scales:{x:{title:{display:!!xLabel,text:xLabel}}, y:{title:{display:!!yLabel,text:yLabel}}}}
            });
        }else{
            fallbackLineChart(canvasId, labels, vals, title + " (fallback)", xLabel, yLabel);
        }
    }

    document.addEventListener('DOMContentLoaded', ()=>{
        metaSummary();
        featureSummaryTable();
        strategySettings();
        indicatorSpecsTable();
        indicatorThresholdTable();
                correlationProfitTable();
                correlationMatrixTable();
                logisticSection();
                treeSection();
                combinationTables();
        patternTables();
        timeTables();
        initCharts();
        if(!hasChartJs){
          const n = document.getElementById('no-chartjs-note');
          n.style.display = 'block';
        }
    });
    """.replace("__DATA_JSON__", data_json)

    # HTML template
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Backtest — Winners vs Losers (offline)</title>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<style>{css}</style>
</head>
<body>
  <h1>Backtest — Winners vs Losers</h1>
    <div class="muted">Self-contained offline report. Generated on {generated_on}</div>
    <div id="meta" class="card"></div>
    {chart_note}

  <div class="grid cols-2">
    <div class="card">
      <h3>Top Winner Patterns</h3>
      <div id="winners-patterns"></div>
    </div>
    <div class="card">
      <h3>Top Loser Patterns</h3>
      <div id="losers-patterns"></div>
    </div>
  </div>

  <div class="card">
    <h3>Feature Summary by Group</h3>
    <div id="feature-summary"></div>
  </div>

    <div class="card">
        <h3>Indicator Inputs</h3>
        <div id="indicator-specs"></div>
    </div>

    <div class="card">
        <h3>Indicator Filters by Profit</h3>
        <div id="indicator-thresholds"></div>
    </div>

    <div class="card">
        <h3>Strategy Settings</h3>
        <div id="strategy-settings"></div>
    </div>

      <div class="grid cols-2">
          <div class="card">
              <h3>Indicator ↔ Profit Correlation</h3>
              <div id="corr-indicators"></div>
          </div>
          <div class="card">
              <h3>Correlation Matrix (Indicators + Profit)</h3>
              <div id="corr-matrix"></div>
          </div>
      </div>

      <div class="grid cols-2">
          <div class="card">
              <h3>Logistic Regression Summary</h3>
              <div id="logistic-summary"></div>
              <div id="logistic-coefs" style="margin-top:12px;"></div>
          </div>
          <div class="card">
              <h3>Decision Tree Summary</h3>
              <div id="tree-summary"></div>
              <div id="tree-importances" style="margin-top:12px;"></div>
          </div>
      </div>

      <div class="grid cols-2">
          <div class="card">
              <h3>Top Feature Pairs (Logistic)</h3>
              <div id="combo-pairs"></div>
          </div>
          <div class="card">
              <h3>Top Feature Triplets (Logistic)</h3>
              <div id="combo-triplets"></div>
          </div>
      </div>

    <div class="grid cols-2">
        <div class="card">
            <h3>Win-rate by Hour (Table)</h3>
            <div id="time-hour"></div>
        </div>
        <div class="card">
            <h3>Win-rate by Weekday (Table)</h3>
            <div id="time-weekday"></div>
        </div>
    </div>

    <div class="grid cols-2">
        <div class="card"><h3 id="heading-ema_slope" data-label="EMA slope">EMA slope — W vs L</h3><canvas id="chart-ema_slope"></canvas></div>
        <div class="card"><h3 id="heading-don_pos" data-label="Donchian position">Donchian position — W vs L</h3><canvas id="chart-don_pos"></canvas></div>
    </div>
    <div class="grid cols-2">
        <div class="card"><h3 id="heading-atr_norm" data-label="ATR normalized">ATR normalized — W vs L</h3><canvas id="chart-atr_norm"></canvas></div>
        <div class="card"><h3 id="heading-rsi" data-label="RSI">RSI — W vs L</h3><canvas id="chart-rsi"></canvas></div>
    </div>
    <div class="card"><h3 id="heading-adx" data-label="ADX">ADX — W vs L</h3><canvas id="chart-adx"></canvas></div>

    <div class="grid cols-2">
        <div class="card"><h3 id="heading-wr-emaslope" data-label="EMA slope">Win-rate by EMA slope</h3><canvas id="chart-wr-emaslope"></canvas></div>
        <div class="card"><h3 id="heading-wr-donpos" data-label="Donchian position">Win-rate by Donchian position</h3><canvas id="chart-wr-donpos"></canvas></div>
    </div>
    <div class="card"><h3 id="heading-wr-atr" data-label="ATR normalized">Win-rate by ATR regime</h3><canvas id="chart-wr-atr"></canvas></div>

  <div class="grid cols-2">
    <div class="card"><h3>Win-rate by Hour</h3><canvas id="chart-wr-hour"></canvas></div>
    <div class="card"><h3>Win-rate by Weekday</h3><canvas id="chart-wr-weekday"></canvas></div>
  </div>

    {chartjs_tag}
  <script>{fallback_js}</script>
  <script>{chart_init}</script>
</body>
</html>
"""
    return html

# -----------------------------
# Main
# -----------------------------

def main():
    _ensure_dir(OUTPUT_DIR)

    # 0–1) Locate latest backtest
    latest_path = _find_latest_backtest()
    folder = _unpack_if_needed(latest_path)

    # 2) Load result files
    trade_json_path, market_feather_path = _match_result_files(folder)
    trades_df, meta_json = _load_trades_json(trade_json_path)
    market_df = _load_market_feather(market_feather_path)  # may be None

    # Sanity
    total_trades = len(trades_df)
    winners_mask = trades_df["profit_ratio"] > 0
    n_w = int(winners_mask.sum())
    n_l = int((~winners_mask).sum())
    pairs = sorted(trades_df["pair"].astype(str).unique().tolist())
    date_min = pd.to_datetime(trades_df["open_date"]).min()
    date_max = pd.to_datetime(trades_df["close_date"]).max()
    date_span = f"{date_min.strftime('%Y-%m-%d')} → {date_max.strftime('%Y-%m-%d')}" if pd.notna(date_min) and pd.notna(date_max) else "n/a"

    baseline_wr = 100.0 * (n_w / max(1, total_trades))

    # 3) Optional feature reconstruction from candles
    timeframe = (meta_json.get("timeframe")
                 or meta_json.get("strategy", {}).get("timeframe")
                 or "15m")
    strategy_settings = {}
    indicator_specs = _build_indicator_specs(str(timeframe))
    settings_keys = [
        "strategy_name",
        "timeframe",
        "timeframe_detail",
        "stake_currency",
        "stake_amount",
        "max_open_trades",
        "max_open_trades_setting",
        "stoploss",
        "trailing_stop",
        "trailing_stop_positive",
        "trailing_stop_positive_offset",
        "trailing_only_offset_is_reached",
        "use_custom_stoploss",
        "minimal_roi",
        "exit_profit_only",
        "exit_profit_offset",
        "trading_mode",
        "margin_mode",
    ]
    for key in settings_keys:
        if key in meta_json and meta_json[key] is not None:
            val = meta_json[key]
            if isinstance(val, dict):
                val = json.dumps(val, ensure_ascii=False)
            strategy_settings[key] = val

    pair_list = sorted(trades_df["pair"].astype(str).unique().tolist())
    candles_by_pair, missing_pairs = _load_all_candle_files(timeframe, pair_list)

    fallback_hour = pd.to_datetime(trades_df["open_date"]).dt.hour
    fallback_weekday = pd.to_datetime(trades_df["open_date"]).dt.weekday

    if candles_by_pair:
        features_at_entry = _align_trade_features(trades_df, candles_by_pair)
        enriched = pd.concat([trades_df.reset_index(drop=True), features_at_entry.reset_index(drop=True)], axis=1)
        if "hour" in enriched.columns:
            enriched["hour"] = enriched["hour"].fillna(fallback_hour)
        else:
            enriched["hour"] = fallback_hour
        if "weekday" in enriched.columns:
            enriched["weekday"] = enriched["weekday"].fillna(fallback_weekday)
        else:
            enriched["weekday"] = fallback_weekday
    else:
        missing_pairs = pair_list
        enriched = trades_df.copy()
        enriched["hour"] = fallback_hour
        enriched["weekday"] = fallback_weekday
        for col in ["ema_slope","don_pos","atr_norm","rsi","adx","don_width","ema_dist","vol_z"]:
            if col not in enriched.columns:
                enriched[col] = np.nan

    if missing_pairs:
        preview = ", ".join(sorted(missing_pairs)[:8])
        if len(missing_pairs) > 8:
            preview += ", …"
        print(f"     Warning: missing indicators for {len(missing_pairs)} pair(s): {preview}")

    # 4) Winners FIRST — distributions & stats
    wmask = enriched["profit_ratio"] > 0

    # Distributions
    dist = {}
    for key in ["ema_slope","don_pos","atr_norm","rsi","adx"]:
        labels, wvals, lvals = _distribution(enriched[key], wmask, num=40)
        dist[key] = {"labels": labels, "winners": wvals, "losers": lvals}

    # Win-rate by binned feature
    bins_emaslope = "quantile_5"
    bins_donpos   = [0.0, .2, .4, .6, .8, 1.0]
    # ATR regime
    if "atr_norm" in enriched.columns and enriched["atr_norm"].notna().sum() >= MIN_BIN_SIZE:
        atr_labels = ["Low","Medium","High"]
        atr_reg = pd.qcut(enriched["atr_norm"], q=3, duplicates="drop", labels=atr_labels)
    else:
        atr_labels = ["Low","Medium","High"]
        atr_reg = pd.Series([np.nan]*len(enriched), index=enriched.index)

    # Build binned WR datasets
    emaslope_labels, emaslope_wr, emaslope_counts = _binned_winrate(enriched["ema_slope"], wmask, bins_emaslope)
    don_labels, don_wr, don_counts = _binned_winrate(enriched["don_pos"], wmask, bins_donpos)

    # ATR WR
    atr_wr = []
    atr_counts = []
    for lab in atr_labels:
        mask = (atr_reg == lab)
        n = int(mask.sum())
        if n < MIN_BIN_SIZE:
            atr_wr.append(None)
            atr_counts.append(n)
        else:
            atr_wr.append(float(100.0 * wmask[mask].mean()))
            atr_counts.append(n)

    # 4b) Feature summary (means/medians/std + effect sizes, optional KS)
    feature_summary = []
    feature_lookup: dict[str, dict[str, dict]] = {}
    feat_list = ["ema_slope","don_pos","atr_norm","rsi","adx","don_width","ema_dist","vol_z","holding_s","hour","weekday"]
    for feat in feat_list:
        if feat not in enriched.columns:
            continue
        wvals = enriched.loc[wmask, feat]
        lvals = enriched.loc[~wmask, feat]
        eff = _effect_size(wvals, lvals)
        pval = _ks_pvalue(wvals, lvals)
        effect_txt = f"d={eff:.2f}" if not np.isnan(eff) else ""
        if pval is not None:
            effect_txt += f"; KS p={pval:.3f}"

        def _row(g, s):
            return {
                "feature": feat,
                "group": g,
                "mean": None if s.dropna().empty else float(np.mean(s.dropna())),
                "median": None if s.dropna().empty else float(np.median(s.dropna())),
                "std": None if s.dropna().empty else float(np.std(s.dropna(), ddof=0)),
                "effect": effect_txt if g == "W" else "",
                "n": int(s.dropna().shape[0]),
            }
        feature_summary.append(_row("W", wvals))
    feature_summary.append(_row("L", lvals))
    feature_lookup.setdefault(feat, {})["W"] = feature_summary[-2]
    feature_lookup[feat]["L"] = feature_summary[-1]

    # Time win-rates
    hour_wr = _winrate_by(enriched["hour"], wmask, list(range(0,24)))
    weekday_map = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}
    weekday_wr = _winrate_by(enriched["weekday"].map(weekday_map), wmask,
                             ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])

    # Patterns
    winners_patterns = _top_patterns(enriched, wmask, baseline_wr, want="winners")
    losers_patterns  = _top_patterns(enriched, wmask, baseline_wr, want="losers")

    # 6) Build data object
    # Build lightweight indicator delta view for summary
    indicator_deltas = []
    for feat, groups in feature_lookup.items():
        if "W" not in groups or "L" not in groups:
            continue
        w = groups["W"]
        l = groups["L"]
        if w["mean"] is None or l["mean"] is None:
            diff = None
        else:
            diff = float(w["mean"] - l["mean"])
        indicator_deltas.append({
            "feature": feat,
            "winner_mean": w["mean"],
            "loser_mean": l["mean"],
            "mean_diff": diff,
            "winner_median": w["median"],
            "loser_median": l["median"],
        })
    indicator_deltas.sort(key=lambda x: 0 if x["mean_diff"] is None else abs(x["mean_diff"]), reverse=True)

    # Compute optimal min/max thresholds for each indicator
    def compute_thresholds(df: pd.DataFrame, indicator: str, wmask: pd.Series, profits: pd.Series):
        if indicator not in df.columns:
            return None
        vals = df[indicator]
        valid_mask = vals.notna() & profits.notna()
        if valid_mask.sum() < max(40, int(0.1 * len(df))):
            return None
        vals = vals[valid_mask]
        profits = profits[valid_mask]
        win_mask = wmask[valid_mask]

        total_profit = float(profits.sum())
        total_wins = int(win_mask.sum())
        total_losses = int((~win_mask).sum())
        if total_wins == 0 or total_losses == 0:
            return None

        # Build candidate grid via quantiles to keep computation reasonable
        quantile_points = np.linspace(0.0, 1.0, 21)
        try:
            grid = np.unique(np.quantile(vals, quantile_points, method="linear"))
        except TypeError:  # numpy < 1.22
            grid = np.unique(np.quantile(vals, quantile_points))
        if len(grid) < 2:
            return None

        best_candidate = None
        best_score = None
        preferred_candidate = None
        preferred_score = None
        MIN_WIN_KEEP_RATIO = 0.6

        for i in range(len(grid)):
            for j in range(i, len(grid)):
                min_t = grid[i]
                max_t = grid[j]
                keep_mask = (vals >= min_t) & (vals <= max_t)
                kept = int(keep_mask.sum())
                if kept < max(30, int(0.05 * len(vals))):
                    continue

                kept_profit = float(profits[keep_mask].sum())
                wins_kept = int(win_mask[keep_mask].sum())
                losses_kept = kept - wins_kept
                cut_losses = total_losses - losses_kept
                lost_wins = total_wins - wins_kept
                profit_change = kept_profit - total_profit
                win_keep_ratio = wins_kept / total_wins if total_wins else 0.0
                loss_cut_ratio = cut_losses / total_losses if total_losses else 0.0

                # Enforce profit not decreasing ("profit same bigger than before")
                if kept_profit + 1e-12 < total_profit:
                    continue

                score = (kept_profit, win_keep_ratio, loss_cut_ratio, -lost_wins)

                candidate = {
                    "min": float(min_t),
                    "max": float(max_t),
                    "losses_cut": int(cut_losses),
                    "wins_lost": int(lost_wins),
                    "wins_kept": int(wins_kept),
                    "losses_kept": int(losses_kept),
                    "kept_trades": int(kept),
                    "total_trades": int(len(vals)),
                    "profit_kept": float(kept_profit),
                    "profit_baseline": float(total_profit),
                    "profit_change": float(profit_change),
                    "win_keep_pct": float(win_keep_ratio),
                    "loss_cut_pct": float(loss_cut_ratio),
                }

                if best_score is None or score > best_score:
                    best_score = score
                    best_candidate = candidate

                if win_keep_ratio >= MIN_WIN_KEEP_RATIO:
                    if preferred_score is None or score > preferred_score:
                        preferred_score = score
                        preferred_candidate = candidate

        return preferred_candidate or best_candidate

    indicator_thresholds = {}
    profit_series = enriched["profit_abs"] if "profit_abs" in enriched.columns else enriched["profit_ratio"]
    for key in ["ema_slope","don_pos","atr_norm","rsi","adx"]:
        th = compute_thresholds(enriched, key, wmask, profit_series)
        if th:
            indicator_thresholds[key] = th

    # 5) Correlation and ML diagnostics (optional deeper insight)
    ml_features_all = [
        "ema_slope",
        "don_pos",
        "atr_norm",
        "rsi",
        "adx",
        "don_width",
        "ema_dist",
        "vol_z",
        "holding_s",
        "hour",
        "weekday",
    ]
    ml_features = [col for col in ml_features_all if col in enriched.columns]
    corr_profit_rows: list[dict[str, float | str]] = []
    corr_matrix_dict: dict[str, object] | None = None
    logistic_summary: dict[str, object] = {"available": False}
    tree_summary: dict[str, object] = {"available": False}
    combo_summary: dict[str, list[dict[str, object]]] = {"pairs": [], "triplets": []}

    if ml_features:
        corr_source = enriched[ml_features + ["profit_ratio"]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(corr_source) >= max(MIN_BIN_SIZE, 40):
            corr_df = corr_source.astype(float)
            corr_mat = corr_df.corr()
            corr_matrix_dict = {
                "features": corr_mat.columns.tolist(),
                "matrix": np.round(corr_mat.values, 6).tolist(),
            }
            if "profit_ratio" in corr_mat.columns:
                for feat in ml_features:
                    if feat in corr_mat.index:
                        value = float(corr_mat.loc[feat, "profit_ratio"])
                        corr_profit_rows.append({
                            "feature": feat,
                            "corr_profit": round(value, 6),
                            "corr_abs": round(abs(value), 6),
                        })
                corr_profit_rows.sort(key=lambda row: row["corr_abs"], reverse=True)

        # Logistic regression / decision tree only when we have adequate data & sklearn
        if _HAVE_SKLEARN and "profit_ratio" in corr_source.columns and len(corr_source) >= max(120, MIN_BIN_SIZE * 2):
            outcome = (corr_source["profit_ratio"] > 0).astype(int)
            if outcome.nunique() >= 2:
                X = corr_source[ml_features].values
                y = outcome.values
                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X,
                        y,
                        test_size=0.3,
                        random_state=42,
                        stratify=y,
                    )
                except ValueError:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.3, random_state=42
                    )

                if len(np.unique(y_train)) >= 2 and len(np.unique(y_test)) >= 2:
                    # Logistic regression (standardized features)
                    try:
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        logi = LogisticRegression(max_iter=1000)
                        logi.fit(X_train_scaled, y_train)
                        y_pred = logi.predict(X_test_scaled)
                        acc = float(np.mean(y_pred == y_test))
                        coeffs = [
                            {
                                "feature": feat,
                                "coef": round(float(coef), 6),
                                "odds_ratio": round(float(np.exp(coef)), 6),
                            }
                            for feat, coef in zip(ml_features, logi.coef_[0])
                        ]
                        logistic_summary = {
                            "available": True,
                            "n_samples": int(len(corr_source)),
                            "train_size": int(len(y_train)),
                            "test_size": int(len(y_test)),
                            "accuracy": round(acc, 4),
                            "intercept": round(float(logi.intercept_[0]), 6),
                            "coefficients": coeffs,
                        }
                    except Exception as exc:  # pragma: no cover - diagnostic only
                        logistic_summary = {
                            "available": False,
                            "reason": str(exc),
                        }

                    # Decision tree (raw features)
                    try:
                        leaf_min = max(5, int(len(y_train) * 0.05))
                        tree = DecisionTreeClassifier(
                            max_depth=3,
                            min_samples_leaf=leaf_min,
                            random_state=42,
                        )
                        tree.fit(X_train, y_train)
                        tree_acc = float(tree.score(X_test, y_test))
                        importances = [
                            {
                                "feature": feat,
                                "importance": round(float(val), 6),
                            }
                            for feat, val in zip(ml_features, tree.feature_importances_)
                        ]
                        tree_summary = {
                            "available": True,
                            "n_samples": int(len(corr_source)),
                            "train_size": int(len(y_train)),
                            "test_size": int(len(y_test)),
                            "accuracy": round(tree_acc, 4),
                            "depth": int(tree.get_depth()),
                            "n_leaves": int(tree.get_n_leaves()),
                            "feature_importances": importances,
                        }
                    except Exception as exc:  # pragma: no cover
                        tree_summary = {
                            "available": False,
                            "reason": str(exc),
                        }

                    # Combination search (pairs / triplets)
                    def _evaluate_combos(size: int, limit: int = 10) -> list[dict[str, object]]:
                        results: list[dict[str, object]] = []
                        feature_count = len(ml_features)
                        min_samples = max(120, int(0.2 * len(corr_source)))
                        for combo in itertools.combinations(ml_features, size):
                            subset = corr_source[list(combo) + ["profit_ratio"]].replace([np.inf, -np.inf], np.nan).dropna()
                            if len(subset) < min_samples:
                                continue
                            y_combo = (subset["profit_ratio"] > 0).astype(int)
                            if y_combo.nunique() < 2:
                                continue
                            X_combo = subset[list(combo)].values
                            y_vals = y_combo.values
                            try:
                                X_tr, X_te, y_tr, y_te = train_test_split(
                                    X_combo,
                                    y_vals,
                                    test_size=0.3,
                                    random_state=42,
                                    stratify=y_vals,
                                )
                            except ValueError:
                                X_tr, X_te, y_tr, y_te = train_test_split(
                                    X_combo,
                                    y_vals,
                                    test_size=0.3,
                                    random_state=42,
                                )
                            if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                                continue

                            scaler_c = StandardScaler()
                            X_tr_s = scaler_c.fit_transform(X_tr)
                            X_te_s = scaler_c.transform(X_te)
                            model = LogisticRegression(max_iter=1000)
                            try:
                                model.fit(X_tr_s, y_tr)
                            except Exception:
                                continue
                            acc = float(model.score(X_te_s, y_te))
                            base = max(float(np.mean(y_tr)), 1.0 - float(np.mean(y_tr)))
                            lift = acc - base
                            coeffs = [round(float(c), 6) for c in model.coef_[0]] if model.coef_.size else []
                            results.append({
                                "features": list(combo),
                                "accuracy": round(acc, 4),
                                "lift": round(lift, 4),
                                "baseline": round(base, 4),
                                "samples": int(len(subset)),
                                "train_size": int(len(y_tr)),
                                "test_size": int(len(y_te)),
                                "coefficients": coeffs,
                            })
                        results.sort(key=lambda item: (item["accuracy"], item["lift"]), reverse=True)
                        return results[:limit]

                    combo_summary["pairs"] = _evaluate_combos(2)
                    combo_summary["triplets"] = _evaluate_combos(3)

    data_obj = {
        "meta": {
            "total_trades": int(total_trades),
            "winners": int(n_w),
            "losers": int(n_l),
            "baseline_winrate_pct": float(baseline_wr),
            "date_span": date_span,
            "pairs": pairs,
            "missing_indicator_pairs": sorted(missing_pairs),
            "strategy_settings": strategy_settings,
        },
        "winners_patterns": winners_patterns,
        "losers_patterns": losers_patterns,
        "distributions": {
            "ema_slope": dist["ema_slope"],
            "don_pos":   dist["don_pos"],
            "atr_norm":  dist["atr_norm"],
            "rsi":       dist["rsi"],
            "adx":       dist["adx"],
        },
        "winrate_bins": {
            "ema_slope": {
                "labels": emaslope_labels,
                "wr_pct": emaslope_wr,
                "counts": emaslope_counts,
                "baseline_wr_pct": float(baseline_wr),
            },
            "don_pos": {
                "labels": [str(x) for x in don_labels],
                "wr_pct": don_wr,
                "counts": don_counts,
                "baseline_wr_pct": float(baseline_wr),
            },
            "atr_regime": {
                "labels": ["Low","Medium","High"],
                "wr_pct": atr_wr,
                "counts": atr_counts,
                "baseline_wr_pct": float(baseline_wr),
            }
        },
        "heatmap": {
            # Placeholder for future ADX x Donchian heatmap if you want to expand
            "xBins": [],
            "yBins": [],
            "zValues": [],
            "counts": []
        },
        "feature_summary": feature_summary,
        "time_winrate": {
            "hour": hour_wr,
            "weekday": weekday_wr
        },
        "indicator_deltas": indicator_deltas,
        "indicator_thresholds": indicator_thresholds,
        "indicator_specs": indicator_specs,
        "ml": {
            "correlation_matrix": corr_matrix_dict,
            "correlation_profit": corr_profit_rows,
            "logistic": logistic_summary,
            "decision_tree": tree_summary,
            "combinations": combo_summary,
        },
    }

    # 6E) Render single-file HTML
    html = _build_html(data_obj)
    _write_text(OUTPUT_HTML, html)

    # Small AI-friendly summary dump (plain text)
    top_winners = winners_patterns[:3]
    top_losers = losers_patterns[:3]
    top_indicators = [d for d in indicator_deltas if d["mean_diff"] is not None][:5]

    lines = [
        f"Generated UTC: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total trades: {total_trades} (winners {n_w}, losers {n_l})",
        f"Baseline win rate: {baseline_wr:.2f}%",
    ]

    if strategy_settings:
        strat_name = strategy_settings.get("strategy_name") or meta_json.get("strategy_name") or "Unknown"
        timeframe_out = strategy_settings.get("timeframe") or timeframe
        stoploss_out = strategy_settings.get("stoploss")
        lines.append(f"Strategy: {strat_name} | Timeframe: {timeframe_out} | Stoploss: {stoploss_out}")

    if indicator_thresholds:
        best_feat, best_stats = max(indicator_thresholds.items(), key=lambda kv: kv[1].get("profit_change", 0.0))
        if best_stats.get("profit_change", 0.0) >= 0:
            lines.append(
                "Best profit-preserving filter: "
                f"{best_feat} between {best_stats['min']:.6f} and {best_stats['max']:.6f} "
                f"(ΔProfit {best_stats['profit_change']:.6f}, losses cut {best_stats['losses_cut']}, wins lost {best_stats['wins_lost']})."
            )

    if missing_pairs:
        preview = ", ".join(sorted(missing_pairs)[:6])
        if len(missing_pairs) > 6:
            preview += ", …"
        lines.append(f"Missing indicator pairs: {preview}")

    if corr_profit_rows:
        lines.append("Indicator correlations vs profit (top 3 | Pearson r):")
        for row in corr_profit_rows[:3]:
            lines.append(
                f"  - {row['feature']}: {row['corr_profit']:+.4f}"
            )

    if logistic_summary.get("available"):
        lines.append(
            f"Logistic regression accuracy: {logistic_summary['accuracy'] * 100:.2f}%"
            f" (train {logistic_summary['train_size']}, test {logistic_summary['test_size']})."
        )
        top_coeffs = sorted(
            logistic_summary.get("coefficients", []),
            key=lambda item: abs(item.get("coef", 0.0)),
            reverse=True,
        )[:3]
        if top_coeffs:
            lines.append("  Coefficient highlights:")
            for coef in top_coeffs:
                lines.append(
                    f"    * {coef['feature']}: coef {coef['coef']:+.4f}, odds {coef['odds_ratio']:.3f}"
                )

    if tree_summary.get("available"):
        lines.append(
            f"Decision tree accuracy: {tree_summary['accuracy'] * 100:.2f}%"
            f" (depth {tree_summary['depth']}, leaves {tree_summary['n_leaves']})."
        )
        top_importances = sorted(
            tree_summary.get("feature_importances", []),
            key=lambda item: item.get("importance", 0.0),
            reverse=True,
        )[:3]
        if top_importances:
            lines.append("  Importance highlights:")
            for imp in top_importances:
                lines.append(
                    f"    * {imp['feature']}: importance {imp['importance']:.3f}"
                )

    for label, combos in (("Pair", combo_summary.get("pairs", [])), ("Triplet", combo_summary.get("triplets", []))):
        if combos:
            best = combos[0]
            lines.append(
                f"Top {label.lower()} combo: {', '.join(best['features'])}"
                f" | accuracy {best['accuracy'] * 100:.2f}% (lift {best['lift'] * 100:.2f}pp, baseline {best['baseline'] * 100:.2f}%)."
            )

            if len(combos) > 1:
                others = [c for c in combos[1:3]]
                if others:
                    lines.append(f"  Next {label.lower()} contenders:")
                    for contender in others:
                        lines.append(
                            "    * "
                            + f"{', '.join(contender['features'])}: {contender['accuracy'] * 100:.2f}%"
                            + f" (lift {contender['lift'] * 100:.2f}pp)"
                        )

    if top_winners:
        lines.append("Top winner patterns:")
        for p in top_winners:
            wr = p.get("winrate_pct")
            lift = p.get("lift_pp")
            wr_str = f"{wr:.2f}%" if isinstance(wr, (int, float)) else "n/a"
            lift_str = f"{lift:+.2f}pp" if isinstance(lift, (int, float)) else "n/a"
            lines.append(f"  - {p['rule']} (win rate {wr_str} | lift {lift_str})")

    if top_losers:
        lines.append("Top loser patterns:")
        for p in top_losers:
            lr = p.get("lossrate_pct")
            lift = p.get("lift_pp")
            lr_str = f"{lr:.2f}%" if isinstance(lr, (int, float)) else "n/a"
            lift_str = f"{lift:+.2f}pp" if isinstance(lift, (int, float)) else "n/a"
            lines.append(f"  - {p['rule']} (loss rate {lr_str} | lift {lift_str})")

    if top_indicators:
        lines.append("Indicator mean differences (winner - loser):")
        for d in top_indicators:
            diff = d.get("mean_diff")
            diff_str = f"{diff:+.6f}" if isinstance(diff, (int, float)) else "n/a"
            lines.append(f"  - {d['feature']}: {diff_str}")

    lines.append("Overall takeaway: Watch listed patterns and indicator shifts for trade filtering.")

    with open(OUTPUT_SUMMARY_TXT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    # 7) Console summary
    print(f"[OK] Report created: {OUTPUT_HTML}")
    print(f"     Trades: {total_trades} | Winners: {n_w} | Losers: {n_l} | Baseline WR: {baseline_wr:.2f}%")
    if missing_pairs:
        print(f"     Warning: missing indicators for {len(missing_pairs)} pair(s)")
    print(f"     Summary text: {OUTPUT_SUMMARY_TXT}")
    if not _read_local_chartjs():
        print("     Note: Chart.js loads from the CDN (https://cdn.jsdelivr.net/npm/chart.js). "
              "Add a local copy to embed it offline:")
        for c in LOCAL_CHARTJS_CANDIDATES:
            print(f"           - {c}")

if __name__ == "__main__":
    main()
