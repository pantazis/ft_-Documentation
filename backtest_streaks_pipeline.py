import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------
# Utilities / IO
# ---------------------------

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
PLOT_DIR = os.path.join(BASE_DIR, "plot", "charts_last")
BT_RESULTS_DIR = os.path.join(BASE_DIR, "backtest_results")
CANDLES_ROOT = os.path.join(BASE_DIR, "data", "binance")


def load_last_backtest_zip() -> str:
    last_ptr = os.path.join(BT_RESULTS_DIR, ".last_result.json")
    with open(last_ptr, "r", encoding="utf-8") as fh:
        last = json.load(fh)
    zipname = last["latest_backtest"]
    return os.path.join(BT_RESULTS_DIR, zipname)


def read_from_zip(zip_path: str, inner_filename: str) -> bytes:
    import zipfile

    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(inner_filename) as fh:
            return fh.read()


def find_main_json_name(zip_path: str) -> str:
    import zipfile

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
    # Prefer main summary json (not config / not strategy-specific dump)
    main = [
        n
        for n in names
        if n.endswith(".json")
        and "_config" not in n
        and "_Strategy" not in n
        and "_DonchianEmaBreakout" not in n
    ]
    if main:
        return main[0]
    # Fallback: first json
    for n in names:
        if n.endswith(".json"):
            return n
    raise FileNotFoundError("No JSON found in backtest zip")


def ensure_plot_dir():
    os.makedirs(PLOT_DIR, exist_ok=True)


def detect_timeframe(prefer: Optional[str] = None) -> str:
    # Try to infer timeframe from available candle files, prefer 15m
    if prefer:
        return prefer
    # Common timeframes to try
    tf_order = ["15m", "5m", "1h", "30m", "1m", "2h", "4h", "1d"]
    # If any file with -15m exists, select it
    for tf in tf_order:
        for fn in os.listdir(CANDLES_ROOT):
            if fn.endswith(f"-{tf}.feather") or fn.endswith(f"-{tf}.json") or fn.endswith(f"-{tf}.json.gz"):
                return tf
    # Default to 15m
    return "15m"


# ---------------------------
# Indicator calculations (pure pandas)
# ---------------------------

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False, min_periods=length).mean()


def rsi_wilder(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    roll_down = down.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    rs = roll_up / roll_down.replace({0: np.nan})
    rsi = 100 - (100 / (1 + rs))
    return rsi


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr


def atr_wilder(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    atr = tr.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    return atr


def adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    tr = true_range(high, low, close)
    atr = tr.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / length, adjust=False, min_periods=length).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / length, adjust=False, min_periods=length).mean() / atr)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    adx_val = dx.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    return adx_val


def donchian_position(close: pd.Series, high: pd.Series, low: pd.Series, window: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
    dc_high = high.rolling(window).max()
    dc_low = low.rolling(window).min()
    width = dc_high - dc_low
    pos = (close - dc_low) / (width.replace(0, np.nan))
    return pos, dc_high, dc_low


def zscore(series: pd.Series, window: int = 30) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std(ddof=0)
    return (series - mean) / std.replace(0, np.nan)


def ema_slope_fraction(ema_series: pd.Series, lookback: int = 14) -> pd.Series:
    prev = ema_series.shift(lookback)
    return (ema_series - prev) / (lookback * ema_series.replace(0, np.nan))


# ---------------------------
# Candles loading
# ---------------------------

def pair_to_filename(pair: str, timeframe: str) -> Optional[str]:
    base = pair.replace("/", "_") + f"-{timeframe}"
    candidates = [
        os.path.join(CANDLES_ROOT, base + ".feather"),
        os.path.join(CANDLES_ROOT, base + ".json"),
        os.path.join(CANDLES_ROOT, base + ".json.gz"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def read_candles(path: str) -> pd.DataFrame:
    if path.endswith(".feather"):
        try:
            df = pd.read_feather(path)
        except Exception as e:
            raise RuntimeError(f"Failed to read feather {path}: {e}")
    elif path.endswith(".json") or path.endswith(".json.gz"):
        try:
            df = pd.read_json(path, lines=False)
            if isinstance(df, pd.DataFrame) and df.shape[0] == 0:
                df = pd.read_json(path, lines=True)
        except ValueError:
            df = pd.read_json(path, lines=True)
    else:
        raise ValueError(f"Unsupported candle file: {path}")

    rename_map = {}
    if "date" in df.columns:
        pass
    elif "timestamp" in df.columns:
        rename_map["timestamp"] = "date"
    elif "time" in df.columns:
        rename_map["time"] = "date"
    df = df.rename(columns=rename_map)

    for c in ["open", "high", "low", "close", "volume"]:
        if c not in df.columns:
            if c == "volume" and "vol" in df.columns:
                df = df.rename(columns={"vol": "volume"})
            else:
                raise ValueError(f"Missing column {c} in {path}")

    if not np.issubdtype(df["date"].dtype, np.datetime64):
        try:
            df["date"] = pd.to_datetime(df["date"], utc=True)
        except Exception:
            df["date"] = pd.to_datetime(df["date"], unit="ms", utc=True)
    else:
        if df["date"].dt.tz is None:
            df["date"] = df["date"].dt.tz_localize("UTC")

    df = df.sort_values("date").reset_index(drop=True)
    return df


def timeframe_to_tdelta(tf: str) -> pd.Timedelta:
    unit = tf[-1]
    val = int(tf[:-1])
    if unit == "m":
        return pd.Timedelta(minutes=val)
    if unit == "h":
        return pd.Timedelta(hours=val)
    if unit == "d":
        return pd.Timedelta(days=val)
    raise ValueError(f"Unsupported timeframe: {tf}")


# ---------------------------
# Equity streak segmentation
# ---------------------------


@dataclass
class Segment:
    start_idx: int
    end_idx: int
    direction: str  # 'up' or 'down'


def zigzag_segments(equity: pd.Series, eps_frac: float = 0.001, min_len: int = 2) -> List[Segment]:
    if equity.empty:
        return []
    e = equity.values.astype(float)
    n = len(e)
    e_min, e_max = float(np.nanmin(e)), float(np.nanmax(e))
    rng = max(e_max - e_min, 1e-9)
    eps = eps_frac * rng

    pivots: List[int] = [0]
    last_pivot = 0
    direction = 0  # 0 unknown, 1 up, -1 down
    extreme_val = e[0]
    extreme_idx = 0
    for i in range(1, n):
        diff = e[i] - extreme_val
        # looking for up or undecided
        if direction >= 0:
            if e[i] >= extreme_val:
                extreme_val = e[i]
                extreme_idx = i
            if (extreme_val - e[i]) > eps and extreme_idx != last_pivot:
                pivots.append(extreme_idx)
                last_pivot = extreme_idx
                direction = -1
                extreme_val = e[i]
                extreme_idx = i
        # looking for down or undecided
        if direction <= 0:
            if e[i] <= extreme_val:
                extreme_val = e[i]
                extreme_idx = i
            if (e[i] - extreme_val) > eps and extreme_idx != last_pivot:
                pivots.append(extreme_idx)
                last_pivot = extreme_idx
                direction = 1
                extreme_val = e[i]
                extreme_idx = i

    if pivots[-1] != n - 1:
        pivots.append(n - 1)

    segs: List[Segment] = []
    for a, b in zip(pivots[:-1], pivots[1:]):
        if b <= a:
            continue
        direction = 'up' if e[b] >= e[a] else 'down'
        segs.append(Segment(a, b, direction))

    merged: List[Segment] = []
    for seg in segs:
        if not merged:
            merged.append(seg)
            continue
        if (seg.end_idx - seg.start_idx + 1) < min_len:
            prev = merged[-1]
            merged[-1] = Segment(prev.start_idx, seg.end_idx, prev.direction)
        else:
            merged.append(seg)

    return merged


def label_streaks_by_close(trades_df: pd.DataFrame, eps_frac: float = 0.001) -> pd.Series:
    equity = trades_df['profit_abs'].cumsum()
    segs = zigzag_segments(equity, eps_frac=eps_frac, min_len=2)
    labels = np.array(['neutral'] * len(trades_df), dtype=object)
    for seg in segs:
        if seg.direction == 'up':
            labels[seg.start_idx:seg.end_idx + 1] = 'winning_streak'
        else:
            labels[seg.start_idx:seg.end_idx + 1] = 'losing_streak'
    return pd.Series(labels, index=trades_df.index)


# ---------------------------
# Feature reconstruction per trade at entry
# ---------------------------


def compute_features_for_pair(df: pd.DataFrame) -> pd.DataFrame:
    ema_len = 100
    ema_series = ema(df['close'], ema_len)
    ema_sl = ema_slope_fraction(ema_series, lookback=14)

    don_pos, don_high, don_low = donchian_position(df['close'], df['high'], df['low'], window=20)
    don_width = (don_high - don_low) / df['close']

    atr = atr_wilder(df['high'], df['low'], df['close'], length=14)
    atr_norm = atr / df['close']

    rsi_val = rsi_wilder(df['close'], length=14)
    adx_val = adx(df['high'], df['low'], df['close'], length=14)

    ema_dist = (df['close'] - ema_series) / ema_series.replace(0, np.nan)
    vol_z = zscore(df['volume'], window=30)

    feats = pd.DataFrame({
        'ema_slope': ema_sl,
        'don_pos': don_pos,
        'atr_norm': atr_norm,
        'rsi': rsi_val,
        'adx': adx_val,
        'don_width': don_width,
        'ema_dist': ema_dist,
        'vol_z': vol_z,
    }, index=df.index)
    feats['date'] = df['date']
    return feats


def get_entry_feature_row(feats: pd.DataFrame, open_dt: pd.Timestamp, timeframe: str) -> Optional[pd.Series]:
    delta = timeframe_to_tdelta(timeframe)
    eligible = feats[feats['date'] + delta <= open_dt]
    if eligible.empty:
        return None
    return eligible.iloc[-1]


# ---------------------------
# Hyperopt (feature screening)
# ---------------------------


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def rates(cm: Dict[str, int]) -> Dict[str, float]:
    tp, tn, fp, fn = cm["tp"], cm["tn"], cm["fp"], cm["fn"]
    pos = tp + fn
    neg = tn + fp
    fnr = fn / pos if pos > 0 else 0.0
    fpr = fp / neg if neg > 0 else 0.0
    tpr = tp / pos if pos > 0 else 0.0
    tnr = tn / neg if neg > 0 else 0.0
    return {"FNR": fnr, "FPR": fpr, "TPR": tpr, "TNR": tnr}


def eval_rule_on_dataset(
    df: pd.DataFrame,
    rule_mask: np.ndarray,
    baseline_wins: int,
    baseline_winrate: float,
    baseline_win_streak_wins: int,
    strict_constraints: bool = True,
) -> Dict[str, Any]:
    y_true = (df['is_losing_streak_trade'].astype(int).values)
    y_pred = (rule_mask.astype(int))
    cm_all = confusion(y_true, y_pred)
    metrics_all = rates(cm_all)

    ws = (df['streak_flag'] == 'winning_streak').values
    if ws.any():
        cm_ws = confusion((y_true[ws]), (y_pred[ws]))
        metrics_ws = rates(cm_ws)
    else:
        cm_ws = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
        metrics_ws = {"FNR": 0.0, "FPR": 0.0, "TPR": 0.0, "TNR": 0.0}

    keep = ~rule_mask
    kept_df = df[keep]
    wins_after = int((kept_df['winner'] == True).sum())
    total_after = int(len(kept_df))
    winrate_after = (wins_after / total_after) if total_after > 0 else 0.0

    wins_ws_after = int(((kept_df['streak_flag'] == 'winning_streak') & (kept_df['winner'] == True)).sum())

    delta_wins = wins_after - baseline_wins
    delta_winrate = winrate_after - baseline_winrate
    delta_ws_wins = wins_ws_after - baseline_win_streak_wins

    ok = True
    if baseline_win_streak_wins > 0:
        if wins_ws_after < 0.98 * baseline_win_streak_wins:
            ok = False
    if baseline_wins > 0:
        if wins_after < 0.98 * baseline_wins:
            ok = False

    loss = (
        1.00 * metrics_all['FNR']
        + 0.75 * metrics_ws['FPR']
        + 0.25 * (abs(min(0.0, delta_wins)) / (baseline_wins if baseline_wins > 0 else 1))
        + 0.25 * abs(min(0.0, delta_winrate))
    )

    return {
        "ok": ok if strict_constraints else True,
        "loss": float(loss),
        "cm_all": cm_all,
        "metrics_all": metrics_all,
        "cm_ws": cm_ws,
        "metrics_ws": metrics_ws,
        "wins_after": wins_after,
        "total_after": total_after,
        "winrate_after": winrate_after,
        "delta_wins": delta_wins,
        "delta_winrate": delta_winrate,
        "delta_ws_wins": delta_ws_wins,
    }


def quantile_thresholds(series: pd.Series, q_step: float = 0.05) -> List[float]:
    qs = np.arange(0.05, 0.96, q_step)
    vals = series.dropna().quantile(qs).unique()
    return [float(v) for v in vals if np.isfinite(v)]


def search_single_rules(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    features: List[str],
    baseline: Dict[str, float],
    seed: int = 42,
) -> List[Dict[str, Any]]:
    random.seed(seed)
    results: List[Dict[str, Any]] = []
    for feat in features:
        s_train = df_train[feat]
        s_val = df_val[feat]
        if s_train.dropna().empty:
            continue
        ths = quantile_thresholds(s_train)
        for t in ths:
            for mode in ('gt', 'lt'):
                if mode == 'gt':
                    rule_mask_val = (s_val.values > t)
                else:
                    rule_mask_val = (s_val.values < t)
                res = eval_rule_on_dataset(
                    df_val,
                    rule_mask_val,
                    baseline_wins=int(baseline['wins']),
                    baseline_winrate=float(baseline['winrate']),
                    baseline_win_streak_wins=int(baseline['win_streak_wins']),
                    strict_constraints=True,
                )
                if res['ok']:
                    results.append(
                        {
                            "type": "single",
                            "feature": feat,
                            "mode": mode,
                            "threshold": float(t),
                            **res,
                        }
                    )
    results.sort(key=lambda x: x['loss'])
    return results


def search_combo_rules(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    top_singles: List[Dict[str, Any]],
    k_features: int,
    combo_size: int,
    max_trials: int,
    baseline: Dict[str, float],
    seed: int = 42,
) -> List[Dict[str, Any]]:
    random.seed(seed)
    ordered_feats: List[str] = []
    for r in top_singles:
        f = r['feature']
        if f not in ordered_feats:
            ordered_feats.append(f)
        if len(ordered_feats) >= k_features:
            break

    feat_thresholds: Dict[str, List[Tuple[str, float]]] = {}
    for f in ordered_feats:
        s = df_train[f]
        if s.dropna().empty:
            continue
        ths = quantile_thresholds(s)
        if len(ths) > 8:
            idx = np.linspace(0, len(ths) - 1, 8).astype(int)
            ths = [ths[i] for i in idx]
        feat_thresholds[f] = [("gt", t) for t in ths] + [("lt", t) for t in ths]

    feats = [f for f in ordered_feats if f in feat_thresholds]
    if len(feats) < combo_size:
        return []

    results: List[Dict[str, Any]] = []
    ops = ['AND', 'OR']
    trials = 0
    rng = random.Random(seed)

    while trials < max_trials:
        chosen = rng.sample(feats, combo_size)
        chosen_specs: List[Tuple[str, str, float]] = []
        for f in chosen:
            mode, thr = rng.choice(feat_thresholds[f])
            chosen_specs.append((f, mode, thr))
        op = rng.choice(ops)

        masks = []
        for f, mode, thr in chosen_specs:
            s_val = df_val[f].values
            if mode == 'gt':
                masks.append(s_val > thr)
            else:
                masks.append(s_val < thr)
        if op == 'AND':
            rule_mask_val = np.logical_and.reduce(masks)
        else:
            rule_mask_val = np.logical_or.reduce(masks)

        res = eval_rule_on_dataset(
            df_val,
            rule_mask_val,
            baseline_wins=int(baseline['wins']),
            baseline_winrate=float(baseline['winrate']),
            baseline_win_streak_wins=int(baseline['win_streak_wins']),
            strict_constraints=True,
        )
        if res['ok']:
            results.append(
                {
                    "type": f"combo_{combo_size}",
                    "features": [
                        {"feature": f, "mode": m, "threshold": float(t)}
                        for f, m, t in chosen_specs
                    ],
                    "op": op,
                    **res,
                }
            )
        trials += 1

    results.sort(key=lambda x: x['loss'])
    return results


# ---------------------------
# Main pipeline
# ---------------------------


def run_pipeline(random_seed: int = 17):
    random.seed(random_seed)
    np.random.seed(random_seed)
    ensure_plot_dir()

    zip_path = load_last_backtest_zip()
    main_json_name = find_main_json_name(zip_path)
    main = json.loads(read_from_zip(zip_path, main_json_name).decode('utf-8'))

    if "strategy" in main and isinstance(main["strategy"], dict):
        strat_name = next(iter(main["strategy"]))
        trades = main["strategy"][strat_name]["trades"]
    elif "trades" in main:
        trades = main["trades"]
    else:
        raise RuntimeError("Could not locate trades in backtest JSON")

    tf = detect_timeframe(prefer="15m")

    tdf = pd.DataFrame(trades)
    if 'close_timestamp' in tdf.columns:
        tdf['close_dt'] = pd.to_datetime(tdf['close_timestamp'], unit='ms', utc=True)
    else:
        tdf['close_dt'] = pd.to_datetime(tdf['close_date'], utc=True)
    if 'open_timestamp' in tdf.columns:
        tdf['open_dt'] = pd.to_datetime(tdf['open_timestamp'], unit='ms', utc=True)
    else:
        tdf['open_dt'] = pd.to_datetime(tdf['open_date'], utc=True)

    tdf = tdf.sort_values('close_dt').reset_index(drop=True)

    tdf['profit_abs'] = tdf['profit_abs'].astype(float)
    tdf['profit_ratio'] = tdf['profit_ratio'].astype(float)
    tdf['winner'] = tdf['profit_ratio'] > 0
    tdf['streak_flag'] = label_streaks_by_close(tdf, eps_frac=0.001)

    pairs = sorted(tdf['pair'].dropna().unique().tolist())
    pair_feat_map: Dict[str, pd.DataFrame] = {}
    warnings: List[str] = []

    for pair in pairs:
        path = pair_to_filename(pair, tf)
        if not path:
            warnings.append(f"Missing candles for {pair} ({tf}) - skipping features.")
            continue
        try:
            cdf = read_candles(path)
        except Exception as e:
            warnings.append(f"Failed to read candles for {pair}: {e}")
            continue
        feats = compute_features_for_pair(cdf)
        pair_feat_map[pair] = feats

    feat_cols = ['ema_slope', 'don_pos', 'atr_norm', 'rsi', 'adx', 'don_width', 'ema_dist', 'vol_z']
    feat_data = {c: [] for c in feat_cols}
    holding_s_list: List[Optional[float]] = []
    hour_list: List[int] = []
    weekday_list: List[int] = []

    for _, row in tdf.iterrows():
        pair = row['pair']
        open_dt = row['open_dt']
        feats = pair_feat_map.get(pair)
        if feats is None:
            for c in feat_cols:
                feat_data[c].append(np.nan)
            holding_s_list.append(float((row['close_dt'] - row['open_dt']).total_seconds()))
            hour_list.append(int(open_dt.hour))
            weekday_list.append(int(open_dt.weekday()))
            continue
        frow = get_entry_feature_row(feats, open_dt, tf)
        if frow is None:
            for c in feat_cols:
                feat_data[c].append(np.nan)
        else:
            for c in feat_cols:
                val = frow.get(c)
                feat_data[c].append(float(val) if pd.notna(val) else np.nan)
        holding_s_list.append(float((row['close_dt'] - row['open_dt']).total_seconds()))
        hour_list.append(int(open_dt.hour))
        weekday_list.append(int(open_dt.weekday()))

    for c in feat_cols:
        tdf[c] = feat_data[c]
    tdf['holding_s'] = holding_s_list
    tdf['hour'] = hour_list
    tdf['weekday'] = weekday_list

    enriched_records: List[Dict[str, Any]] = []
    for _, r in tdf.iterrows():
        enriched_records.append(
            {
                "pair": r['pair'],
                "open_dt": pd.to_datetime(r['open_dt']).tz_convert('UTC').isoformat().replace('+00:00', 'Z'),
                "close_dt": pd.to_datetime(r['close_dt']).tz_convert('UTC').isoformat().replace('+00:00', 'Z'),
                "profit_abs": float(r['profit_abs']),
                "profit_ratio": float(r['profit_ratio']),
                "winner": bool(r['winner']),
                "streak_flag": str(r['streak_flag']),
                "features": {
                    "ema_slope": _clean_float(r.get('ema_slope')),
                    "don_pos": _clean_float(r.get('don_pos')),
                    "atr_norm": _clean_float(r.get('atr_norm')),
                    "rsi": _clean_float(r.get('rsi')),
                    "adx": _clean_float(r.get('adx')),
                    "don_width": _clean_float(r.get('don_width')),
                    "ema_dist": _clean_float(r.get('ema_dist')),
                    "vol_z": _clean_float(r.get('vol_z')),
                    "holding_s": _safe_seconds(r.get('holding_s')),
                    "hour": int(r.get('hour')) if pd.notna(r.get('hour')) else None,
                    "weekday": int(r.get('weekday')) if pd.notna(r.get('weekday')) else None,
                },
            }
        )

    enriched_path = os.path.join(PLOT_DIR, "Backtest_Streaks_enriched.json")
    with open(enriched_path, "w", encoding="utf-8") as fh:
        json.dump(enriched_records, fh, ensure_ascii=False)

    df = tdf.copy()
    df['is_losing_streak_trade'] = (df['streak_flag'] == 'losing_streak')

    cutoff = df['close_dt'].quantile(0.7)
    df_train = df[df['close_dt'] <= cutoff].copy()
    df_val = df[df['close_dt'] > cutoff].copy()

    baseline = {
        'wins': int((df_val['winner'] == True).sum()),
        'winrate': float((df_val['winner'] == True).mean()) if len(df_val) else 0.0,
        'win_streak_wins': int(
            ((df_val['streak_flag'] == 'winning_streak') & (df_val['winner'] == True)).sum()
        ),
    }

    screen_features = [
        'ema_slope',
        'don_pos',
        'atr_norm',
        'rsi',
        'adx',
        'don_width',
        'ema_dist',
        'vol_z',
        'holding_s',
        'hour',
        'weekday',
    ]

    singles = search_single_rules(df_train, df_val, screen_features, baseline, seed=random_seed)
    top_singles = singles[:10]

    combos2: List[Dict[str, Any]] = []
    combos3: List[Dict[str, Any]] = []
    if top_singles:
        combos2 = search_combo_rules(
            df_train,
            df_val,
            top_singles,
            k_features=min(6, len(top_singles)),
            combo_size=2,
            max_trials=200,
            baseline=baseline,
            seed=random_seed,
        )
        combos3 = search_combo_rules(
            df_train,
            df_val,
            top_singles,
            k_features=min(6, len(top_singles)),
            combo_size=3,
            max_trials=300,
            baseline=baseline,
            seed=random_seed,
        )

    def summarize_best(results: List[Dict[str, Any]], topn: int = 3) -> List[Dict[str, Any]]:
        if not results:
            return []
        best_loss = results[0]['loss']
        kept: List[Dict[str, Any]] = []
        for r in results:
            if r['loss'] <= 1.01 * best_loss:
                kept.append(r)
            if len(kept) >= topn:
                break
        return kept

    best_single = summarize_best(singles, topn=3)
    best_two = summarize_best(combos2, topn=3)
    best_three = summarize_best(combos3, topn=3)

    report = {
        "baseline": baseline,
        "best_single": best_single,
        "best_two": best_two,
        "best_three": best_three,
        "notes": {
            "constraints": {
                "max_win_streak_wins_drop_pct": 2.0,
                "max_total_wins_drop_pct": 2.0,
            },
            "timeframe": tf,
            "seed": random_seed,
            "warnings": warnings,
        },
    }

    feat_json_path = os.path.join(PLOT_DIR, "Backtest_Streaks_feature_screen.json")
    with open(feat_json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    summary_lines: List[str] = []
    seg_counts = df['streak_flag'].value_counts().to_dict()
    summary_lines.append("Streak trade counts:")
    for k in ['winning_streak', 'losing_streak', 'neutral']:
        summary_lines.append(f"  {k}: {seg_counts.get(k, 0)}")
    summary_lines.append("")
    summary_lines.append(
        f"Validation baseline: wins={baseline['wins']}, winrate={baseline['winrate']:.3f}, winning-streak wins={baseline['win_streak_wins']}"
    )
    summary_lines.append("")

    def rule_to_str(r: Dict[str, Any]) -> str:
        if r['type'] == 'single':
            return f"{r['feature']} {r['mode']} {r['threshold']:.5f}"
        else:
            parts = [f"{f['feature']} {f['mode']} {f['threshold']:.5f}" for f in r['features']]
            return "(" + (f" {r['op']} ".join(parts)) + ")"

    def block(results: List[Dict[str, Any]], title: str):
        if not results:
            summary_lines.append(f"No {title} rules meeting constraints.")
            return
        summary_lines.append(f"Top {title} rules:")
        for r in results:
            summary_lines.append(
                f"- {rule_to_str(r)} | loss={r['loss']:.4f} | FNR={r['metrics_all']['FNR']:.3f} | FPR_ws={r['metrics_ws']['FPR']:.3f} | Δwins={r['delta_wins']} | Δwinrate={r['delta_winrate']:.3f}"
            )
        summary_lines.append("")

    block(best_single, "single-indicator")
    block(best_two, "2-indicator")
    block(best_three, "3-indicator")

    feat_txt_path = os.path.join(PLOT_DIR, "Backtest_Streaks_feature_screen.txt")
    with open(feat_txt_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(summary_lines))

    print("; ".join(summary_lines[:6]))


def _clean_float(x: Any) -> Optional[float]:
    try:
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            return None
        return float(x)
    except Exception:
        return None


def _safe_seconds(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


if __name__ == "__main__":
    run_pipeline()

