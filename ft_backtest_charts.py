#!/usr/bin/env python3
"""
ft_backtest_charts.py

Generate charts from Freqtrade backtest results and export a single HTML
dashboard using Chart.js (loaded from CDN).

Usage:
  python user_data/documentation/ft_backtest_charts.py \
      --use-last \
      --results-dir user_data/backtest_results \
      --outdir user_data/plot/charts_last
"""

import argparse
import json
import zipfile
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import timezone


# -----------------------------
# Load helpers
# -----------------------------

def _load_last_result(results_dir: Path) -> Path:
    """Find .last_result.json which points to latest backtest zip."""
    lastfile = results_dir / ".last_result.json"
    if not lastfile.exists():
        raise FileNotFoundError(f"{lastfile} not found")

    with open(lastfile, "r", encoding="utf-8") as f:
        info = json.load(f)

    zipname = info.get("latest_backtest")
    if not zipname:
        raise ValueError("No latest_backtest in .last_result.json")

    zippath = results_dir / zipname
    if not zippath.exists():
        raise FileNotFoundError(f"Backtest zip {zippath} not found")

    return zippath


def _load_backtest_json(backtest_zip: Path) -> dict:
    """Open backtest zip and return main JSON dict."""
    with zipfile.ZipFile(backtest_zip, "r") as zf:
        # main json is the one without suffixes like _config, _Strategy
        mains = [n for n in zf.namelist()
                 if n.endswith(".json")
                 and "_config" not in n
                 and "_Strategy" not in n]
        if not mains:
            raise FileNotFoundError("No main backtest JSON inside zip.")
        with zf.open(mains[0]) as f:
            data = json.load(f)
    print(f"[info] Loaded JSON from zip member: {mains[0]}")
    return data


# -----------------------------
# Trade parsing
# -----------------------------

def _parse_trades(data: dict) -> pd.DataFrame:
    """Extract trades DataFrame from freqtrade backtest JSON."""
    # 1) Try top-level
    trades = data.get('trades', [])

    # 2) Try results.trades
    if not isinstance(trades, list) or len(trades) == 0:
        results = data.get('results') or {}
        if isinstance(results, dict) and isinstance(results.get('trades'), list):
            trades = results['trades']

    # 3) Try nested under strategy
    if not trades:
        strat = data.get('strategy')
        if isinstance(strat, dict) and strat:
            for _name, strat_blob in strat.items():
                if isinstance(strat_blob, dict) and isinstance(strat_blob.get('trades'), list):
                    trades = strat_blob['trades']
                    break

    if not trades:
        raise ValueError("No trades found. Export backtest with '--export trades'.")

    df = pd.json_normalize(trades)

    # Convert timestamps to UTC ISO strings for Chart.js
    def to_dt(series):
        v = series.dropna()
        # If numeric, guess unit
        if not v.empty and pd.api.types.is_numeric_dtype(series.dtype):
            sample = float(v.iloc[0])
            fmt = 'ms' if sample > 1e12 else 's'
            return pd.to_datetime(series, unit=fmt, utc=True)
        return pd.to_datetime(series, utc=True, errors='coerce')

    open_col  = next((c for c in ['open_date','open_time','open_timestamp'] if c in df.columns), None)
    close_col = next((c for c in ['close_date','close_time','close_timestamp'] if c in df.columns), None)
    if open_col:  df['open_dt']  = to_dt(df[open_col])
    if close_col: df['close_dt'] = to_dt(df[close_col])

    # Profit columns
    profit_abs_col   = next((c for c in ['profit_abs','profit_abs_usdc','profit_abs_usdt','profit','profit_abs_quote']
                             if c in df.columns), None)
    profit_ratio_col = next((c for c in ['profit_ratio','profit_pct','profit_percent','profit_ratio_pct']
                             if c in df.columns), None)

    if profit_abs_col and profit_abs_col != 'profit_abs':
        df = df.rename(columns={profit_abs_col: 'profit_abs'})
    elif 'profit_abs' not in df.columns and profit_ratio_col and 'stake_amount' in df.columns:
        df['profit_abs'] = df[profit_ratio_col].astype(float) * df['stake_amount'].astype(float)
    else:
        df['profit_abs'] = df.get('profit_abs', 0.0)

    if profit_ratio_col and profit_ratio_col != 'profit_ratio':
        df = df.rename(columns={profit_ratio_col: 'profit_ratio'})
    elif 'profit_ratio' not in df.columns:
        if 'stake_amount' in df.columns:
            df['profit_ratio'] = df['profit_abs'].astype(float) / df['stake_amount'].replace(0, pd.NA).astype(float)
        else:
            df['profit_ratio'] = 0.0

    # Pair
    if 'pair' not in df.columns:
        alt = next((c for c in ['pair_name','symbol','market'] if c in df.columns), None)
        if alt:
            df = df.rename(columns={alt: 'pair'})
        else:
            df['pair'] = 'UNKNOWN/QUOTE'

    # Sort
    if 'close_dt' in df.columns:
        df = df.sort_values('close_dt').reset_index(drop=True)
    elif 'open_dt' in df.columns:
        df = df.sort_values('open_dt').reset_index(drop=True)

    return df


# -----------------------------
# Data assembly for Chart.js
# -----------------------------

def _safe_iso(ts: pd.Timestamp) -> str:
    if pd.isna(ts):
        return None
    # Ensure timezone-aware ISO string (Chart.js time scale likes ISO 8601)
    if ts.tzinfo is None:
        ts = ts.tz_localize(timezone.utc)
    return ts.isoformat()


def build_chart_payloads(df: pd.DataFrame) -> dict:
    payload = {}

    # Equity curve (by close_dt)
    eq = df.set_index('close_dt')['profit_abs'].cumsum().dropna()
    payload['equity_curve'] = [
        {'x': _safe_iso(idx), 'y': float(val)} for idx, val in eq.items()
    ]

    # Drawdown
    roll_max = eq.cummax()
    dd = eq - roll_max
    payload['drawdown'] = [
        {'x': _safe_iso(idx), 'y': float(val)} for idx, val in dd.items()
    ]

    # Daily PnL (date labels)
    if 'close_dt' in df.columns:
        daily = df.groupby(df['close_dt'].dt.date)['profit_abs'].sum()
    else:
        daily = pd.Series(dtype=float)
    payload['daily_pnl'] = {
        'labels': [str(d) for d in daily.index],
        'values': [float(v) for v in daily.values],
    }

    # PnL by pair
    bypair = df.groupby('pair')['profit_abs'].sum().sort_values()
    payload['pnl_by_pair'] = {
        'labels': list(bypair.index),
        'values': [float(v) for v in bypair.values],
    }

    # Winrate by pair
    wr = df.groupby('pair')['profit_ratio'].apply(lambda x: float((x > 0).mean()))
    wr = wr.sort_values()
    payload['winrate_by_pair'] = {
        'labels': list(wr.index),
        'values': [float(v) for v in wr.values],
    }

    # Profit ratio histogram (pre-binned in Python)
    # Compute histogram without plotting:
    counts, edges = np.histogram(df['profit_ratio'].astype(float).dropna(), bins=50)
    # Center labels:
    centers = 0.5 * (edges[:-1] + edges[1:])
    payload['returns_hist'] = {
        'labels': [round(float(c), 4) for c in centers],
        'values': [int(v) for v in counts],
    }

    # Cleanup accidental plt usage
    import matplotlib.pyplot as plt
    plt.close('all')

    return payload


# -----------------------------
# HTML writer (Chart.js)
# -----------------------------

HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Freqtrade Backtest Charts</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="preconnect" href="https://cdn.jsdelivr.net">
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
<style>
  :root { --bg:#0f172a; --panel:#111827; --text:#e5e7eb; --grid:#374151; --accent:#22d3ee; }
  body { background:var(--bg); color:var(--text); font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,'Helvetica Neue',Arial;}
  .wrap { max-width:1200px; margin:24px auto; padding:0 16px; }
  h1 { font-size:22px; margin:8px 0 16px; }
  .grid { display:grid; grid-template-columns:1fr; gap:18px; }
  @media(min-width:900px){ .grid{grid-template-columns:1fr ;} }
  .card { background:var(--panel); border-radius:16px; padding:16px; box-shadow:0 0 0 1px #1f2937; }
  canvas { width:100%; height:360px; }
  .footer { opacity:.7; font-size:12px; margin:24px 0 8px; }
</style>
</head>
<body>
  <div class="wrap">
    <h1>Freqtrade Backtest Charts</h1>
    <div class="grid">
      <div class="card"><h3>Equity Curve</h3><canvas id="equity"></canvas></div>
      <div class="card"><h3>Drawdown</h3><canvas id="drawdown"></canvas></div>
      <div class="card"><h3>Daily Profit</h3><canvas id="daily"></canvas></div>
      <div class="card"><h3>PnL by Pair</h3><canvas id="pairpnl"></canvas></div>
      <div class="card"><h3>Winrate by Pair</h3><canvas id="winrate"></canvas></div>
      <div class="card"><h3>Profit Ratio Histogram</h3><canvas id="hist"></canvas></div>
    </div>
    <div class="footer">Generated with Chart.js (CDN)</div>
  </div>

<script>
const PAYLOAD = __PAYLOAD__;

function asLineChart(el, dataPoints, label) {
  return new Chart(el, {
    type: 'line',
    data: {
      datasets: [{
        label: label,
        data: dataPoints,
        fill: false,
        tension: 0.15
      }]
    },
    options: {
      responsive: true,
      interaction: { mode: 'index', intersect: false },
      scales: {
        x: {
          type: 'time',
          time: { tooltipFormat: 'yyyy-MM-dd HH:mm' }
        },
        y: { beginAtZero: false }
      },
      plugins: {
        legend: { display: false }
      }
    }
  });
}

function asBarChart(el, labels, values, horizontal=false, label='Value') {
  return new Chart(el, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{ label: label, data: values }]
    },
    options: {
      indexAxis: horizontal ? 'y' : 'x',
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { beginAtZero: true },
        y: { beginAtZero: true }
      }
    }
  });
}

window.addEventListener('DOMContentLoaded', () => {
  // Line charts
  asLineChart(document.getElementById('equity'), PAYLOAD.equity_curve, 'Equity');
  asLineChart(document.getElementById('drawdown'), PAYLOAD.drawdown, 'Drawdown');

  // Bars
  asBarChart(document.getElementById('daily'), PAYLOAD.daily_pnl.labels, PAYLOAD.daily_pnl.values, false, 'USDC');
  asBarChart(document.getElementById('pairpnl'), PAYLOAD.pnl_by_pair.labels, PAYLOAD.pnl_by_pair.values, true, 'USDC');
  asBarChart(document.getElementById('winrate'), PAYLOAD.winrate_by_pair.labels, PAYLOAD.winrate_by_pair.values, true, 'Winrate');

  // Histogram
  asBarChart(document.getElementById('hist'), PAYLOAD.returns_hist.labels, PAYLOAD.returns_hist.values, false, 'Count');
});
</script>
</body>
</html>
"""

def write_html_report(payload: dict, outdir: Path, filename: str = "report.html") -> Path:
    outpath = outdir / filename
    html = HTML_TEMPLATE.replace("__PAYLOAD__", json.dumps(payload))
    outpath.write_text(html, encoding="utf-8")
    return outpath


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-last", action="store_true", help="Use latest backtest from .last_result.json")
    parser.add_argument("--results-dir", type=str, default="user_data/backtest_results")
    parser.add_argument("--outdir", type=str, default="user_data/plot/charts_last")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not args.use_last:
        raise ValueError("Please use --use-last (only mode implemented).")

    backtest_zip = _load_last_result(results_dir)
    print(f"[info] Using last backtest: {backtest_zip}")

    data = _load_backtest_json(backtest_zip)
    df = _parse_trades(data)

    payload = build_chart_payloads(df)
    report_path = write_html_report(payload, outdir)
    print(f"[info] HTML report saved to {report_path}")

if __name__ == "__main__":
    main()
