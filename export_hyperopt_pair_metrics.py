import json
import sys
import argparse
import shutil
from pathlib import Path
from datetime import datetime


def find_latest_hyperopt_file(base_dir: Path) -> Path:
    last_result = base_dir / ".last_result.json"
    if last_result.exists():
        try:
            latest_name = json.loads(last_result.read_text(encoding="utf-8")).get(
                "latest_hyperopt"
            )
            if latest_name:
                f = base_dir / latest_name
                if f.exists():
                    return f
        except Exception:
            pass

    # Fallback: newest .fthypt by mtime
    candidates = sorted(base_dir.glob("*.fthypt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError("No .fthypt files found in hyperopt_results")
    return candidates[0]


def load_fthypt(path: Path) -> dict:
    """Load freqtrade .fthypt file.
    Some versions write minified JSON, others write JSONL (one JSON per line).
    Return the last JSON object containing results_metrics, or the last object.
    """
    text = path.read_text(encoding="utf-8")
    # Try plain JSON first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: JSONL - parse line by line and keep the last dict
    last_obj = None
    last_with_metrics = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        last_obj = obj
        if isinstance(obj, dict) and obj.get("results_metrics"):
            last_with_metrics = obj

    if last_with_metrics is not None:
        return last_with_metrics
    if last_obj is not None:
        return last_obj

    raise json.JSONDecodeError("No valid JSON object found in fthypt file", text, 0)


def _parse_duration_minutes(duration_str: str):
    if not duration_str:
        return ""
    try:
        parts = duration_str.split(":")
        if len(parts) == 2:
            h, m = parts
            return int(h) * 60 + int(m)
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 60 + int(m) + int(s) / 60.0
    except Exception:
        return ""
    return ""


def export_metrics(results_metrics: dict, outdir: Path, run_id: str, mirror_dir: Path | None = None) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)

    per_pair = results_metrics.get("results_per_pair") or []
    # Create a combined CSV as well
    csv_path = outdir / f"{run_id}.csv"
    with csv_path.open("w", encoding="utf-8") as csv:
        csv.write(
            "pair,trades,profit_total,profit_total_abs,sharpe,r2,duration_avg,duration_avg_mins,max_drawdown_abs,max_drawdown_account\n"
        )
        for item in per_pair:
            pair = item.get("key") or item.get("pair") or "UNKNOWN"
            trades = item.get("trades", 0)
            profit_total = item.get("profit_total", 0.0)
            profit_total_abs = item.get("profit_total_abs", 0.0)
            sharpe = item.get("sharpe", 0.0)
            r2 = item.get("r2", "") or item.get("r_squared", "")
            duration_avg = item.get("duration_avg", "")
            duration_mins = _parse_duration_minutes(duration_avg) if isinstance(duration_avg, str) else ""
            mdd_abs = item.get("max_drawdown_abs", "")
            mdd_account = item.get("max_drawdown_account", "")

            # Write per-coin file
            safe_name = pair.replace("/", "_")
            fn = outdir / f"{safe_name}.txt"
            with fn.open("w", encoding="utf-8") as f:
                f.write(f"pair: {pair}\n")
                f.write(f"trades: {trades}\n")
                f.write(f"profit_total: {profit_total}\n")
                f.write(f"profit_total_abs: {profit_total_abs}\n")
                f.write(f"sharpe: {sharpe}\n")
                f.write(f"r2: {r2}\n")
                f.write(f"duration_avg: {duration_avg}\n")
                f.write(f"duration_avg_mins: {duration_mins}\n")
                f.write(f"max_drawdown_abs: {mdd_abs}\n")
                f.write(f"max_drawdown_account: {mdd_account}\n")

            # Optionally mirror to a flat by-pair directory
            if mirror_dir is not None:
                mirror_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(fn, mirror_dir / f"{safe_name}.txt")

            csv.write(
                f"{pair},{trades},{profit_total},{profit_total_abs},{sharpe},{r2},{duration_avg},{duration_mins},{mdd_abs},{mdd_account}\n"
            )

    return csv_path


def main():
    parser = argparse.ArgumentParser(description="Export per-pair profit and sharpe from latest hyperopt run")
    parser.add_argument("--fthypt", dest="fthypt", help="Path to .fthypt file (defaults to latest)")
    parser.add_argument("--outdir", dest="outdir", help="Output base directory (defaults to user_data/hyperopt_results/metrics)")
    parser.add_argument("--by-pair-dir", dest="by_pair_dir", help="Also copy all per-pair files into this flat folder")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    hopt_dir = repo_root / "user_data" / "hyperopt_results"

    if args.fthypt:
        hopt_file = Path(args.fthypt)
        if not hopt_file.exists():
            # allow relative to hyperopt_results
            candidate = hopt_dir / args.fthypt
            if candidate.exists():
                hopt_file = candidate
            else:
                print(f"Specified .fthypt not found: {args.fthypt}")
                sys.exit(2)
    else:
        try:
            hopt_file = find_latest_hyperopt_file(hopt_dir)
        except FileNotFoundError as e:
            print(str(e))
            sys.exit(2)

    try:
        data = load_fthypt(hopt_file)
    except json.JSONDecodeError as e:
        print(f"Failed to parse {hopt_file.name}: {e}")
        sys.exit(3)

    results_metrics = data.get("results_metrics") or {}
    if not results_metrics:
        print("No results_metrics found in hyperopt file (did hyperopt run with backtesting metrics?)")
        sys.exit(4)

    run_id = hopt_file.stem  # e.g., strategy_XYZ_YYYY-mm-dd_HH-MM-SS
    base_outdir = Path(args.outdir) if args.outdir else (hopt_dir / "metrics")
    outdir = base_outdir / run_id

    mirror_dir = Path(args.by_pair_dir) if args.by_pair_dir else None

    csv_path = export_metrics(results_metrics, outdir, run_id, mirror_dir)
    print(f"Wrote per-pair metrics to {outdir}")
    print(f"Combined CSV: {csv_path}")


if __name__ == "__main__":
    main()
