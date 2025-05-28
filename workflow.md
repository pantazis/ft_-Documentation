# Freqtrade Linux Workflow (using single `config.json`)

## 0. Research & Build Your Strategy

1. Watch trading strategy videos/tutorials on YouTube.
2. Export the video’s transcript (YouTube → ⋯ → "Show transcript" → copy).
3. Distill the logic into a Python strategy class under `user_data/strategies/` *(generate or refine boilerplate with **ChatGPT‑o3** if available).*
4. Test that the strategy runs with `--strategy-list`.

*(Skip if you already have a strategy file.)*

---

## 1. Prerequisites

- Docker & Docker Compose installed
- Freqtrade repo cloned to `~/freqtrade/`
- **One** config file at `~/freqtrade/user_data/config.json` containing API keys, exchange, pairs, stake, risk, etc.

---

## 2. Open a Terminal & Navigate

```bash
cd ft_userdata/user_data
```

---

## 3. Start Core Services

```bash
sudo docker compose up -d
```

This launches Redis, PostgreSQL, and the bot base image in detached mode.

---

## 4. Download Historical Candles

Set the timeframe you care about (e.g., 15m, 1h, 4h):

```bash
sudo docker compose run --rm freqtrade download-data   --exchange binance   --timeframes 15m 1h 4h
```

> **Tip:** Re‑run this periodically to keep your dataset fresh.

---

## 5. Inspect Available Data

```bash
sudo docker compose run --rm freqtrade list-data --show-timerange
```

Ensure the date range is long enough for both optimization **and** final backtesting.

---

## 6. Hyperopt (Parameter Optimisation)

Optimise your strategy on an *in‑sample* segment:

```bash
sudo docker compose run --rm freqtrade hyperopt   --strategy MyStrategy   --hyperopt-loss CustomSharpeHyperOptLoss   --spaces buy roi stoploss trades   --timerange 20240301-20240415   --epochs 50   --config user_data/config.json
```

- Adjust `--timerange` to the period you want to *train* on.
- Results are stored in `user_data/hyperopt_results/`.

---

## 7. Backtest the Optimised Strategy

Validate out‑of‑sample:

```bash
sudo docker compose run --rm freqtrade backtesting   --strategy MyStrategy   --timerange 20240416-20250523   --config user_data/config.json   --export trades
```

Compare metrics with and without optimisation to avoid over‑fitting.

---

## 8. Dry‑Run Trading (Paper Mode)

```bash
sudo docker compose run --rm freqtrade trade   --strategy MyStrategy   --config user_data/config.json
```

Make sure `"dry_run": true` in `config.json`. Monitor results and logs.

---

## 9. Live Trading

1. Edit `config.json`:

   - `"dry_run": false`
   - Production API key/secret
   - Confirm stake settings, ROI table, stoploss, protections

2. Start live mode:

   ```bash
   sudo docker compose run --rm freqtrade trade      --strategy MyStrategy      --config user_data/config.json
   ```

3. (Optional) Web UI:

   ```bash
   sudo docker compose run --rm -p 8080:8080 freqtrade webserver      --config user_data/config.json
   ```

   Open `http://localhost:8080` in your browser.

---

### House‑Keeping

Stop all containers:

```bash
sudo docker compose down
```

Update image to latest version:

```bash
sudo docker compose pull
```

---

### Cheatsheet

| Task            | Command Skeleton                            |
| --------------- | ------------------------------------------- |
| Download data   | `download-data --timeframes 15m 1h`         |
| List data range | `list-data --show-timerange`                |
| Hyperopt        | `hyperopt --timerange YYYYMMDD-YYYYMMDD`    |
| Backtest        | `backtesting --timerange YYYYMMDD-YYYYMMDD` |
| Dry‑run trade   | `trade` (ensure `"dry_run": true`)          |
| Live trade      | `trade` (ensure `"dry_run": false`)         |
| Web server      | `webserver -p 8080:8080`                    |

---

**You’re ready!** Keep refining your strategy, re‑run hyperopt, and promote only well‑tested changes to live trading.
