# Quick-Start Guide for Linux (Freqtrade)

Follow these steps to run Freqtrade on Linux. Pick your config file (`config.panagiotis.json` or `config.pantazis.json`) as needed.

---

## 1. Prerequisites

- **Docker & Docker Compose** installed  
- Freqtrade repository cloned under `~/freqtrade/`  
- Two config files in `~/freqtrade/user_data/`:  
  - `config.panagiotis.json`  
  - `config.pantazis.json`  

Each config must include your API keys, trading pairs, stake settings, risk management, etc.

---

## 2. Open a Terminal & Navigate

```bash
# Change into the freqtrade user_data directory
cd ~/freqtrade/user_data
```

---

## 3. Start the Services

Bring up Redis, PostgreSQL, the bot container, etc., in detached mode:

```bash
sudo docker compose up -d
```

---

## 4. Download Historical Candles

Fetch 1h OHLCV data from Binance (adjust as needed):

```bash
sudo docker compose run --rm freqtrade download-data \
  --exchange binance \
  --timeframes 15m
```

---

## 5. Verify Available Data and Timeranges

Check the available historical data and its timerange for each pair and timeframe:

```bash
sudo docker compose run --rm freqtrade list-data --show-timerange
```
docker compose run --rm freqtrade list-data --show-timerange


### Explanation:
- This command lists all the pairs and timeframes for which historical data is available in your `user_data/data` directory.
- It also shows the **start** and **end** dates of the data for each pair and timeframe.
- Use this information to ensure you have sufficient data for backtesting or hyperoptimization.

---

## 6. Run a Backtest

Choose your config and backtest:

```bash
sudo docker compose run --rm freqtrade backtesting \
  --strategy BBRRSIStrategy \
  --config user_data/config.<your_user>.json
```

_Example for Panagiotis:_

```bash
sudo docker compose run --rm freqtrade backtesting \
  --strategy BBRRSIStrategy \
  --config user_data/config.panagiotis.json
```

---

## 7. Hyperopt (Parameter Optimization)

Tune your strategy parameters over a given timerange:

```bash
sudo docker compose run --rm freqtrade hyperopt \
  --strategy BBRRSIStrategy2 \
  --hyperopt-loss CustomSharpeHyperOptLoss \
  --spaces buy roi stoploss trades \
  --timerange 20250313-20250423 \
  --epochs 50 \
  --config user_data/config.<your_user>.json
```

---

## 8. Dry-Run Trading (Simulated)

Verify setup in “paper” mode (no real orders):

```bash
sudo docker compose run --rm freqtrade trade \
  --strategy BBRRSIStrategy \
  --config user_data/config.<your_user>.json
```

Ensure your config has:
```json
"dry_run": true
```

---

## 9. Live Trading

1. **Update your config** (`config.<your_user>.json`):  
   - Set `"dry_run": false`  
   - Add your live API key/secret  
   - Define stake currency, stake amount, stoploss, ROI, etc.  

2. **Run live mode**:

   ```bash
   sudo docker compose run --rm freqtrade trade \
     --strategy BBRRSIStrategy \
     --config user_data/config.<your_user>.json
   ```

3. **(Optional) Web Dashboard**:

   ```bash
   sudo docker compose run --rm -p 8080:8080 freqtrade webserver \
     --config user_data/config.<your_user>.json
   ```

   Open your browser at `http://localhost:8080`.

---

> **Tip:**  
> To stop all services:  
> ```bash
> sudo docker compose down
> ```  
> Then restart only the bot container anytime by re-running the `trade…` command.