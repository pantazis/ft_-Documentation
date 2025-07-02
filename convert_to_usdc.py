#!/usr/bin/env python3
"""
convert_to_usdc.py
==================

One‑shot converter that liquidates **every non‑stable‑coin spot balance**
into **USDC**.  It handles dust (balances worth < $1) in three ways:

* **Skip** (default) – leaves the dust untouched.
* **Sweep to BNB** ("Convert Small Balances" endpoint) then sells BNB→USDC.
* **Custom threshold** – change the $1 rule with --min-notional.

Safeguards
----------
* **--dry-run** mode prints every step but sends *no* orders.
* Uses Binance `LOT_SIZE` & `MIN_NOTIONAL` filters so orders won’t be rejected.
* Refuses to start if API creds are missing.

Usage
-----
```bash
pip install python-binance tabulate python-dotenv

# dry run (recommended first)
BINANCE_KEY=... BINANCE_SECRET=... python convert_to_usdc.py --dry-run

# live run, skip dust
python convert_to_usdc.py

# live run, sweep dust < $1 to BNB and then to USDC
python convert_to_usdc.py --convert-dust
```
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from decimal import Decimal, ROUND_DOWN
from typing import List, Tuple

from binance.client import Client
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv
from tabulate import tabulate

##############################################################################
# 1)  CLI & Configuration
##############################################################################

parser = argparse.ArgumentParser(description="Liquidate all spot assets to USDC")
parser.add_argument("--dry-run", action="store_true", help="log actions without sending orders")
parser.add_argument("--min-notional", type=float, default=1.0,
                    help="USD value below which a balance is treated as dust (default: 1)")
parser.add_argument("--convert-dust", action="store_true",
                    help="sweep balances below --min-notional to BNB, then sell BNB→USDC")
args = parser.parse_args()

DRY_RUN: bool = args.dry_run
MIN_NOTIONAL: Decimal = Decimal(str(args.min_notional))
CONVERT_DUST: bool = args.convert_dust

load_dotenv()
API_KEY =      "HUNmNEHs4ewSrRCKX5xihW4HHfkJAi6S1Btjt5LPFDFbmHa2IuUvZhlOo7SZ6G1p"
API_SECRET = "HZZDYs6DaSpsWoWpTjfiaRiSkL0YSosRobOqQBvKdjHVQrA72tuYBFiDqFpTefhw"
if not API_KEY or not API_SECRET:
    sys.exit("✘  BINANCE_KEY / BINANCE_SECRET not set (env or .env file)")

client = Client(API_KEY, API_SECRET, tld="com")
PREFERRED_QUOTES: Tuple[str, ...] = ("USDC", "USDT")  # search order

##############################################################################
# 2)  Helpers
##############################################################################


def has_pair(asset: str, quote: str) -> bool:
    return client.get_symbol_info(f"{asset}{quote}") is not None


def get_symbol_info_cached(symbol: str) -> dict:
    # light caching so we don't hit rate limits fetching the same info repeatedly
    if not hasattr(get_symbol_info_cached, "_cache"):
        get_symbol_info_cached._cache = {}
    cache = get_symbol_info_cached._cache
    if symbol not in cache:
        cache[symbol] = client.get_symbol_info(symbol)
    return cache[symbol]


def get_precision(symbol_info: dict) -> int:
    """Get the precision for quantity based on LOT_SIZE filter"""
    for f in symbol_info["filters"]:
        if f["filterType"] == "LOT_SIZE":
            step_size = f["stepSize"]
            # Handle different step size formats
            if step_size == "1.00000000":
                return 0  # whole numbers only
            elif step_size == "0.10000000":
                return 1  # 1 decimal place
            elif step_size == "0.01000000":
                return 2  # 2 decimal places
            elif step_size == "0.00100000":
                return 3  # 3 decimal places
            elif step_size == "0.00010000":
                return 4  # 4 decimal places
            elif step_size == "0.00001000":
                return 5  # 5 decimal places
            elif step_size == "0.00000100":
                return 6  # 6 decimal places
            elif step_size == "0.00000010":
                return 7  # 7 decimal places
            elif step_size == "0.00000001":
                return 8  # 8 decimal places
            else:
                # For other formats, calculate precision from step size
                return abs(Decimal(step_size).as_tuple().exponent)
    return 8  # safe fallback


def round_qty(qty: Decimal, precision: int) -> Decimal:
    """Round quantity to the correct precision"""
    return qty.quantize(Decimal(f"1e-{precision}"), ROUND_DOWN)


def validate_and_round_qty(qty: Decimal, symbol_info: dict) -> Decimal:
    """Validate quantity against LOT_SIZE filter and round appropriately"""
    lot_size_filter = None
    for f in symbol_info["filters"]:
        if f["filterType"] == "LOT_SIZE":
            lot_size_filter = f
            break
    
    if not lot_size_filter:
        return qty.quantize(Decimal("0.00000001"), ROUND_DOWN)
    
    step_size = Decimal(lot_size_filter["stepSize"])
    min_qty = Decimal(lot_size_filter["minQty"])
    max_qty = Decimal(lot_size_filter["maxQty"])
    
    # Round down to nearest step
    if step_size > 0:
        qty_rounded = (qty // step_size) * step_size
    else:
        qty_rounded = qty.quantize(Decimal("0.00000001"), ROUND_DOWN)
    
    # Check bounds
    if qty_rounded < min_qty:
        return Decimal("0")
    if qty_rounded > max_qty:
        qty_rounded = max_qty
        
    return qty_rounded


def get_price(symbol: str) -> Decimal:
    return Decimal(client.get_symbol_ticker(symbol=symbol)["price"])


def market_sell(symbol: str, qty: Decimal) -> dict:
    if DRY_RUN:
        print(f"    ↪︎ [dry‑run] would sell {qty} {symbol[:-4]} in {symbol}")
        return {"cummulativeQuoteQty": "0"}
    return client.order_market_sell(symbol=symbol, quantity=float(qty))


def dust_convert(assets: List[str]) -> dict | None:
    """Use Binance dust‑transfer endpoint to sweep tiny assets to BNB."""
    if DRY_RUN or not assets:
        print(f"    ↪︎ [dry‑run] would sweep dust: {', '.join(assets)} → BNB")
        return None
    try:
        # python‑binance helper: client.transfer_dust(asset=[...])
        return client.transfer_dust(asset=assets)
    except BinanceAPIException as e:
        print(f"‼️  Dust convert failed: {e.message}")
        return None

##############################################################################
# 3)  Main routine
##############################################################################

def convert_all() -> None:
    acct = client.get_account()
    balances = {b["asset"]: Decimal(b["free"]) for b in acct["balances"] if Decimal(b["free"]) > 0}

    dust_assets: List[str] = []
    usdt_buffer = Decimal("0")
    conversions: List[Tuple[str, Decimal, str, Decimal]] = []

    for asset, qty in balances.items():
        if asset in PREFERRED_QUOTES:  # skip USDC & USDT for now
            continue

        # choose best quote (USDC then USDT)
        quote = next((q for q in PREFERRED_QUOTES if has_pair(asset, q)), None)
        if not quote:
            print(f"⏩  {asset}: no suitable quote market → skipped")
            continue

        symbol = f"{asset}{quote}"
        info = get_symbol_info_cached(symbol)
        qty_r = validate_and_round_qty(qty, info)
        if qty_r == 0:
            print(f"⏩  {asset}: quantity {qty} too small after rounding for {symbol} → skipped")
            continue

        # notional check
        price = get_price(symbol)
        notional = price * qty_r
        if notional < MIN_NOTIONAL:
            print(f"⏩  {asset}: value {notional:.4f} < {MIN_NOTIONAL} → marking as dust")
            dust_assets.append(asset)
            continue

        print(f"⚡ Selling {qty_r} {asset} for {quote} (≈{notional:.2f} {quote}) …")
        try:
            order = market_sell(symbol, qty_r)
            quote_amt = Decimal(order.get("cummulativeQuoteQty", "0"))
            conversions.append((asset, qty_r, quote, quote_amt))
            if quote == "USDT":
                usdt_buffer += quote_amt
            time.sleep(0.4)  # protect weight
        except BinanceAPIException as e:
            print(f"‼️  {asset}: order failed → {e.message}")

    # ----- Optional dust sweep -----
    if CONVERT_DUST and dust_assets:
        print(f"🚮 Converting dust {dust_assets} → BNB via Binance dust tool …")
        resp = dust_convert(dust_assets)
        if resp and not DRY_RUN:
            # response contains list of transfers with amounts
            total_bnb = sum(Decimal(i["transfered_amount"])
                            for i in resp["transferResult"])
            if total_bnb > 0:
                print(f"⚡ Selling dust‑BNB {total_bnb} → USDC …")
                if has_pair("BNB", "USDC"):
                    symbol = "BNBUSDC"
                    info = get_symbol_info_cached(symbol)
                    qty_r = validate_and_round_qty(total_bnb, info)
                    if qty_r > 0:
                        order = market_sell(symbol, qty_r)
                        conversions.append(("BNB", qty_r, "USDC",
                                            Decimal(order.get("cummulativeQuoteQty", "0"))))

    # ----- Aggregate USDT → USDC -----
    if usdt_buffer > 0:
        symbol = "USDTUSDC" if has_pair("USDT", "USDC") else None
        if symbol:
            info = get_symbol_info_cached(symbol)
            qty_r = validate_and_round_qty(usdt_buffer, info)
            if qty_r > 0:
                print(f"⚡ Converting aggregated {qty_r} USDT → USDC …")
                try:
                    order = market_sell(symbol, qty_r)
                    conversions.append(("USDT", qty_r, "USDC",
                                        Decimal(order.get("cummulativeQuoteQty", "0"))))
                except BinanceAPIException as e:
                    print(f"‼️  USDT→USDC failed → {e.message}")
            else:
                print(f"⏩  USDT buffer {usdt_buffer} too small after rounding → skipped")

    # ----- Reporting -----
    print("\n===== CONVERSION SUMMARY =====")
    if conversions:
        rows = [(a, q, f"→ {quo}", r) for a, q, quo, r in conversions]
        print(tabulate(rows, headers=["Asset Sold", "Qty", "Pair", "Quote Received"]))
    else:
        print("No assets converted.")

    acct = client.get_account()
    final_balances = [(b["asset"], b["free"]) for b in acct["balances"] if Decimal(b["free"]) > 0]
    print("\n===== FINAL BALANCES =====")
    print(tabulate(final_balances, headers=["Asset", "Free"]))


if __name__ == "__main__":
    convert_all()
