import ccxt

# 1) Instantiate Binance with rate limit
exchange = ccxt.binance({
    'enableRateLimit': True,
})

# 2) Fetch all markets and filter for spot pairs quoted in USDC
markets = exchange.fetch_markets()
spot_symbols = {
    market["symbol"] 
    for market in markets 
    if market["type"] == "spot" and market["quote"] == "USDC"
}

# 3) Fetch all tickers (latest 24h data)
tickers = exchange.fetch_tickers()

# 4) Filter for spot pairs (using our filtered symbols) and collect their 24h volume
spot_pairs = [
    (symbol, data.get('MarketCapPairList', 0.0))
    for symbol, data in tickers.items()
    if symbol in spot_symbols
]

# 5) Sort descending by the volume value, take top 100
top100 = sorted(spot_pairs, key=lambda x: x[1], reverse=True)[:50]

# 6) Extract just the symbols
top100_symbols = [symbol for symbol, vol in top100]

# 7) Print or save the top 100 symbols
print(top100_symbols)




# docker run --rm `
#   -v "${PWD}:/usr/src/app" `
#   -w /usr/src/app `
#   python:3.12-slim `
#   sh -c "pip install --no-cache-dir ccxt && python user_data/optimize/get_top100.py"
