import ccxt

# 1) Instantiate Binance
exchange = ccxt.binance({
    'enableRateLimit': True,
})

# 2) Fetch all tickers (latest 24h data)
tickers = exchange.fetch_tickers()

# 3) Filter for USDT-quoted markets and collect their 24h quoteVolume
usdt_pairs = [
    (symbol, data.get('marketCap', 0.0))
    for symbol, data in tickers.items()
    if symbol.endswith('/USDC')
]

# 4) Sort descending by volume, take top 100
top100 = sorted(usdt_pairs, key=lambda x: x[1], reverse=True)[:30]

# 5) Extract just the symbols
top100_symbols = [symbol for symbol, vol in top100]

# 6) Print or save
print(top100_symbols)




# docker run --rm `
#   -v "${PWD}:/usr/src/app" `
#   -w /usr/src/app `
#   python:3.12-slim `
#   sh -c "pip install --no-cache-dir ccxt && python user_data/optimize/get_top100.py"
