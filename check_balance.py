import ccxt

# Your Binance API credentials
api_key = "HUNmNEHs4ewSrRCKX5xihW4HHfkJAi6S1Btjt5LPFDFbmHa2IuUvZhlOo7SZ6G1p"
api_secret = "HZZDYs6DaSpsWoWpTjfiaRiSkL0YSosRobOqQBvKdjHVQrA72tuYBFiDqFpTefhw"

# Initialize Binance exchange
exchange = ccxt.binance({
    'apiKey': api_key,
    'secret': api_secret,
    'enableRateLimit': True,
})

# Fetch account balance
try:
    balance = exchange.fetch_balance()
    print("Your Binance account balance:")
    for currency, details in balance['total'].items():
        if details > 0:  # Show only non-zero balances
            print(f"{currency}: {details}")
except ccxt.BaseError as e:
    print(f"An error occurred: {e}")