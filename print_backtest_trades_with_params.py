
import json
import pandas as pd


# Load backtest result
with open("..\\backtest_results\\backtest-result-2025-07-28_09-25-04\\backtest-result-2025-07-28_09-25-04.json", "r") as f:
    backtest_data = json.load(f)

# Load backtest parameters
with open("..\\backtest_results\\backtest-result-2025-07-28_09-25-04\\backtest-result-2025-07-28_09-25-04_DonchianEmaBreakoutStrategy.json", "r") as f:
    param_data = json.load(f)

# Load hyperopt result
with open("..\\hyperopt_results\\strategy_DonchianEmaBreakoutStrategy_2025-07-28_09-22-12.json", "r") as f:
    hyperopt_params = json.load(f)


# Extract trades

# Extract trades from hyperopt results_metrics if available
hyperopt_trades = None
if 'results_metrics' in hyperopt_params and 'trades' in hyperopt_params['results_metrics']:
    hyperopt_trades = hyperopt_params['results_metrics']['trades']
trades = backtest_data['strategy']['DonchianEmaBreakoutStrategy']['trades']
df = pd.DataFrame(trades)
df['open_date'] = pd.to_datetime(df['open_date'])
df['close_date'] = pd.to_datetime(df['close_date'])
columns_to_show = ['pair', 'open_date', 'close_date', 'open_rate', 'close_rate', 'profit_ratio', 'profit_abs', 'exit_reason']

# Print hyperopt trade table if available
if hyperopt_trades is not None:
    df_ho = pd.DataFrame(hyperopt_trades)
    df_ho['open_date'] = pd.to_datetime(df_ho['open_date'])
    df_ho['close_date'] = pd.to_datetime(df_ho['close_date'])
    print("=== HYPEROPT TRADE TABLE ===")
    print(df_ho[columns_to_show].to_string(index=False))
else:
    print("=== HYPEROPT TRADE TABLE ===\nNo trade table found in hyperopt result.")

# Print backtest trade table
print("\n=== BACKTEST TRADE TABLE ===")
print(df[columns_to_show].to_string(index=False))

# Extract and flatten parameters
bt_params = param_data['params']
flat_bt_params = {
    'donchian_window': bt_params['buy']['donchian_window'],
    'ema_len': bt_params['buy']['ema_len'],
    'negative_stoploss': bt_params['sell']['negative_stoploss'],
    'positive_stoploss': bt_params['sell']['positive_stoploss'],
    'stoploss': bt_params['stoploss']['stoploss'],
    'roi': bt_params['roi']
}

# hyperopt_params is now loaded from file
# Try to extract trades from hyperopt result (if available)

# Extract trades from hyperopt results_metrics if available
hyperopt_trades = None
if 'results_metrics' in hyperopt_params and 'trades' in hyperopt_params['results_metrics']:
    hyperopt_trades = hyperopt_params['results_metrics']['trades']

# Compare trade tables if hyperopt trades are available
if hyperopt_trades is not None:
    df_ho = pd.DataFrame(hyperopt_trades)
    # Convert date columns for fair comparison
    df_ho['open_date'] = pd.to_datetime(df_ho['open_date'])
    df_ho['close_date'] = pd.to_datetime(df_ho['close_date'])
    # Sort both DataFrames for row order independence
    df_sorted = df.sort_values(by=['pair', 'open_date', 'close_date']).reset_index(drop=True)
    df_ho_sorted = df_ho.sort_values(by=['pair', 'open_date', 'close_date']).reset_index(drop=True)
    # Compare
    trades_match = df_sorted[columns_to_show].equals(df_ho_sorted[columns_to_show])
    print(f"\n=== TRADE TABLE COMPARISON ===")
    print("Trade tables MATCH!" if trades_match else "Trade tables DIFFER!")
else:
    print("\nNo trade table found in hyperopt result for comparison.")

# Compare and report
print("\n=== PARAMETER COMPARISON ===")
match = True
for key in ['donchian_window', 'ema_len', 'negative_stoploss', 'positive_stoploss', 'stoploss']:
    bt_value = flat_bt_params[key]
    ho_value = hyperopt_params['params_dict'][key]
    is_close = abs(bt_value - ho_value) < 1e-6
    print(f"{key}: Backtest={bt_value}, Hyperopt={ho_value} -> {'MATCH' if is_close else 'DIFFER'}")
    match = match and is_close


# Compare ROI dictionaries
roi_match = flat_bt_params['roi'] == hyperopt_params['params_details']['roi']
print(f"roi: Backtest={flat_bt_params['roi']}, Hyperopt={hyperopt_params['params_details']['roi']} -> {'MATCH' if roi_match else 'DIFFER'}")
match = match and roi_match

# Final result
print("\nAll parameters match!" if match else "\nSome parameters differ.")

# Print total profit for backtest and hyperopt
try:
    bt_total_profit = backtest_data['strategy']['DonchianEmaBreakoutStrategy']['profit_total']
except Exception:
    bt_total_profit = backtest_data.get('total_profit', backtest_data.get('results_metrics', {}).get('profit_total', 'N/A'))
ho_total_profit = hyperopt_params.get('total_profit', hyperopt_params.get('results_metrics', {}).get('profit_total', 'N/A'))
print(f"\nBacktest total_profit: {bt_total_profit}")
print(f"Hyperopt total_profit: {ho_total_profit}")
