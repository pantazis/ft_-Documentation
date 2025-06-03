# CustomSharpeHyperOptLoss.py
from __future__ import annotations

from datetime import datetime
from typing import Dict, Any, List

import numpy as np
from pandas import DataFrame
from freqtrade.constants import Config
from freqtrade.optimize.hyperopt import IHyperOptLoss 


class winrate(IHyperOptLoss):
    """
    Hyperopt loss class that *maximises* the annualised Sharpe ratio
    (after subtracting a fixed slippage/fee).  Because Hyperopt minimises
    the returned value, we negate the Sharpe ratio so “smaller is better”.
    """

    SLIPPAGE_PER_TRADE: float = 0.0005  # 0.05 % haircut
    DAYS_IN_YEAR: int = 365

    # ------------------------------------------------------------------ #
    @staticmethod
    def hyperopt_loss_function(
        results: DataFrame,        
        trade_count: int,
        min_date: datetime,
        max_date: datetime,
        config: Config,
        processed: Dict[str, DataFrame],
        *args: Any,
        **kwargs: Any,
    ) -> float:
        """
        Objective function for Hyperopt.  Returns *negative* Sharpe ratio:
        a smaller number means a higher Sharpe (i.e. a better strategy).
        """
        # 1 ─────────────────────────────── period length ─────────────────
        days_period = max((max_date - min_date).days, 1)  # avoid div-0

        # 2 ─────────────────────────── per-trade returns ─────────────────
        # results['profit_ratio'] is already a decimal e.g. 0.0123 = 1.23 %
        profits: np.ndarray = results["profit_ratio"].to_numpy(dtype=float)

        # Apply fixed costs / slippage to *each* trade
        profits -= winrate.SLIPPAGE_PER_TRADE
        computed_win_rate = results["profit_ratio"].gt(0).mean()
        if computed_win_rate == 1.0:
            computed_win_rate = 0.01
   

        # 3 ─────────────────────────── annualised return ─────────────────
        expected_yearly_return = profits.sum() / days_period

        # 4 ──────────────────────── standard deviation ──────────────────
        returns_std = np.std(profits)
        average_profit = profits.mean()
        
        total_profit = profits.sum()
        number_of_trades = len(profits)
            

        max_drawdown = kwargs.get('max_drawdown', 0)
        if (
            number_of_trades == 0
            or computed_win_rate == 1.0
            or total_profit == 0
            or max_drawdown == 0
        ):
            sharpe_ratio = -99.0
            
            
        else:
            sharpe_ratio = (number_of_trades * total_profit)  * (0.5 + computed_win_rate)  # neutral value when variance is zero

        # 5 ───────────────────────────── convert to loss ─────────────────
        print(f"Sharpe Ratio: {sharpe_ratio}, Win Rate: {computed_win_rate}, Total Profit: {total_profit}, Number of Trades: {number_of_trades}")
      
        loss = -sharpe_ratio
        return loss
