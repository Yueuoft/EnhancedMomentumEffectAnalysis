import pandas as pd
import numpy as np
from scipy import stats


def annualized_sharpe(returns: pd.Series, periods_per_year: int = 12):
    returns = returns.dropna()
    if len(returns) < 2 or returns.std() == 0:
        return np.nan
    return np.sqrt(periods_per_year) * returns.mean() / returns.std()


def t_test_mean(returns: pd.Series):
    """
    One-sample t-test against mean = 0
    """
    returns = returns.dropna()
    if len(returns) < 2:
        return np.nan, np.nan
    t_stat, p_value = stats.ttest_1samp(returns, 0.0)
    return t_stat, p_value


def summarize_results(results: pd.DataFrame):
    spread = results["spread_return"].dropna()
    winners = results["winner_return"].dropna()
    losers = results["loser_return"].dropna()

    t_stat, p_value = t_test_mean(spread)

    summary = {
        "winner_mean_monthly": winners.mean(),
        "loser_mean_monthly": losers.mean(),
        "spread_mean_monthly": spread.mean(),
        "winner_sharpe": annualized_sharpe(winners),
        "loser_sharpe": annualized_sharpe(losers),
        "spread_sharpe": annualized_sharpe(spread),
        "spread_t_stat": t_stat,
        "spread_p_value": p_value,
        "num_periods": len(spread),
        "avg_stocks_used": results["n_stocks"].mean() if "n_stocks" in results.columns else np.nan
    }
    return pd.Series(summary)


def cumulative_returns(returns: pd.Series):
    return (1 + returns.fillna(0)).cumprod()
