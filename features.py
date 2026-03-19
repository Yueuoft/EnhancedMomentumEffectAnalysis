import pandas as pd


def to_monthly_prices(daily_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Convert daily prices to month-end prices.
    Use 'ME' to avoid FutureWarning from pandas.
    """
    return daily_prices.resample("ME").last()


def compute_momentum_features(
    monthly_prices: pd.DataFrame,
    lookback: int = 6,
    holding: int = 1
):
    """
    past_returns: return over the previous lookback months
    future_returns: next holding-period return
    """
    past_returns = monthly_prices.pct_change(lookback)
    future_returns = monthly_prices.pct_change(holding).shift(-holding)
    return past_returns, future_returns
