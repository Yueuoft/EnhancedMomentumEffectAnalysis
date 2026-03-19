import pandas as pd


def form_momentum_portfolios(
    past_returns: pd.DataFrame,
    future_returns: pd.DataFrame,
    top_quantile: float = 0.2
) -> pd.DataFrame:
    """
    Each month:
    - rank stocks by past return
    - winners = top quantile
    - losers = bottom quantile
    - evaluate next-month returns
    """
    results = []

    for date in past_returns.index:
        signal = past_returns.loc[date].dropna()
        future = future_returns.loc[date].dropna()

        common = signal.index.intersection(future.index)
        signal = signal.loc[common]
        future = future.loc[common]

        # Require enough names to make quantile portfolios meaningful
        if len(signal) < 20:
            continue

        n_top = max(1, int(len(signal) * top_quantile))

        ranked = signal.sort_values()
        losers = ranked.index[:n_top]
        winners = ranked.index[-n_top:]

        winner_ret = future.loc[winners].mean()
        loser_ret = future.loc[losers].mean()
        spread_ret = winner_ret - loser_ret

        results.append({
            "date": date,
            "winner_return": winner_ret,
            "loser_return": loser_ret,
            "spread_return": spread_ret,
            "n_stocks": len(common),
            "n_portfolio": n_top
        })

    if not results:
        raise ValueError("No portfolio periods were formed. Check data coverage and parameters.")

    return pd.DataFrame(results).set_index("date")
