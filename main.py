!pip install yfinance pandas numpy matplotlib scipy lxml html5lib

import pandas as pd
import matplotlib.pyplot as plt

from data_loader import download_prices_batch
from features import to_monthly_prices, compute_momentum_features
from portfolio import form_momentum_portfolios
from evaluation import summarize_results, cumulative_returns


def get_sp500_tickers():
    """
    Pull S&P 500 constituents from Wikipedia and format for Yahoo Finance.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url)[0]
    tickers = table["Symbol"].tolist()

    # Yahoo Finance uses '-' instead of '.' in tickers like BRK.B -> BRK-B
    tickers = [ticker.replace(".", "-") for ticker in tickers]
    return tickers


def run_strategy_for_lookback(monthly_prices, lookback, holding=1, top_quantile=0.2):
    past_ret, future_ret = compute_momentum_features(
        monthly_prices,
        lookback=lookback,
        holding=holding
    )

    results = form_momentum_portfolios(
        past_ret,
        future_ret,
        top_quantile=top_quantile
    )

    summary = summarize_results(results)
    return results, summary


def main():
    # 1) Load larger universe
    tickers = get_sp500_tickers()
    print(f"Number of tickers requested: {len(tickers)}")

    prices = download_prices_batch(
        tickers=tickers,
        start="2015-01-01",
        end="2025-01-01",
        batch_size=50
    )

    print(f"Downloaded usable price series for {prices.shape[1]} stocks.")
    monthly_prices = to_monthly_prices(prices)

    # 2) Compare multiple lookback horizons
    lookbacks = [3, 6, 12]
    all_results = {}
    all_summaries = {}

    for lb in lookbacks:
        print(f"\nRunning momentum strategy with {lb}-month lookback...")
        results, summary = run_strategy_for_lookback(
            monthly_prices,
            lookback=lb,
            holding=1,
            top_quantile=0.2
        )
        all_results[lb] = results
        all_summaries[lb] = summary

    summary_df = pd.DataFrame(all_summaries).T
    summary_df.index.name = "lookback_months"

    print("\n=== Summary Across Lookbacks ===")
    print(summary_df.round(4))

    # 3) Choose best strategy by spread Sharpe, then by spread mean monthly
    summary_sorted = summary_df.sort_values(
        by=["spread_sharpe", "spread_mean_monthly"],
        ascending=False
    )
    best_lookback = summary_sorted.index[0]
    best_results = all_results[best_lookback]

    print(f"\nBest lookback selected: {best_lookback} months")
    print("\n=== Best Strategy Summary ===")
    print(summary_df.loc[best_lookback].round(4))

    # 4) Plot cumulative returns for best strategy
    winner_curve = cumulative_returns(best_results["winner_return"])
    loser_curve = cumulative_returns(best_results["loser_return"])
    spread_curve = cumulative_returns(best_results["spread_return"])

    plt.figure(figsize=(10, 6))
    plt.plot(winner_curve, label="Winners")
    plt.plot(loser_curve, label="Losers")
    plt.plot(spread_curve, label="Winner-Loser Spread")
    plt.title(f"Momentum Strategy Cumulative Returns (Best Lookback = {best_lookback}M)")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
