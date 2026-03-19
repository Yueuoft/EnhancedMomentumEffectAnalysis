!pip install yfinance pandas numpy matplotlib scipy lxml html5lib

import matplotlib.pyplot as plt
import pandas as pd

from data_loader import download_prices_batch
from features import to_monthly_prices, compute_momentum_features
from portfolio import form_momentum_portfolios
from evaluation import summarize_results, cumulative_returns


def get_large_cap_tickers():
    return [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "TSLA", "BRK-B", "UNH",
        "XOM", "JNJ", "JPM", "V", "PG", "AVGO", "MA", "HD", "CVX", "LLY",
        "MRK", "ABBV", "PEP", "COST", "KO", "BAC", "ADBE", "WMT", "CRM", "NFLX",
        "MCD", "AMD", "TMO", "CSCO", "ACN", "ABT", "DHR", "LIN", "PFE", "CMCSA",
        "VZ", "DIS", "TXN", "INTC", "QCOM", "HON", "AMGN", "LOW", "UNP", "IBM",
        "CAT", "SPGI", "GE", "INTU", "GS", "RTX", "BKNG", "ISRG", "BLK", "AXP",
        "NOW", "DE", "MDT", "SYK", "PLD", "TJX", "ADP", "MMC", "AMT", "LMT",
        "MO", "GILD", "C", "SCHW", "CB", "CI", "TMUS", "SO", "DUK", "ZTS",
        "EOG", "BSX", "USB", "PNC", "CL", "TGT", "APD", "BDX", "FIS", "EQIX",
        "NSC", "ITW", "REGN", "SLB", "MU", "VRTX", "ELV", "CME", "AON", "SHW",
        "ICE", "ETN", "PYPL", "MPC", "KLAC", "EW", "GD", "EMR", "MAR", "ORLY",
        "FDX", "GM", "F", "ROP", "ADI", "HCA", "PSA", "MET", "SNPS", "AEP",
        "OXY", "MCK", "D", "TRV", "SRE", "KMB", "NOC", "AFL", "ALL", "WMB",
        "ROST", "AZO", "JCI", "GIS", "AIG", "KMI", "LHX", "CTAS", "MSI", "ADM",
        "PAYX", "IDXX", "TT", "PH", "CMI", "A", "DOW", "YUM", "STZ", "MS",
        "EXC", "PRU", "PCAR", "RSG", "CHTR", "ODFL", "MNST", "DVN", "HAL", "BIIB"
    ]


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
    tickers = get_large_cap_tickers()
    print(f"Number of tickers requested: {len(tickers)}")

    prices = download_prices_batch(
        tickers=tickers,
        start="2015-01-01",
        end="2025-01-01",
        batch_size=50
    )

    print(f"Downloaded usable price series for {prices.shape[1]} stocks.")

    monthly_prices = to_monthly_prices(prices)

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

    summary_sorted = summary_df.sort_values(
        by=["spread_sharpe", "spread_mean_monthly"],
        ascending=False
    )
    best_lookback = summary_sorted.index[0]
    best_results = all_results[best_lookback]

    print(f"\nBest lookback selected: {best_lookback} months")
    print("\n=== Best Strategy Summary ===")
    print(summary_df.loc[best_lookback].round(4))

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
