import yfinance as yf
import pandas as pd
from typing import List


def download_prices_batch(
    tickers: List[str],
    start: str = "2015-01-01",
    end: str = "2025-01-01",
    batch_size: int = 50
) -> pd.DataFrame:
    """
    Download adjusted close prices in batches to reduce Yahoo Finance failures
    for large universes like the S&P 500.
    """
    all_prices = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        print(f"Downloading batch {i // batch_size + 1} / {(len(tickers) - 1) // batch_size + 1} ...")

        data = yf.download(
            batch,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            group_by="column"
        )

        if data.empty:
            continue

        # Usually Yahoo returns multi-index columns with "Close"
        if isinstance(data.columns, pd.MultiIndex):
            if "Close" in data.columns.get_level_values(0):
                batch_prices = data["Close"]
            else:
                batch_prices = data
        else:
            # Single ticker case
            if "Close" in data.columns:
                batch_prices = data[["Close"]].copy()
                if len(batch) == 1:
                    batch_prices.columns = batch
            else:
                batch_prices = data.copy()

        all_prices.append(batch_prices)

    if not all_prices:
        raise ValueError("No price data was downloaded. Please check internet connection or ticker symbols.")

    prices = pd.concat(all_prices, axis=1)

    # Remove duplicate columns if any appear
    prices = prices.loc[:, ~prices.columns.duplicated()]

    # Drop columns with all missing values
    prices = prices.dropna(axis=1, how="all").sort_index()

    return prices
