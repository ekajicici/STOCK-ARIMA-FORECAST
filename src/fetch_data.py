# src/fetch_data.py
import yfinance as yf
import pandas as pd
from pathlib import Path

def download_stock(ticker: str, start: str = "2005-01-01", end: str = None, out_dir: str = "../data/raw"):
    """
    Downloads historical daily stock data for a given ticker using yfinance.
    Saves CSV to data/raw/<ticker>_raw.csv and returns a DataFrame.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ticker_obj = yf.Ticker(ticker)
    # Use history to get OHLC + volume
    df = ticker_obj.history(start=start, end=end, auto_adjust=False)  # keep this as raw
    if df.empty:
        raise ValueError(f"No data returned for ticker {ticker}")

    csv_path = out_dir / f"{ticker}_raw.csv"
    df.to_csv(csv_path)
    print(f"Saved raw data to {csv_path}")
    return df
