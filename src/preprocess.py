# src/preprocess.py
import pandas as pd
from pathlib import Path

def prepare_monthly_series(df, price_col="Close", resample_rule="M", out_dir="../data/processed", ticker="TICKER"):
    """
    Convert daily OHLC dataframe into a monthly time series of closing price (end-of-month).
    - df: DataFrame returned by yfinance history()
    - price_col: column to use (e.g., 'Close' or 'Adj Close' if you set auto_adjust=True)
    - resample_rule: pandas resample rule, default 'M' (month end)
    Saves processed CSV to data/processed/<ticker>_monthly.csv
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)

    # Keep price column and drop NaNs
    series = df[price_col].dropna()

    # Resample to month-end (last available observation in the month)
    monthly = series.resample(resample_rule).last().ffill()  # forward fill small gaps

    csv_path = out_dir / f"{ticker}_monthly.csv"
    monthly.to_csv(csv_path, header=[price_col])
    print(f"Saved processed monthly series to {csv_path}")
    return monthly
