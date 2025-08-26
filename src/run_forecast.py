# src/run_forecast.py
import argparse
from pathlib import Path
from src.fetch_data import download_stock
from src.preprocess import prepare_monthly_series
from src.model import fit_auto_arima, forecast_and_plot

# Always resolve paths from the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Make sure folders exist
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

def main(ticker: str, start: str = "2008-01-01", periods: int = 12, price_col: str = "Close"):
    print(f"Running ARIMA monthly forecast for {ticker}")

    # Download raw stock data
    df = download_stock(ticker.upper(), start=start, end=None, out_dir=DATA_RAW_DIR)

    # Convert to monthly data
    monthly = prepare_monthly_series(df, price_col=price_col, out_dir=DATA_PROCESSED_DIR, ticker=ticker.upper())

    # Fit ARIMA
    model = fit_auto_arima(monthly, seasonal=True, m=12, out_dir=OUTPUTS_DIR, ticker=ticker.upper())

    # Forecast + plot
    forecast_df = forecast_and_plot(model, monthly, periods=periods, out_dir=OUTPUTS_DIR, ticker=ticker.upper())

    print(f"✅ Done. Check results in: {OUTPUTS_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker", help="Stock ticker symbol (e.g. AAPL, MSFT)")
    parser.add_argument("--start", default="2008-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--periods", type=int, default=12, help="Forecast periods (months)")
    parser.add_argument("--price_col", default="Close", help="Price column to use (Close or Adj Close)")
    args = parser.parse_args()
    main(args.ticker, start=args.start, periods=args.periods, price_col=args.price_col)
