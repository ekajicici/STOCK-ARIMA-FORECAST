# src/model.py
import pmdarima as pm
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def fit_auto_arima(series, seasonal=True, m=12, out_dir="../outputs", ticker="TICKER"):
    """Fit ARIMA model to a monthly series"""
    model = pm.auto_arima(
        series,
        seasonal=seasonal,
        m=m,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore"
    )
    return model


def forecast_and_plot(model, series, periods=12, out_dir="../outputs", ticker="TICKER"):
    """Forecast and save CSV + PNG plot"""
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Forecast
    forecast, conf_int = model.predict(n_periods=periods, return_conf_int=True)
    forecast_index = pd.date_range(series.index[-1] + pd.offsets.MonthEnd(1),
                                   periods=periods, freq="M")

    forecast_df = pd.DataFrame({
        "forecast": forecast,
        "conf_lower": conf_int[:, 0],
        "conf_upper": conf_int[:, 1],
    }, index=forecast_index)

    # 🔹 Clean date format
    forecast_df.index = forecast_df.index.strftime("%Y-%m")

    # 🔹 Add last actual value from history
    last_actual = pd.DataFrame({
        "forecast": [series.iloc[-1]],
        "conf_lower": [series.iloc[-1]],
        "conf_upper": [series.iloc[-1]],
    }, index=[series.index[-1].strftime("%Y-%m")])

    forecast_df = pd.concat([last_actual, forecast_df])

    # Save CSV
    csv_path = Path(out_dir) / f"{ticker}_forecast.csv"
    forecast_df.to_csv(csv_path)
    print(f"✅ Forecast CSV saved: {csv_path}")

    # ---- Plot ----
    plt.figure(figsize=(12, 6))
    plt.plot(series.index, series.values, label="History", color="blue")
    plt.plot(forecast_index, forecast, label="Forecast", color="red")
    plt.fill_between(forecast_index,
                     conf_int[:, 0],
                     conf_int[:, 1],
                     color="pink", alpha=0.3, label="Confidence Interval")

    plt.title(f"{ticker} Stock Price Forecast (ARIMA)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)

    # Save plot
    plot_path = Path(out_dir) / f"{ticker}_forecast.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📊 Forecast plot saved: {plot_path}")

    return forecast_df
