import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MIN_DATA_POINTS      = 3
HOLTWINTERS_MIN_ROWS = 24


def _empty_result(error: str) -> dict[str, Any]:
    return {
        "model_type": "",
        "training_rows": 0,
        "periods_ahead": 0,
        "predictions": [],
        "error": error,
    }


def _parse_time_series(data_result: list[dict[str, Any]]) -> pd.DataFrame:
    if not data_result:
        raise ValueError("data_result is empty - no data to forecast from.")

    first_row = data_result[0]
    if "forecast_date" not in first_row:
        raise ValueError(
            "Column 'forecast_date' not found. Ensure the SQL aliases the date column as 'forecast_date'."
        )
    if "forecast_value" not in first_row:
        raise ValueError(
            "Column 'forecast_value' not found. Ensure the SQL aliases the numeric metric as 'forecast_value'."
        )

    df = pd.DataFrame(data_result)[["forecast_date", "forecast_value"]]

    try:
        df["period"] = pd.to_datetime(df["forecast_date"]).dt.to_period("M")
    except Exception as e:
        raise ValueError(f"Could not parse 'forecast_date' as dates: {e}")

    try:
        df["value"] = pd.to_numeric(df["forecast_value"], errors="raise")
    except Exception as e:
        raise ValueError(f"Could not parse 'forecast_value' as numbers: {e}")

    df = (
        df.groupby("period", as_index=False)["value"]
        .sum()
        .sort_values("period")
        .reset_index(drop=True)
    )
    return df


def _next_periods(last_period: pd.Period, n: int) -> list[str]:
    return [str(last_period + i) for i in range(1, n + 1)]


def _run_linear_regression(df: pd.DataFrame, periods_ahead: int) -> dict[str, Any]:
    from sklearn.linear_model import LinearRegression

    X = np.arange(len(df)).reshape(-1, 1)
    y = df["value"].values

    model = LinearRegression()
    model.fit(X, y)

    future_X = np.arange(len(df), len(df) + periods_ahead).reshape(-1, 1)
    future_values = model.predict(future_X)
    future_labels = _next_periods(df["period"].iloc[-1], periods_ahead)

    predictions = [
        {"period": p, "value": round(float(v), 2)}
        for p, v in zip(future_labels, future_values)
    ]

    logger.info(
        "[forecaster] LinearRegression: %d training points, slope=%.4f",
        len(df), model.coef_[0],
    )

    return {
        "model_type": "LinearRegression",
        "training_rows": len(df),
        "periods_ahead": periods_ahead,
        "predictions": predictions,
        "error": "",
    }


def _run_holt_winters(df: pd.DataFrame, periods_ahead: int) -> dict[str, Any]:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    series = df["value"].values.astype(float)

    try:
        model = ExponentialSmoothing(
            series,
            trend="add",
            seasonal="add",
            seasonal_periods=12,
            initialization_method="estimated",
        )
        fit = model.fit(optimized=True)
        future_values = fit.forecast(periods_ahead)
    except Exception as hw_error:
        logger.warning(
            "[forecaster] HoltWinters failed (%s). Falling back to LinearRegression.",
            hw_error,
        )
        return _run_linear_regression(df, periods_ahead)

    future_labels = _next_periods(df["period"].iloc[-1], periods_ahead)

    predictions = [
        {"period": p, "value": round(float(v), 2)}
        for p, v in zip(future_labels, future_values)
    ]

    logger.info(
        "[forecaster] HoltWinters: %d training points, %d periods ahead.",
        len(df), periods_ahead,
    )

    return {
        "model_type": "HoltWinters",
        "training_rows": len(df),
        "periods_ahead": periods_ahead,
        "predictions": predictions,
        "error": "",
    }


def run_forecast(
    data_result: list[dict[str, Any]],
    periods_ahead: int = 3,
) -> dict[str, Any]:
    """
    Entry point called by forecast_node in src/nodes.py.

    Returns
    dict with keys: model_type, training_rows, periods_ahead, predictions, error
    """
    logger.info(
        "[forecaster] run_forecast: %d rows, %d periods ahead.",
        len(data_result), periods_ahead,
    )

    try:
        df = _parse_time_series(data_result)
    except ValueError as e:
        logger.warning("[forecaster] Parse error: %s", e)
        return _empty_result(str(e))

    n_rows = len(df)
    logger.info("[forecaster] %d monthly data points parsed.", n_rows)

    if n_rows < MIN_DATA_POINTS:
        msg = (
            f"Insufficient data: only {n_rows} monthly point(s) found. "
            f"Minimum {MIN_DATA_POINTS} required. Try a broader date range."
        )
        return _empty_result(msg)

    try:
        if n_rows >= HOLTWINTERS_MIN_ROWS:
            return _run_holt_winters(df, periods_ahead)
        else:
            return _run_linear_regression(df, periods_ahead)
    except Exception as e:
        logger.error("[forecaster] Unexpected error: %s", e, exc_info=True)
        return _empty_result(f"Unexpected forecasting error: {e}")
