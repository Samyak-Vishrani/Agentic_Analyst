import logging
from typing import Any

import pandas as pd
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

COLOUR_HISTORICAL= "#3B82F6"   # blue - historical data series
COLOUR_FORECAST= "#EF4444"   # red - forecast series
COLOUR_FITTED= "#93C5FD"   # light blue - fitted/trend line
COLOUR_BAR= "#6366F1"   # indigo - bar charts
COLOUR_SCATTER= "#8B5CF6"   # violet - scatter plots

CHART_HEIGHT = 420
FONT_FAMILY  = "Inter, system-ui, sans-serif"


# Internal helpers - shape detection
def _is_date_like(series: pd.Series) -> bool:
    """True if column looks like a date, period, or month string."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if hasattr(series, "dt"):
        return True
    # Check string patterns like "2024-01", "2024-01-01"
    if series.dtype == object or str(series.dtype) == "str" or str(series.dtype) == "string":
        sample = series.dropna().head(5).astype(str)
        return sample.str.match(r"^\d{4}-\d{2}").all()
    return False


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def _is_categorical(series: pd.Series) -> bool:
    return series.dtype == object or str(series.dtype) in ("str", "string") or pd.api.types.is_categorical_dtype(series)


def _base_layout(title: str) -> dict:
    """Shared layout config applied to every chart."""
    return dict(
        title = dict(text=title, font=dict(size=14, family=FONT_FAMILY)),
        height = CHART_HEIGHT,
        margin = dict(l=40, r=20, t=50, b=60),
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor = "rgba(0,0,0,0)",
        font = dict(family=FONT_FAMILY, size=12),
        xaxis = dict(showgrid=False, zeroline=False),
        yaxis = dict(showgrid=True, gridcolor="rgba(0,0,0,0.06)", zeroline=False),
        legend = dict(orientation="h", y=-0.2),
        hovermode = "x unified",
    )


# Chart builders - one per chart type

def _line_chart(df: pd.DataFrame, x_col: str, y_col: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x = df[x_col].astype(str),
        y = df[y_col],
        mode = "lines+markers",
        name = y_col,
        line = dict(color=COLOUR_HISTORICAL, width=2),
        marker = dict(size=5),
    ))
    fig.update_layout(
        **_base_layout(f"{y_col} over {x_col}"),
        xaxis_title = x_col,
        yaxis_title = y_col,
    )
    return fig


def _bar_chart(df: pd.DataFrame, x_col: str, y_col: str) -> go.Figure:
    df_sorted = df.sort_values(y_col, ascending=False)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x = df_sorted[x_col].astype(str),
        y = df_sorted[y_col],
        name = y_col,
        marker_color = COLOUR_BAR,
    ))
    fig.update_layout(
        **_base_layout(f"{y_col} by {x_col}"),
        xaxis_title  = x_col,
        yaxis_title  = y_col,
        xaxis_tickangle = -35,
    )
    return fig


def _scatter_chart(df: pd.DataFrame, x_col: str, y_col: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x = df[x_col],
        y = df[y_col],
        mode = "markers",
        name = f"{x_col} vs {y_col}",
        marker = dict(color=COLOUR_SCATTER, size=7, opacity=0.7),
    ))
    fig.update_layout(
        **_base_layout(f"{x_col} vs {y_col}"),
        xaxis_title = x_col,
        yaxis_title = y_col,
    )
    return fig


def _horizontal_bar(df: pd.DataFrame, numeric_cols: list[str]) -> go.Figure:
    """Single-row result - render each numeric column as a horizontal bar."""
    row = df.iloc[0]
    labels = numeric_cols
    values = [row[c] for c in numeric_cols]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x = values,
        y = labels,
        orientation = "h",
        marker_color = COLOUR_BAR,
    ))
    fig.update_layout(
        **_base_layout("Key Metrics"),
        xaxis_title = "Value",
    )
    return fig


def _forecast_overlay_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    forecast_result: dict[str, Any],
) -> go.Figure:
    """
    Two-series overlay chart:
      Series 1 - historical data_result (blue line)
      Series 2 - forecast_result predictions (red dashed line + markers)

    Both plotted on the same x-axis with a vertical boundary line at the
    forecast start to clearly mark where history ends and projection begins.
    """
    predictions  = forecast_result.get("predictions", [])
    model_type   = forecast_result.get("model_type", "Forecast")
    training_rows = forecast_result.get("training_rows", len(df))

    fig = go.Figure()

    # ── Series 1: historical ──────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x = df[x_col].astype(str),
        y = df[y_col],
        mode = "lines+markers",
        name = "Historical",
        line = dict(color=COLOUR_HISTORICAL, width=2),
        marker = dict(size=5),
    ))

    # ── Series 2: forecast ────────────────────────────────────────────────────
    if predictions:
        pred_x = [p["period"] for p in predictions]
        pred_y = [p["value"]  for p in predictions]

        # Bridge point - connect last historical to first forecast visually
        last_hist_x = str(df[x_col].iloc[-1])
        last_hist_y = float(df[y_col].iloc[-1])

        fig.add_trace(go.Scatter(
            x = [last_hist_x] + pred_x,
            y = [last_hist_y] + pred_y,
            mode = "lines+markers",
            name = f"{model_type} Forecast",
            line = dict(color=COLOUR_FORECAST, width=2, dash="dash"),
            marker = dict(size=8, symbol="diamond"),
        ))

        # ── Vertical boundary line at forecast start ──────────────────────────
        # add_vline does not support string x-axes; use a shape instead
        fig.add_shape(
            type = "line",
            xref = "x",
            yref = "paper",
            x0 = last_hist_x,
            x1 = last_hist_x,
            y0 = 0,
            y1 = 1,
            line = dict(color="rgba(0,0,0,0.25)", width=1, dash="dot"),
        )
        fig.add_annotation(
            x = last_hist_x,
            y = 1,
            yref = "paper",
            text = "Forecast →",
            showarrow = False,
            xanchor = "left",
            font = dict(size=11, color="rgba(0,0,0,0.4)"),
        )

    fig.update_layout(
        **_base_layout(
            f"Revenue Forecast - {model_type} "
            f"(trained on {training_rows} months)"
        ),
        xaxis_title = x_col,
        yaxis_title = y_col,
        xaxis_tickangle = -35,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_chart(
    data_result: list[dict[str, Any]],
    query: str = "",
    forecast_result: dict[str, Any] | None = None,
) -> go.Figure | None:
    """
    Inspect data shape and return the most appropriate Plotly figure.

    Parameters
    ----------
    data_result     : rows from sql_executor_node (may be 5-row preview)
    query           : original user question (reserved for future heuristics)
    forecast_result : structured output from forecast_node (Fix 2)
                      When provided and non-empty, renders a two-series
                      historical + forecast overlay chart.

    Returns
    -------
    go.Figure | None
        None when data shape doesn't fit any supported chart type.
        app.py handles None gracefully - shows dataframe only.
    """
    if not data_result:
        logger.debug("[visualiser] Empty data_result - no chart.")
        return None

    try:
        df = pd.DataFrame(data_result)
    except Exception as e:
        logger.warning("[visualiser] Could not build DataFrame: %s", e)
        return None

    if df.empty:
        return None

    cols = list(df.columns)
    numeric_cols = [c for c in cols if _is_numeric(df[c])]
    date_cols = [c for c in cols if _is_date_like(df[c])]
    cat_cols = [c for c in cols if _is_categorical(df[c])
                    and c not in date_cols]

    logger.info(
        "[visualiser] cols=%s numeric=%s date=%s cat=%s forecast=%s",
        cols, numeric_cols, date_cols, cat_cols,
        bool(forecast_result and forecast_result.get("predictions")),
    )

    # ── Forecast overlay (Fix 2) ──────────────────────────────────────────────
    # Takes priority over standard chart selection when forecast data present.
    if (
        forecast_result
        and forecast_result.get("predictions")
        and not forecast_result.get("error")
    ):
        # Need a date/period x-axis and a numeric y-axis
        x_col = date_cols[0] if date_cols else (
            next((c for c in cols if "date" in c.lower() or
                  "period" in c.lower() or "month" in c.lower()), None)
        )
        y_col = numeric_cols[0] if numeric_cols else None

        if x_col and y_col:
            logger.info("[visualiser] Rendering forecast overlay chart.")
            return _forecast_overlay_chart(df, x_col, y_col, forecast_result)

    # ── Date + numeric → line chart ───────────────────────────────────────────
    if date_cols and numeric_cols:
        logger.info("[visualiser] Rendering line chart.")
        return _line_chart(df, date_cols[0], numeric_cols[0])

    # ── Categorical + numeric → bar chart ─────────────────────────────────────
    if cat_cols and numeric_cols:
        n_unique = df[cat_cols[0]].nunique()
        if n_unique <= 30:
            logger.info("[visualiser] Rendering bar chart (%d categories).", n_unique)
            return _bar_chart(df, cat_cols[0], numeric_cols[0])

    # ── Two numeric cols → scatter ────────────────────────────────────────────
    if len(numeric_cols) >= 2:
        logger.info("[visualiser] Rendering scatter chart.")
        return _scatter_chart(df, numeric_cols[0], numeric_cols[1])

    # ── Single row + multiple numeric → horizontal bar ────────────────────────
    if len(df) == 1 and len(numeric_cols) >= 2:
        logger.info("[visualiser] Rendering horizontal bar (single row KPI).")
        return _horizontal_bar(df, numeric_cols)

    logger.info("[visualiser] No matching chart type - returning None.")
    return None