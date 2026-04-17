from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px


def latency_bar_chart(df: pd.DataFrame) -> Any:
    if df.empty or "latency_s" not in df.columns:
        return None
    plot_df = df.copy()
    plot_df["latency_s"] = pd.to_numeric(plot_df["latency_s"], errors="coerce")
    fig = px.bar(
        plot_df,
        x="prompt_id",
        y="latency_s",
        color="technique" if "technique" in df.columns else None,
        barmode="group",
        title="Latency by prompt and technique",
    )
    return fig


def score_bar_chart(df: pd.DataFrame) -> Any:
    if df.empty or "score" not in df.columns:
        return None
    plot_df = df.copy()
    plot_df["score"] = pd.to_numeric(plot_df["score"], errors="coerce")
    fig = px.bar(
        plot_df,
        x="prompt_id",
        y="score",
        color="model" if "model" in df.columns else None,
        title="Heuristic score by prompt",
    )
    return fig
