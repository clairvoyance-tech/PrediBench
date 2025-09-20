from datetime import date, datetime

import pandas as pd
from plotly import graph_objects as go

from predibench.logger_config import get_logger

logger = get_logger(__name__)

FONT_FAMILY = "Arial"
BOLD_FONT_FAMILY = "Arial"

AI_MODEL_BRAND_COLORS = {
    "xAI": "#1a1a1a",        # Very dark gray (almost black) for xAI
    "OpenAI": "#d3d3d3",     # Light gray for OpenAI
    "Qwen": "#8B5CF6",
    "DeepSeek": "#0D28F3",
    "Meta": "#0082FB",
    "Google": "#4285F4",
    "Perplexity": "#30b8c6",
    "Anthropic": "#CC785C",
    "Baseline": "#808080",
}


def get_model_color(model_name: str) -> str:
    """Get the brand color for an AI model provider.

    Args:
        model_name: Name of the AI model or provider

    Returns:
        Hex color code for the model's brand color, defaults to gray if not found
    """
    # Handle common model name variations and extract provider name
    model_lower = model_name.lower()

    # Direct matches
    for provider, color in AI_MODEL_BRAND_COLORS.items():
        if provider.lower() in model_lower:
            return color

    # Handle specific model patterns
    if "gpt" in model_lower or "chatgpt" in model_lower:
        return AI_MODEL_BRAND_COLORS["OpenAI"]
    elif "claude" in model_lower:
        return AI_MODEL_BRAND_COLORS["Anthropic"]
    elif "llama" in model_lower:
        return AI_MODEL_BRAND_COLORS["Meta"]
    elif "gemini" in model_lower or "bard" in model_lower:
        return AI_MODEL_BRAND_COLORS["Google"]
    elif "qwen" in model_lower:
        return AI_MODEL_BRAND_COLORS["Qwen"]
    elif "deepseek" in model_lower:
        return AI_MODEL_BRAND_COLORS["DeepSeek"]
    elif "grok" in model_lower:
        return AI_MODEL_BRAND_COLORS["xAI"]
    elif "baseline" in model_lower:
        return AI_MODEL_BRAND_COLORS["Baseline"]

    # Default to gray for unknown models
    return "#808080"


def date_to_string(date: datetime) -> str:
    """Convert a datetime object to YYYY-MM-DD string format."""
    return date.strftime("%Y-%m-%d")


def string_to_date(date_str: str) -> datetime:
    """Convert a YYYY-MM-DD string to datetime object."""
    return datetime.strptime(date_str, "%Y-%m-%d")


def convert_polymarket_time_to_datetime(time_str: str) -> datetime:
    """Convert a Polymarket time string to a datetime object."""
    return datetime.fromisoformat(time_str.replace("Z", "")).replace(tzinfo=None)


def apply_template(
    fig: go.Figure,
    template="none",
    annotation_text="",
    title=None,
    width=600,
    height=500,
    font_size=14,
):
    """Applies template in-place to input fig."""
    layout_updates = {
        "template": template,
        "width": width,
        "height": height,
        "font": dict(family=FONT_FAMILY, size=font_size),
        "title_font_family": BOLD_FONT_FAMILY,
        "title_font_size": 24,
        "title_xanchor": "center",
        "title_font_weight": "bold",
        "legend": dict(
            itemsizing="constant",
            title_font_family=BOLD_FONT_FAMILY,
            font=dict(family=BOLD_FONT_FAMILY, size=font_size),
            itemwidth=30,
        ),
    }
    if len(annotation_text) > 0:
        layout_updates["annotations"] = [
            dict(
                text=f"<i>{annotation_text}</i>",
                xref="paper",
                yref="paper",
                x=1.05,
                y=-0.05,
                xanchor="left",
                yanchor="top",
                showarrow=False,
                font=dict(size=font_size),
            )
        ]
    # Don't save titles in JSON figures - titles will be handled in React
    # if title is not None:
    #     layout_updates["title"] = title
    fig.update_layout(layout_updates)
    fig.update_xaxes(
        title_font_family=FONT_FAMILY,
        tickfont_family=FONT_FAMILY,
        tickfont_size=font_size,
        linewidth=1,
    )
    fig.update_yaxes(
        title_font_family=FONT_FAMILY,
        tickfont_family=FONT_FAMILY,
        tickfont_size=font_size,
        linewidth=1,
    )
    return


def _to_date_index(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with index converted to Python date objects.

    Ensures consistent comparisons and intersections between positions (date)
    and prices indices. Duplicates (same day) keep the last value.
    """
    if df is None or len(df.index) == 0:
        return df
    new_index: list[date] = []
    for idx in df.index:
        if isinstance(idx, datetime):
            new_index.append(idx.date())
        elif hasattr(idx, "date") and not isinstance(idx, date):
            # e.g., pandas Timestamp
            new_index.append(idx.date())
        else:
            new_index.append(idx)
    df2 = df.copy()
    df2.index = pd.Index(new_index)
    # remove duplicates by keeping last
    df2 = df2[~df2.index.duplicated(keep="last")]
    return df2
