#!/usr/bin/env python3
"""
Market Dynamics Analysis: Price Adjustment Speed and Bet vs Edge Consistency

This script analyzes:
1. How quickly market prices adjust to news events
2. Consistency between agent bets and edge (estimated odds vs market prices)
3. Calibration improvements over time
4. Comparison with Kelly criterion
"""

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from predibench.backend.data_loader import get_data_for_backend
from predibench.utils import apply_template, get_model_color
from predibench.common import FRONTEND_PUBLIC_PATH


def analyze_price_volatility_around_events(backend_data) -> Dict:
    """Analyze price movements around significant events."""
    price_changes = []

    for event in backend_data.events:
        for market in event.markets:
            if market.prices and len(market.prices) > 1:
                prices_df = pd.DataFrame([
                    {"date": pd.to_datetime(p.date), "price": p.value}
                    for p in market.prices
                ]).set_index("date").sort_index()

                # Calculate daily price changes
                prices_df["price_change"] = prices_df["price"].diff()
                prices_df["price_change_pct"] = prices_df["price"].pct_change()
                prices_df["volatility"] = prices_df["price_change"].abs()

                # Find significant price movements (>5% in a day)
                significant_moves = prices_df[prices_df["price_change_pct"].abs() > 0.05]

                for date, row in significant_moves.iterrows():
                    # Look at price adjustment in following days
                    future_window = prices_df[date:date + timedelta(days=7)]
                    if len(future_window) > 1:
                        adjustment_speed = _calculate_adjustment_speed(future_window)

                        price_changes.append({
                            "event_id": event.id,
                            "market_id": market.id,
                            "event_title": event.title,
                            "market_question": market.question,
                            "shock_date": date,
                            "initial_change": row["price_change_pct"],
                            "adjustment_speed": adjustment_speed,
                            "volatility_1d": row["volatility"],
                            "price_before": prices_df.loc[date, "price"] - row["price_change"],
                            "price_after": prices_df.loc[date, "price"]
                        })

    return pd.DataFrame(price_changes)


def _calculate_adjustment_speed(price_window: pd.DataFrame) -> float:
    """Calculate how quickly price adjusts after initial shock."""
    if len(price_window) < 2:
        return 0.0

    initial_price = price_window.iloc[0]["price"]
    final_price = price_window.iloc[-1]["price"]

    # Calculate half-life of adjustment
    price_changes = price_window["price_change"].fillna(0)
    if len(price_changes) < 2:
        return 0.0

    # Simple measure: days to reach 50% of total adjustment
    total_adjustment = abs(final_price - initial_price)
    if total_adjustment < 0.001:  # Minimal adjustment
        return 0.0

    cumulative_adj = 0.0
    for i, change in enumerate(price_changes[1:], 1):
        cumulative_adj += abs(change)
        if cumulative_adj >= total_adjustment * 0.5:
            return i

    return len(price_changes)


def analyze_bet_vs_edge_consistency(backend_data) -> pd.DataFrame:
    """Analyze consistency between agent bets and their estimated edge."""
    bet_edge_data = []

    for decision in backend_data.model_decisions:
        # Skip baseline models
        if "baseline" in decision.model_info.inference_provider.lower():
            continue

        for event_decision in decision.event_investment_decisions:
            for market_decision in event_decision.market_investment_decisions:
                # Get market price at decision time
                event = backend_data.event_details.get(event_decision.event_id)
                if not event:
                    continue

                market = next((m for m in event.markets if m.id == market_decision.market_id), None)
                if not market or not market.prices:
                    continue

                # Find market price closest to decision date
                decision_date = pd.to_datetime(str(decision.target_date))
                market_price = None

                for price_point in market.prices:
                    price_date = pd.to_datetime(price_point.date)
                    if abs((price_date - decision_date).days) <= 1:
                        market_price = price_point.value
                        break

                if market_price is None:
                    continue

                estimated_prob = market_decision.decision.estimated_probability
                bet_amount = market_decision.decision.bet
                confidence = market_decision.decision.confidence

                # Calculate edge
                edge = estimated_prob - market_price

                # Determine if bet direction is consistent with edge
                # More precise consistency check
                if abs(edge) < 0.01:  # No significant edge
                    consistent = abs(bet_amount) < 0.05  # Should bet very little
                elif edge > 0.01:  # Positive edge
                    consistent = bet_amount > 0.01  # Should bet positive
                else:  # Negative edge (edge < -0.01)
                    consistent = bet_amount < -0.01  # Should bet negative

                # Kelly criterion optimal bet
                if market_price > 0 and market_price < 1:
                    # Proper Kelly formula for binary outcomes
                    if estimated_prob > market_price:
                        # Kelly = (bp - q) / b where b=odds-1, p=prob, q=1-p
                        odds_for_yes = (1 - market_price) / market_price  # If market price is 0.6, odds are 0.4/0.6 = 2/3
                        kelly_bet = (estimated_prob * odds_for_yes - (1 - estimated_prob)) / odds_for_yes
                        kelly_bet = max(0, min(1, kelly_bet))  # Clamp to [0, 1] for positive bets
                    elif estimated_prob < market_price:
                        # Short Kelly bet (betting against)
                        odds_for_no = market_price / (1 - market_price)
                        kelly_bet = ((1 - estimated_prob) * odds_for_no - estimated_prob) / odds_for_no
                        kelly_bet = -max(0, min(1, kelly_bet))  # Negative for betting against
                    else:
                        kelly_bet = 0
                else:
                    kelly_bet = 0

                bet_edge_data.append({
                    "model_id": decision.model_id,
                    "model_name": decision.model_info.model_pretty_name,
                    "provider": decision.model_info.inference_provider,
                    "event_id": event_decision.event_id,
                    "market_id": market_decision.market_id,
                    "date": str(decision.target_date),
                    "estimated_prob": estimated_prob,
                    "market_price": market_price,
                    "edge": edge,
                    "bet_amount": bet_amount,
                    "confidence": confidence,
                    "consistent": consistent,
                    "kelly_bet": kelly_bet,
                    "bet_vs_kelly": abs(bet_amount - kelly_bet),
                    "abs_edge": abs(edge)
                })

    return pd.DataFrame(bet_edge_data)


def create_price_adjustment_visualization(price_volatility_df: pd.DataFrame) -> go.Figure:
    """Create visualization of price adjustment speed around events."""
    if price_volatility_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No significant price movements found",
                          xref="paper", yref="paper", x=0.5, y=0.5)
        apply_template(fig, title="Market Price Adjustment Dynamics", width=1200, height=800)
        return fig

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Price Adjustment Speed Distribution",
            "Initial Shock vs Adjustment Speed",
            "Price Volatility Examples",
            "Adjustment Speed by Market Type"
        ],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )

    # 1. Adjustment speed histogram
    fig.add_trace(
        go.Histogram(
            x=price_volatility_df["adjustment_speed"],
            nbinsx=20,
            name="Adjustment Speed (days)",
            showlegend=False
        ),
        row=1, col=1
    )

    # 2. Scatter: Initial shock vs adjustment speed
    fig.add_trace(
        go.Scatter(
            x=price_volatility_df["initial_change"].abs(),
            y=price_volatility_df["adjustment_speed"],
            mode="markers",
            text=price_volatility_df["event_title"],
            name="Markets",
            marker=dict(
                size=8,
                color=price_volatility_df["volatility_1d"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Volatility")
            ),
            showlegend=False
        ),
        row=1, col=2
    )

    # 3. Example price movements (top 5 by volatility)
    top_volatile = price_volatility_df.nlargest(5, "volatility_1d")
    for i, (_, row) in enumerate(top_volatile.iterrows()):
        fig.add_trace(
            go.Scatter(
                x=[0, 1, row["adjustment_speed"]],
                y=[row["price_before"], row["price_after"], row["price_after"]],
                mode="lines+markers",
                name=f"{row['event_title'][:30]}...",
                line=dict(width=2),
                showlegend=False
            ),
            row=2, col=1
        )

    # 4. Adjustment speed by quartiles of shock magnitude
    price_volatility_df["shock_quartile"] = pd.qcut(
        price_volatility_df["initial_change"].abs(),
        q=4,
        labels=["Q1 (Small)", "Q2", "Q3", "Q4 (Large)"]
    )

    for quartile in price_volatility_df["shock_quartile"].unique():
        if pd.isna(quartile):
            continue
        quartile_data = price_volatility_df[price_volatility_df["shock_quartile"] == quartile]
        fig.add_trace(
            go.Box(
                y=quartile_data["adjustment_speed"],
                name=str(quartile),
                showlegend=False
            ),
            row=2, col=2
        )

    # Update layout
    fig.update_layout(
        title="Market Price Adjustment Dynamics",
        height=800,
        width=1200
    )

    fig.update_xaxes(title_text="Adjustment Speed (days)", row=1, col=1)
    fig.update_xaxes(title_text="Initial Shock Magnitude", row=1, col=2)
    fig.update_xaxes(title_text="Days After Shock", row=2, col=1)
    fig.update_xaxes(title_text="Shock Magnitude Quartile", row=2, col=2)

    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Adjustment Speed (days)", row=1, col=2)
    fig.update_yaxes(title_text="Price", row=2, col=1)
    fig.update_yaxes(title_text="Adjustment Speed (days)", row=2, col=2)

    apply_template(fig, title="Market Price Adjustment Dynamics", width=1200, height=800)
    return fig



def create_edge_vs_bet_scatter(bet_edge_df: pd.DataFrame) -> go.Figure:
    """Create scatter plot of edge vs bet amount."""
    fig = go.Figure()

    if bet_edge_df.empty:
        fig.add_annotation(text="No betting data found",
                          xref="paper", yref="paper", x=0.5, y=0.5)
        apply_template(fig, title="Edge vs Bet Amount by Model", width=800, height=600)
        return fig

    for i, model_name in enumerate(bet_edge_df["model_name"].unique()):
        model_data = bet_edge_df[bet_edge_df["model_name"] == model_name]
        color = get_model_color(model_name)

        fig.add_trace(
            go.Scatter(
                x=model_data["edge"],
                y=model_data["bet_amount"],
                mode="markers",
                name=model_name,
                marker=dict(color=color, size=6, opacity=0.7),
            )
        )

    # Add diagonal line for perfect consistency
    edge_range = [-0.5, 0.5]
    fig.add_trace(
        go.Scatter(
            x=edge_range, y=edge_range,
            mode="lines",
            name="Perfect Edge-Bet Alignment",
            line=dict(dash="dash", color="black", width=2),
        )
    )

    fig.update_layout(
        title="Edge vs Bet Amount by Model",
        xaxis_title="Edge (Estimated Prob - Market Price)",
        yaxis_title="Bet Amount",
        height=600,
        width=800
    )

    apply_template(fig, title="Edge vs Bet Amount by Model", width=800, height=600)
    return fig


def create_consistency_rate_chart(bet_edge_df: pd.DataFrame) -> go.Figure:
    """Create bar chart of consistency rates by model."""
    fig = go.Figure()

    if bet_edge_df.empty:
        fig.add_annotation(text="No betting data found",
                          xref="paper", yref="paper", x=0.5, y=0.5)
        apply_template(fig, title="Bet-Edge Consistency Rate by Model", width=1000, height=600)
        return fig

    consistency_by_model = bet_edge_df.groupby("model_name")["consistent"].agg(["mean", "count"]).reset_index()
    consistency_by_model = consistency_by_model[consistency_by_model["count"] >= 10]  # Filter models with enough data
    consistency_by_model = consistency_by_model.sort_values("mean", ascending=True)

    # Use brand colors from utils
    colors = [get_model_color(model_name) for model_name in consistency_by_model["model_name"]]

    fig.add_trace(
        go.Bar(
            x=consistency_by_model["mean"],
            y=consistency_by_model["model_name"],
            orientation='h',
            text=[f"{rate:.1%} (n={count})" for rate, count in zip(consistency_by_model["mean"], consistency_by_model["count"])],
            textposition="outside",
            marker=dict(color=colors)
        )
    )

    fig.update_layout(
        title="Bet-Edge Consistency Rate by Model",
        xaxis_title="Consistency Rate",
        yaxis_title="Model",
        height=max(500, len(consistency_by_model) * 35),  # Increased height for better readability
        width=1000,  # Increased width for longer model names
        margin=dict(l=200, r=100, t=80, b=60)  # Increased left margin for model names
    )

    apply_template(fig, title="Bet-Edge Consistency Rate by Model", width=1000, height=max(500, len(consistency_by_model) * 35))
    return fig


def create_kelly_comparison_chart(bet_edge_df: pd.DataFrame) -> go.Figure:
    """Create scatter plot comparing actual bets vs Kelly optimal bets."""
    fig = go.Figure()

    if bet_edge_df.empty:
        fig.add_annotation(text="No betting data found",
                          xref="paper", yref="paper", x=0.5, y=0.5)
        apply_template(fig, title="Actual Bets vs Kelly Optimal Bets", width=800, height=600)
        return fig

    fig.add_trace(
        go.Scatter(
            x=bet_edge_df["kelly_bet"],
            y=bet_edge_df["bet_amount"],
            mode="markers",
            marker=dict(
                color=bet_edge_df["confidence"],
                colorscale="Viridis",
                size=6,
                opacity=0.6,
                colorbar=dict(title="Confidence Level")
            ),
            text=[f"Model: {model}<br>Edge: {edge:.3f}<br>Confidence: {conf}"
                  for model, edge, conf in zip(bet_edge_df["model_name"], bet_edge_df["edge"], bet_edge_df["confidence"])],
            hovertemplate="%{text}<extra></extra>",
            name="Actual vs Kelly"
        )
    )

    # Add diagonal for perfect Kelly following
    kelly_range = [-1, 1]
    fig.add_trace(
        go.Scatter(
            x=kelly_range, y=kelly_range,
            mode="lines",
            name="Perfect Kelly Alignment",
            line=dict(dash="dash", color="red", width=2),
        )
    )

    fig.update_layout(
        title="Actual Bets vs Kelly Optimal Bets",
        xaxis_title="Kelly Optimal Bet",
        yaxis_title="Actual Bet Amount",
        height=600,
        width=800
    )

    apply_template(fig, title="Actual Bets vs Kelly Optimal Bets", width=800, height=600)
    return fig


def create_probability_calibration_chart(bet_edge_df: pd.DataFrame) -> go.Figure:
    """Create scatter plot comparing estimated probabilities vs market prices."""
    fig = go.Figure()

    if bet_edge_df.empty:
        fig.add_annotation(text="No betting data found",
                          xref="paper", yref="paper", x=0.5, y=0.5)
        apply_template(fig, title="Model Probability Estimates vs Market Prices", width=800, height=600)
        return fig

    for i, model_name in enumerate(bet_edge_df["model_name"].unique()):
        model_data = bet_edge_df[bet_edge_df["model_name"] == model_name]
        color = get_model_color(model_name)

        fig.add_trace(
            go.Scatter(
                x=model_data["market_price"],
                y=model_data["estimated_prob"],
                mode="markers",
                name=model_name,
                marker=dict(
                    color=color,
                    size=6,
                    opacity=0.7
                ),
                text=[f"Model: {model_name}<br>Market: {market:.3f}<br>Estimated: {est:.3f}<br>Edge: {edge:.3f}<br>Confidence: {conf}"
                      for market, est, edge, conf in zip(
                          model_data["market_price"],
                          model_data["estimated_prob"],
                          model_data["edge"],
                          model_data["confidence"]
                      )],
                hovertemplate="%{text}<extra></extra>"
            )
        )

    # Add diagonal line for perfect calibration
    perfect_line = [0, 1]
    fig.add_trace(
        go.Scatter(
            x=perfect_line, y=perfect_line,
            mode="lines",
            name="Perfect Calibration",
            line=dict(dash="dash", color="black", width=2),
        )
    )

    # Add lines for common bias patterns
    # Overconfidence bias (estimates more extreme than market)
    fig.add_trace(
        go.Scatter(
            x=[0, 0.5, 1], y=[0, 0.3, 0.7],
            mode="lines",
            name="Underconfidence Pattern",
            line=dict(dash="dot", color="gray", width=1),
            opacity=0.5
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[0, 0.5, 1], y=[0, 0.7, 1],
            mode="lines",
            name="Overconfidence Pattern",
            line=dict(dash="dot", color="orange", width=1),
            opacity=0.5
        )
    )

    fig.update_layout(
        title="Model Probability Estimates vs Market Prices",
        xaxis_title="Market Price",
        yaxis_title="Estimated Probability",
        height=600,
        width=800,
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1])
    )

    apply_template(fig, title="Model Probability Estimates vs Market Prices", width=800, height=600)
    return fig


def create_consistency_vs_7day_returns_chart(bet_edge_df: pd.DataFrame, backend_data) -> go.Figure:
    """Create scatter plot showing correlation between consistency rate and 7-day returns."""
    fig = go.Figure()

    if bet_edge_df.empty:
        fig.add_annotation(text="No betting data found",
                          xref="paper", yref="paper", x=0.5, y=0.5)
        apply_template(fig, title="Consistency Rate vs 7-Day Returns", width=900, height=600)
        return fig

    # Calculate consistency rate by model
    consistency_by_model = bet_edge_df.groupby("model_name")["consistent"].agg(["mean", "count"]).reset_index()
    consistency_by_model = consistency_by_model[consistency_by_model["count"] >= 10]  # Filter models with enough data

    # Get 7-day returns from performance data
    model_returns = []
    for model_name in consistency_by_model["model_name"]:
        # Find corresponding performance data
        for model_id, performance in backend_data.performance_per_model.items():
            if performance.model_name == model_name:
                seven_day_return = performance.average_returns.seven_day_return
                model_returns.append({
                    "model_name": model_name,
                    "consistency_rate": consistency_by_model[consistency_by_model["model_name"] == model_name]["mean"].iloc[0],
                    "seven_day_return": seven_day_return,
                    "n_decisions": consistency_by_model[consistency_by_model["model_name"] == model_name]["count"].iloc[0]
                })
                break

    returns_df = pd.DataFrame(model_returns)

    if returns_df.empty:
        fig.add_annotation(text="No performance data found",
                          xref="paper", yref="paper", x=0.5, y=0.5)
        apply_template(fig, title="Consistency Rate vs 7-Day Returns", width=900, height=600)
        return fig

    fig.add_trace(
        go.Scatter(
            x=returns_df["consistency_rate"],
            y=returns_df["seven_day_return"],
            mode="markers+text",
            text=returns_df["model_name"],
            textposition="top center",
            textfont=dict(size=10),
            hovertext=[f"{name}<br>Consistency: {cons:.1%}<br>7d Return: {ret:.1%}<br>n={n}"
                      for name, cons, ret, n in zip(
                          returns_df["model_name"],
                          returns_df["consistency_rate"],
                          returns_df["seven_day_return"],
                          returns_df["n_decisions"]
                      )],
            hovertemplate="%{hovertext}<extra></extra>",
            marker=dict(
                size=12,
                color="#4169E1",  # Consistent blue color
                opacity=0.8
            ),
            showlegend=False
        )
    )

    # Add correlation trend line
    if len(returns_df) > 1:
        correlation = returns_df["consistency_rate"].corr(returns_df["seven_day_return"])
        z = np.polyfit(returns_df["consistency_rate"], returns_df["seven_day_return"], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(returns_df["consistency_rate"].min(), returns_df["consistency_rate"].max(), 100)

        fig.add_trace(
            go.Scatter(
                x=x_trend,
                y=p(x_trend),
                mode="lines",
                name=f"Trend (r={correlation:.3f})",
                line=dict(dash="dash", color="red", width=2)
            )
        )

    fig.update_layout(
        title="Consistency Rate vs 7-Day Average Returns",
        xaxis_title="Bet-Edge Consistency Rate",
        yaxis_title="7-Day Average Return",
        height=600,
        width=900,
        xaxis=dict(tickformat=".0%"),
        yaxis=dict(tickformat=".1%")
    )

    apply_template(fig, title="Consistency Rate vs 7-Day Returns", width=900, height=600)
    return fig


def create_consistency_vs_brier_chart(bet_edge_df: pd.DataFrame, backend_data) -> go.Figure:
    """Create scatter plot showing correlation between consistency rate and Brier score."""
    fig = go.Figure()

    if bet_edge_df.empty:
        fig.add_annotation(text="No betting data found",
                          xref="paper", yref="paper", x=0.5, y=0.5)
        apply_template(fig, title="Consistency Rate vs Brier Score", width=900, height=600)
        return fig

    # Calculate consistency rate by model
    consistency_by_model = bet_edge_df.groupby("model_name")["consistent"].agg(["mean", "count"]).reset_index()
    consistency_by_model = consistency_by_model[consistency_by_model["count"] >= 10]  # Filter models with enough data

    # Get Brier scores from performance data
    model_data = []
    for model_name in consistency_by_model["model_name"]:
        # Find corresponding performance data
        for model_id, performance in backend_data.performance_per_model.items():
            if performance.model_name == model_name:
                brier_score = performance.final_brier_score
                model_data.append({
                    "model_name": model_name,
                    "consistency_rate": consistency_by_model[consistency_by_model["model_name"] == model_name]["mean"].iloc[0],
                    "brier_score": brier_score,
                    "n_decisions": consistency_by_model[consistency_by_model["model_name"] == model_name]["count"].iloc[0]
                })
                break

    data_df = pd.DataFrame(model_data)

    if data_df.empty:
        fig.add_annotation(text="No Brier score data found",
                          xref="paper", yref="paper", x=0.5, y=0.5)
        apply_template(fig, title="Consistency Rate vs Brier Score", width=900, height=600)
        return fig

    fig.add_trace(
        go.Scatter(
            x=data_df["consistency_rate"],
            y=data_df["brier_score"],
            mode="markers+text",
            text=data_df["model_name"],
            textposition="top center",
            textfont=dict(size=10),
            hovertext=[f"{name}<br>Consistency: {cons:.1%}<br>Brier Score: {brier:.3f}<br>n={n}"
                      for name, cons, brier, n in zip(
                          data_df["model_name"],
                          data_df["consistency_rate"],
                          data_df["brier_score"],
                          data_df["n_decisions"]
                      )],
            hovertemplate="%{hovertext}<extra></extra>",
            marker=dict(
                size=12,
                color="#4169E1",  # Use consistent blue color instead of colorscale
                opacity=0.8,
                line=dict(width=1, color="black")
            ),
            showlegend=False
        )
    )

    # Add correlation trend line
    if len(data_df) > 1:
        correlation = data_df["consistency_rate"].corr(data_df["brier_score"])
        z = np.polyfit(data_df["consistency_rate"], data_df["brier_score"], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(data_df["consistency_rate"].min(), data_df["consistency_rate"].max(), 100)

        fig.add_trace(
            go.Scatter(
                x=x_trend,
                y=p(x_trend),
                mode="lines",
                name=f"Trend (r={correlation:.3f})",
                line=dict(dash="dash", color="red", width=2)
            )
        )

    fig.update_layout(
        title="Consistency Rate vs Brier Score",
        xaxis_title="Bet-Edge Consistency Rate",
        yaxis_title="Brier Score (lower = better calibration)",
        height=600,
        width=900,
        xaxis=dict(tickformat=".0%")
    )

    apply_template(fig, title="Consistency Rate vs Brier Score", width=900, height=600)
    return fig


def create_brier_vs_market_correlation_chart(bet_edge_df: pd.DataFrame, backend_data) -> go.Figure:
    """Create scatter plot showing Brier score vs correlation between model estimates and market prices."""
    fig = go.Figure()

    if bet_edge_df.empty:
        fig.add_annotation(text="No betting data found",
                          xref="paper", yref="paper", x=0.5, y=0.5)
        apply_template(fig, title="Brier Score vs Model-Market Correlation", width=900, height=600)
        return fig

    # Calculate correlation between model estimates and market prices for each model
    model_correlations = []

    for model_name in bet_edge_df["model_name"].unique():
        model_data = bet_edge_df[bet_edge_df["model_name"] == model_name]

        if len(model_data) >= 10:  # Need enough data points for meaningful correlation
            # Calculate correlation between estimated probabilities and market prices
            correlation = model_data["estimated_prob"].corr(model_data["market_price"])

            if not pd.isna(correlation):
                # Get Brier score from performance data
                brier_score = None
                for model_id, performance in backend_data.performance_per_model.items():
                    if performance.model_name == model_name:
                        brier_score = performance.final_brier_score
                        break

                if brier_score is not None:
                    model_correlations.append({
                        "model_name": model_name,
                        "market_correlation": correlation,
                        "brier_score": brier_score,
                        "n_decisions": len(model_data)
                    })

    if not model_correlations:
        fig.add_annotation(text="Insufficient data for correlation analysis",
                          xref="paper", yref="paper", x=0.5, y=0.5)
        apply_template(fig, title="Brier Score vs Model-Market Correlation", width=900, height=600)
        return fig

    corr_df = pd.DataFrame(model_correlations)

    fig.add_trace(
        go.Scatter(
            x=corr_df["market_correlation"],
            y=corr_df["brier_score"],
            mode="markers+text",
            text=corr_df["model_name"],
            textposition="top center",
            textfont=dict(size=10),
            hovertext=[f"{name}<br>Market Correlation: {corr:.3f}<br>Brier Score: {brier:.3f}<br>n={n}"
                      for name, corr, brier, n in zip(
                          corr_df["model_name"],
                          corr_df["market_correlation"],
                          corr_df["brier_score"],
                          corr_df["n_decisions"]
                      )],
            hovertemplate="%{hovertext}<extra></extra>",
            marker=dict(
                size=12,
                color="#4169E1",  # Consistent blue color
                opacity=0.8,
                line=dict(width=1, color="black")
            ),
            showlegend=False
        )
    )

    # Add correlation trend line
    if len(corr_df) > 1:
        correlation = corr_df["market_correlation"].corr(corr_df["brier_score"])
        z = np.polyfit(corr_df["market_correlation"], corr_df["brier_score"], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(corr_df["market_correlation"].min(), corr_df["market_correlation"].max(), 100)

        fig.add_trace(
            go.Scatter(
                x=x_trend,
                y=p(x_trend),
                mode="lines",
                name=f"Trend (r={correlation:.3f})",
                line=dict(dash="dash", color="red", width=2)
            )
        )

    fig.update_layout(
        title="Brier Score vs Model-Market Correlation",
        xaxis_title="Correlation between Model Estimates and Market Prices",
        yaxis_title="Brier Score (lower = better calibration)",
        height=600,
        width=900,
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, None])
    )

    apply_template(fig, title="Brier Score vs Model-Market Correlation", width=900, height=600)
    return fig


def create_brier_vs_7day_returns_chart(bet_edge_df: pd.DataFrame, backend_data) -> go.Figure:
    """Create scatter plot showing Brier score vs 7-day average returns."""
    fig = go.Figure()

    if bet_edge_df.empty:
        fig.add_annotation(text="No betting data found",
                          xref="paper", yref="paper", x=0.5, y=0.5)
        apply_template(fig, title="Brier Score vs 7-Day Average Returns", width=900, height=600)
        return fig

    # Get model data with both Brier scores and 7-day returns
    model_data = []

    # Get models that have betting decisions
    models_with_decisions = bet_edge_df.groupby("model_name")["consistent"].count()
    models_with_enough_data = models_with_decisions[models_with_decisions >= 10].index

    for model_name in models_with_enough_data:
        # Get performance data
        brier_score = None
        seven_day_return = None

        for model_id, performance in backend_data.performance_per_model.items():
            if performance.model_name == model_name:
                brier_score = performance.final_brier_score
                seven_day_return = performance.average_returns.seven_day_return
                break

        if brier_score is not None and seven_day_return is not None:
            model_data.append({
                "model_name": model_name,
                "brier_score": brier_score,
                "seven_day_return": seven_day_return,
                "n_decisions": models_with_decisions[model_name]
            })

    if not model_data:
        fig.add_annotation(text="No performance data found",
                          xref="paper", yref="paper", x=0.5, y=0.5)
        apply_template(fig, title="Brier Score vs 7-Day Average Returns", width=900, height=600)
        return fig

    data_df = pd.DataFrame(model_data)

    fig.add_trace(
        go.Scatter(
            x=data_df["brier_score"],
            y=data_df["seven_day_return"],
            mode="markers+text",
            text=data_df["model_name"],
            textposition="top center",
            textfont=dict(size=10),
            hovertext=[f"{name}<br>Brier Score: {brier:.3f}<br>7d Return: {ret:.1%}<br>n={n}"
                      for name, brier, ret, n in zip(
                          data_df["model_name"],
                          data_df["brier_score"],
                          data_df["seven_day_return"],
                          data_df["n_decisions"]
                      )],
            hovertemplate="%{hovertext}<extra></extra>",
            marker=dict(
                size=12,
                color="#4169E1",  # Consistent blue color
                opacity=0.8,
                line=dict(width=1, color="black")
            ),
            showlegend=False
        )
    )

    # Add correlation trend line
    if len(data_df) > 1:
        correlation = data_df["brier_score"].corr(data_df["seven_day_return"])
        z = np.polyfit(data_df["brier_score"], data_df["seven_day_return"], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(data_df["brier_score"].min(), data_df["brier_score"].max(), 100)

        fig.add_trace(
            go.Scatter(
                x=x_trend,
                y=p(x_trend),
                mode="lines",
                name=f"Trend (r={correlation:.3f})",
                line=dict(dash="dash", color="red", width=2)
            )
        )

    fig.update_layout(
        title="Brier Score vs 7-Day Average Returns",
        xaxis_title="Brier Score (lower = better calibration)",
        yaxis_title="7-Day Average Return",
        height=600,
        width=900,
        yaxis=dict(tickformat=".1%")
    )

    apply_template(fig, title="Brier Score vs 7-Day Average Returns", width=900, height=600)
    return fig


def create_bet_amount_vs_confidence_chart_single_model(bet_edge_df: pd.DataFrame, model_name: str, edge_threshold: float = 0.05) -> go.Figure:
    """Create scatter plot showing absolute bet amount vs confidence for a single model."""
    fig = go.Figure()

    if bet_edge_df.empty:
        fig.add_annotation(text="No betting data found",
                          xref="paper", yref="paper", x=0.5, y=0.5)
        apply_template(fig, width=600, height=500)
        return fig

    # Filter by edge threshold and selected model
    filtered_df = bet_edge_df[
        (bet_edge_df["abs_edge"] > edge_threshold) &
        (bet_edge_df["model_name"] == model_name)
    ]

    if filtered_df.empty:
        fig.add_annotation(text=f"No data for {model_name} with |edge| > {edge_threshold}",
                          xref="paper", yref="paper", x=0.5, y=0.5)
        apply_template(fig, width=600, height=500)
        return fig

    color = get_model_color(model_name)

    fig.add_trace(
        go.Scatter(
            x=filtered_df["confidence"],
            y=filtered_df["bet_amount"].abs(),
            mode="markers",
            name=f"{model_name} (n={len(filtered_df)})",
            marker=dict(
                color=color,
                size=8,
                opacity=0.7
            ),
            text=[f"Model: {model_name}<br>Confidence: {conf}<br>Bet: {bet:.3f}<br>Edge: {edge:.3f}"
                  for conf, bet, edge in zip(
                      filtered_df["confidence"],
                      filtered_df["bet_amount"],
                      filtered_df["edge"]
                  )],
            hovertemplate="%{text}<extra></extra>"
        )
    )

    # Add reference lines and formatting
    fig.update_layout(
        xaxis_title="Confidence Level",
        yaxis_title="Absolute Bet Amount",
        showlegend=True
    )

    apply_template(fig, width=600, height=500)
    return fig


def create_probability_calibration_chart_single_model(bet_edge_df: pd.DataFrame, model_name: str) -> go.Figure:
    """Create scatter plot comparing estimated probabilities vs market prices for a single model."""
    fig = go.Figure()

    if bet_edge_df.empty:
        fig.add_annotation(text="No betting data found",
                          xref="paper", yref="paper", x=0.5, y=0.5)
        apply_template(fig, width=600, height=500)
        return fig

    # Filter by selected model
    filtered_df = bet_edge_df[bet_edge_df["model_name"] == model_name]

    if filtered_df.empty:
        fig.add_annotation(text=f"No data for {model_name}",
                          xref="paper", yref="paper", x=0.5, y=0.5)
        apply_template(fig, width=600, height=500)
        return fig

    color = get_model_color(model_name)

    fig.add_trace(
        go.Scatter(
            x=filtered_df["market_price"],
            y=filtered_df["estimated_prob"],
            mode="markers",
            name=f"{model_name} (n={len(filtered_df)})",
            marker=dict(
                color=color,
                size=8,
                opacity=0.7
            ),
            text=[f"Model: {model_name}<br>Market: {market:.3f}<br>Estimated: {est:.3f}<br>Edge: {edge:.3f}<br>Confidence: {conf}"
                  for market, est, edge, conf in zip(
                      filtered_df["market_price"],
                      filtered_df["estimated_prob"],
                      filtered_df["edge"],
                      filtered_df["confidence"]
                  )],
            hovertemplate="%{text}<extra></extra>"
        )
    )

    # Add perfect calibration line (diagonal)
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect Calibration",
            line=dict(color="gray", dash="dash"),
            showlegend=True
        )
    )

    fig.update_layout(
        xaxis_title="Market Price",
        yaxis_title="Model Estimated Probability",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        showlegend=True
    )

    apply_template(fig, width=600, height=500)
    return fig


def create_bet_amount_vs_confidence_chart_filtered(bet_edge_df: pd.DataFrame, models: List[str], edge_threshold: float = 0.05) -> go.Figure:
    """Create scatter plot showing absolute bet amount vs confidence for specific models."""
    fig = go.Figure()

    if bet_edge_df.empty:
        fig.add_annotation(text="No betting data found",
                          xref="paper", yref="paper", x=0.5, y=0.5)
        apply_template(fig, width=800, height=600)
        return fig

    # Filter by edge threshold and selected models
    filtered_df = bet_edge_df[
        (bet_edge_df["abs_edge"] > edge_threshold) &
        (bet_edge_df["model_name"].isin(models))
    ]

    if filtered_df.empty:
        fig.add_annotation(text=f"No data for selected models with |edge| > {edge_threshold}",
                          xref="paper", yref="paper", x=0.5, y=0.5)
        apply_template(fig, width=800, height=600)
        return fig

    for i, model_name in enumerate(models):
        model_data = filtered_df[filtered_df["model_name"] == model_name]
        if len(model_data) == 0:
            continue

        color = get_model_color(model_name)

        fig.add_trace(
            go.Scatter(
                x=model_data["confidence"],
                y=model_data["bet_amount"].abs(),
                mode="markers",
                name=f"{model_name} (n={len(model_data)})",
                marker=dict(
                    color=color,
                    size=6,
                    opacity=0.6
                ),
                text=[f"Model: {model_name}<br>Confidence: {conf}<br>Bet: {bet:.3f}<br>Edge: {edge:.3f}"
                      for conf, bet, edge in zip(
                          model_data["confidence"],
                          model_data["bet_amount"],
                          model_data["edge"]
                      )],
                hovertemplate="%{text}<extra></extra>"
            )
        )

    # Add reference lines and formatting
    fig.update_layout(
        xaxis_title="Confidence Level",
        yaxis_title="Absolute Bet Amount",
        showlegend=True
    )

    apply_template(fig, width=800, height=600)
    return fig


def create_probability_calibration_chart_filtered(bet_edge_df: pd.DataFrame, models: List[str]) -> go.Figure:
    """Create scatter plot comparing estimated probabilities vs market prices for specific models."""
    fig = go.Figure()

    if bet_edge_df.empty:
        fig.add_annotation(text="No betting data found",
                          xref="paper", yref="paper", x=0.5, y=0.5)
        apply_template(fig, width=800, height=600)
        return fig

    # Filter by selected models
    filtered_df = bet_edge_df[bet_edge_df["model_name"].isin(models)]

    if filtered_df.empty:
        fig.add_annotation(text="No data for selected models",
                          xref="paper", yref="paper", x=0.5, y=0.5)
        apply_template(fig, width=800, height=600)
        return fig

    for i, model_name in enumerate(models):
        model_data = filtered_df[filtered_df["model_name"] == model_name]
        if len(model_data) == 0:
            continue

        color = get_model_color(model_name)

        fig.add_trace(
            go.Scatter(
                x=model_data["market_price"],
                y=model_data["estimated_prob"],
                mode="markers",
                name=f"{model_name} (n={len(model_data)})",
                marker=dict(
                    color=color,
                    size=6,
                    opacity=0.7
                ),
                text=[f"Model: {model_name}<br>Market: {market:.3f}<br>Estimated: {est:.3f}<br>Edge: {edge:.3f}<br>Confidence: {conf}"
                      for market, est, edge, conf in zip(
                          model_data["market_price"],
                          model_data["estimated_prob"],
                          model_data["edge"],
                          model_data["confidence"]
                      )],
                hovertemplate="%{text}<extra></extra>"
            )
        )

    # Add perfect calibration line (diagonal)
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect Calibration",
            line=dict(color="gray", dash="dash"),
            showlegend=True
        )
    )

    fig.update_layout(
        xaxis_title="Market Price",
        yaxis_title="Model Estimated Probability",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        showlegend=True
    )

    apply_template(fig, width=800, height=600)
    return fig


def create_bet_amount_vs_confidence_chart(bet_edge_df: pd.DataFrame, edge_threshold: float = 0.05) -> go.Figure:
    """Create scatter plot showing absolute bet amount vs confidence by model, filtered by edge threshold."""
    fig = go.Figure()

    if bet_edge_df.empty:
        fig.add_annotation(text="No betting data found",
                          xref="paper", yref="paper", x=0.5, y=0.5)
        apply_template(fig, title=f"Absolute Bet Amount vs Confidence (|Edge| > {edge_threshold:.0%})", width=800, height=600)
        return fig

    # Filter by edge threshold - only include decisions with significant edge
    filtered_df = bet_edge_df[bet_edge_df["abs_edge"] > edge_threshold]

    if filtered_df.empty:
        fig.add_annotation(text=f"No data with |edge| > {edge_threshold}",
                          xref="paper", yref="paper", x=0.5, y=0.5)
        apply_template(fig, title=f"Absolute Bet Amount vs Confidence (|Edge| > {edge_threshold:.0%})", width=800, height=600)
        return fig

    for i, model_name in enumerate(filtered_df["model_name"].unique()):
        model_data = filtered_df[filtered_df["model_name"] == model_name]
        if len(model_data) == 0:
            continue

        color = get_model_color(model_name)

        fig.add_trace(
            go.Scatter(
                x=model_data["confidence"],
                y=model_data["bet_amount"].abs(),
                mode="markers",
                name=f"{model_name} (n={len(model_data)})",
                marker=dict(
                    color=color,
                    size=6,
                    opacity=0.6
                ),
                text=[f"Model: {model_name}<br>Confidence: {conf}<br>Bet: {bet:.3f}<br>Edge: {edge:.3f}"
                      for conf, bet, edge in zip(
                          model_data["confidence"],
                          model_data["bet_amount"],
                          model_data["edge"]
                      )],
                hovertemplate="%{text}<extra></extra>"
            )
        )

    fig.update_layout(
        title=f"Absolute Bet Amount vs Confidence by Model<br><sub>Filtered: |Edge| > {edge_threshold:.2%} ({len(filtered_df)}/{len(bet_edge_df)} decisions)</sub>",
        xaxis_title="Confidence Level (1-10)",
        yaxis_title="Absolute Bet Amount",
        height=600,
        width=800,
        xaxis=dict(range=[0, 10]),
        yaxis=dict(range=[0, None])
    )

    apply_template(fig, title=f"Absolute Bet Amount vs Confidence (|Edge| > {edge_threshold:.0%})", width=800, height=600)
    return fig


def create_decision_consistency_over_time_chart(bet_edge_df: pd.DataFrame) -> go.Figure:
    """Create visualization showing how consistent model decisions are over time for the same events."""
    fig = go.Figure()

    if bet_edge_df.empty:
        fig.add_annotation(text="No betting data found",
                          xref="paper", yref="paper", x=0.5, y=0.5)
        apply_template(fig, title="Decision Consistency Over Time", width=1200, height=800)
        return fig

    # Group by model and event to find cases with multiple time points
    bet_edge_df["date_parsed"] = pd.to_datetime(bet_edge_df["date"])
    multi_decision_events = []

    for model_name in bet_edge_df["model_name"].unique():
        model_data = bet_edge_df[bet_edge_df["model_name"] == model_name]

        for event_id in model_data["event_id"].unique():
            event_decisions = model_data[model_data["event_id"] == event_id].sort_values("date_parsed")

            if len(event_decisions) >= 2:  # Need at least 2 decisions over time
                multi_decision_events.append({
                    "model_name": model_name,
                    "event_id": event_id,
                    "decisions": event_decisions
                })

    if not multi_decision_events:
        fig.add_annotation(text="No events with multiple decisions over time found",
                          xref="paper", yref="paper", x=0.5, y=0.5)
        apply_template(fig, title="Decision Consistency Over Time", width=1200, height=800)
        return fig

    colors = px.colors.qualitative.Set3

    # Create subplots for different metrics
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Estimated Probability Over Time",
            "Bet Amount Over Time",
            "Confidence Over Time",
            "Edge Over Time"
        ],
        vertical_spacing=0.12
    )

    model_color_map = {model: get_model_color(model) for model in bet_edge_df["model_name"].unique()}

    # Plot time series for each model-event combination
    for item in multi_decision_events[:20]:  # Limit to top 20 for readability
        model_name = item["model_name"]
        event_id = item["event_id"]
        decisions = item["decisions"]
        color = model_color_map[model_name]

        trace_name = f"{model_name} - Event {event_id[-4:]}"  # Show last 4 chars of event ID

        # 1. Estimated Probability
        fig.add_trace(
            go.Scatter(
                x=decisions["date_parsed"],
                y=decisions["estimated_prob"],
                mode="lines+markers",
                name=trace_name,
                line=dict(color=color),
                showlegend=True,
                legendgroup=model_name
            ),
            row=1, col=1
        )

        # 2. Bet Amount
        fig.add_trace(
            go.Scatter(
                x=decisions["date_parsed"],
                y=decisions["bet_amount"],
                mode="lines+markers",
                name=trace_name,
                line=dict(color=color),
                showlegend=False,
                legendgroup=model_name
            ),
            row=1, col=2
        )

        # 3. Confidence
        fig.add_trace(
            go.Scatter(
                x=decisions["date_parsed"],
                y=decisions["confidence"],
                mode="lines+markers",
                name=trace_name,
                line=dict(color=color),
                showlegend=False,
                legendgroup=model_name
            ),
            row=2, col=1
        )

        # 4. Edge
        fig.add_trace(
            go.Scatter(
                x=decisions["date_parsed"],
                y=decisions["edge"],
                mode="lines+markers",
                name=trace_name,
                line=dict(color=color),
                showlegend=False,
                legendgroup=model_name
            ),
            row=2, col=2
        )

    fig.update_layout(
        title=f"Decision Consistency Over Time by Model and Event<br><sub>{len(multi_decision_events)} model-event pairs with multiple decisions</sub>",
        height=800,
        width=1200
    )

    # Update axes
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=2)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=2)

    fig.update_yaxes(title_text="Estimated Probability", row=1, col=1)
    fig.update_yaxes(title_text="Bet Amount", row=1, col=2)
    fig.update_yaxes(title_text="Confidence (1-10)", row=2, col=1)
    fig.update_yaxes(title_text="Edge", row=2, col=2)

    apply_template(fig, title="Decision Consistency Over Time", width=1200, height=800)
    return fig


def create_decision_correlation_matrix(bet_edge_df: pd.DataFrame) -> go.Figure:
    """Create correlation matrix showing consistency metrics for same model-event pairs over time."""
    fig = go.Figure()

    if bet_edge_df.empty:
        fig.add_annotation(text="No betting data found",
                          xref="paper", yref="paper", x=0.5, y=0.5)
        apply_template(fig, title="Decision Consistency Correlation Matrix", width=800, height=600)
        return fig

    bet_edge_df["date_parsed"] = pd.to_datetime(bet_edge_df["date"])

    correlation_data = []

    for model_name in bet_edge_df["model_name"].unique():
        model_data = bet_edge_df[bet_edge_df["model_name"] == model_name]

        for event_id in model_data["event_id"].unique():
            event_decisions = model_data[model_data["event_id"] == event_id].sort_values("date_parsed")

            if len(event_decisions) >= 3:  # Need at least 3 points for meaningful correlation
                # Calculate correlations between consecutive time periods
                metrics = ["estimated_prob", "bet_amount", "confidence", "edge"]

                for metric in metrics:
                    values = event_decisions[metric].values
                    if len(set(values)) > 1:  # Need variation to calculate correlation
                        # Calculate autocorrelation (correlation with previous value)
                        if len(values) >= 2:
                            prev_values = values[:-1]
                            curr_values = values[1:]
                            if len(set(prev_values)) > 1 and len(set(curr_values)) > 1:
                                correlation = np.corrcoef(prev_values, curr_values)[0, 1]
                                if not np.isnan(correlation):
                                    correlation_data.append({
                                        "model_name": model_name,
                                        "event_id": event_id,
                                        "metric": metric,
                                        "correlation": correlation,
                                        "n_decisions": len(event_decisions)
                                    })

    if not correlation_data:
        fig.add_annotation(text="Insufficient data for correlation analysis",
                          xref="paper", yref="paper", x=0.5, y=0.5)
        apply_template(fig, title="Decision Consistency Correlation Matrix", width=800, height=600)
        return fig

    corr_df = pd.DataFrame(correlation_data)

    # Create correlation summary by model and metric
    summary = corr_df.groupby(["model_name", "metric"])["correlation"].agg(["mean", "count"]).reset_index()
    summary = summary[summary["count"] >= 2]  # Need at least 2 event-model pairs

    if summary.empty:
        fig.add_annotation(text="Insufficient data for correlation summary",
                          xref="paper", yref="paper", x=0.5, y=0.5)
        apply_template(fig, title="Decision Consistency Correlation Matrix", width=800, height=600)
        return fig

    # Pivot for heatmap
    heatmap_data = summary.pivot(index="model_name", columns="metric", values="mean")

    fig.add_trace(
        go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale="RdYlGn",
            zmid=0,
            colorbar=dict(title="Avg Correlation"),
            text=np.round(heatmap_data.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10}
        )
    )

    fig.update_layout(
        title="Decision Consistency Correlation Matrix<br><sub>Average correlation between consecutive decisions for same model-event pairs</sub>",
        xaxis_title="Decision Metric",
        yaxis_title="Model",
        height=600,
        width=800
    )

    apply_template(fig, title="Decision Consistency Correlation Matrix", width=800, height=600)
    return fig


def create_calibration_trends_visualization(bet_edge_df: pd.DataFrame) -> go.Figure:
    """Create visualization showing calibration improvements over time."""
    if bet_edge_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No betting data found",
                          xref="paper", yref="paper", x=0.5, y=0.5)
        apply_template(fig, title="Consistency Rate Trends Over Time by Model", width=1000, height=500)
        return fig

    bet_edge_df["date_parsed"] = pd.to_datetime(bet_edge_df["date"])

    # Group by model and time (weekly for better granularity)
    model_trends = []
    for model_name in bet_edge_df["model_name"].unique():
        model_data = bet_edge_df[bet_edge_df["model_name"] == model_name]
        weekly_stats = model_data.groupby(model_data["date_parsed"].dt.to_period("W")).agg({
            "consistent": "mean",
            "abs_edge": "mean",
            "confidence": "mean",
            "bet_vs_kelly": "mean"
        }).reset_index()
        weekly_stats["model_name"] = model_name
        weekly_stats["date_parsed"] = weekly_stats["date_parsed"].dt.to_timestamp()
        if len(weekly_stats) >= 3:  # Only include models with enough time series data
            model_trends.append(weekly_stats)

    if not model_trends:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient data for trend analysis",
                          xref="paper", yref="paper", x=0.5, y=0.5)
        apply_template(fig, title="Consistency Rate Trends Over Time by Model", width=1000, height=500)
        return fig

    trends_df = pd.concat(model_trends, ignore_index=True)

    fig = go.Figure()

    for i, model_name in enumerate(trends_df["model_name"].unique()):
        model_data = trends_df[trends_df["model_name"] == model_name]
        color = get_model_color(model_name)

        fig.add_trace(
            go.Scatter(
                x=model_data["date_parsed"],
                y=model_data["consistent"],
                mode="lines+markers",
                name=model_name,
                line=dict(color=color, width=2),
                marker=dict(size=6)
            )
        )

    fig.update_layout(
        title="Consistency Rate Trends Over Time by Model",
        xaxis_title="Date",
        yaxis_title="Consistency Rate",
        height=500,
        width=1000,
        yaxis=dict(range=[0, 1])
    )

    apply_template(fig, title="Consistency Rate Trends Over Time by Model", width=1000, height=500)
    return fig


def main():
    """Main analysis function."""
    print("Loading backend data...")
    backend_data = get_data_for_backend()

    print(f"Loaded {len(backend_data.model_decisions)} model decisions")
    print(f"Loaded {len(backend_data.events)} events")

    # Create output directory
    output_dir = Path("analyses/market_dynamics")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create frontend JSON output directory
    frontend_json_dir = FRONTEND_PUBLIC_PATH / "market_dynamics"
    frontend_json_dir.mkdir(parents=True, exist_ok=True)

    # Analysis 1: Price adjustment speed
    print("Analyzing price adjustment dynamics...")
    price_volatility_df = analyze_price_volatility_around_events(backend_data)

    if not price_volatility_df.empty:
        print(f"Found {len(price_volatility_df)} significant price movements")

        # Create price adjustment visualization
        fig1 = create_price_adjustment_visualization(price_volatility_df)
        fig1.update_layout(width=1400, height=900)
        fig1.write_html(output_dir / "price_adjustment_dynamics.html")
        with open(output_dir / "price_adjustment_dynamics.json", "w") as f:
            f.write(fig1.to_json())
        with open(frontend_json_dir / "price_adjustment_dynamics.json", "w") as f:
            f.write(fig1.to_json())
        print("Saved price_adjustment_dynamics.html and .json (local + frontend)")

        # Save summary statistics
        summary_stats = {
            "avg_adjustment_speed": price_volatility_df["adjustment_speed"].mean(),
            "median_adjustment_speed": price_volatility_df["adjustment_speed"].median(),
            "avg_initial_shock": price_volatility_df["initial_change"].abs().mean(),
            "total_significant_moves": len(price_volatility_df)
        }
        print(f"Average adjustment speed: {summary_stats['avg_adjustment_speed']:.2f} days")
        print(f"Median adjustment speed: {summary_stats['median_adjustment_speed']:.2f} days")

    # Analysis 2: Bet vs Edge consistency
    print("Analyzing bet vs edge consistency...")
    bet_edge_df = analyze_bet_vs_edge_consistency(backend_data)

    if not bet_edge_df.empty:
        print(f"Analyzed {len(bet_edge_df)} betting decisions (excluding baseline)")

        # Create individual visualizations

        # 1. Edge vs Bet Scatter Plot
        fig_edge_bet = create_edge_vs_bet_scatter(bet_edge_df)
        fig_edge_bet.update_layout(width=1000, height=700)
        fig_edge_bet.write_html(output_dir / "edge_vs_bet_scatter.html")
        with open(output_dir / "edge_vs_bet_scatter.json", "w") as f:
            f.write(fig_edge_bet.to_json())
        with open(frontend_json_dir / "edge_vs_bet_scatter.json", "w") as f:
            f.write(fig_edge_bet.to_json())
        print("Saved edge_vs_bet_scatter.html and .json (local + frontend)")

        # 2. Consistency Rate Chart
        fig_consistency = create_consistency_rate_chart(bet_edge_df)
        fig_consistency.update_layout(width=1200, height=700)  # Better readability
        fig_consistency.write_html(output_dir / "consistency_rates.html")
        with open(output_dir / "consistency_rates.json", "w") as f:
            f.write(fig_consistency.to_json())
        with open(frontend_json_dir / "consistency_rates.json", "w") as f:
            f.write(fig_consistency.to_json())
        print("Saved consistency_rates.html and .json (local + frontend)")

        # 3. Kelly Comparison Chart
        fig_kelly = create_kelly_comparison_chart(bet_edge_df)
        fig_kelly.update_layout(width=1000, height=700)
        fig_kelly.write_html(output_dir / "kelly_comparison.html")
        with open(output_dir / "kelly_comparison.json", "w") as f:
            f.write(fig_kelly.to_json())
        with open(frontend_json_dir / "kelly_comparison.json", "w") as f:
            f.write(fig_kelly.to_json())
        print("Saved kelly_comparison.html and .json (local + frontend)")

        # 4. Probability Calibration Chart
        fig_prob_cal = create_probability_calibration_chart(bet_edge_df)
        fig_prob_cal.update_layout(width=1000, height=700)
        fig_prob_cal.write_html(output_dir / "probability_calibration.html")
        with open(output_dir / "probability_calibration.json", "w") as f:
            f.write(fig_prob_cal.to_json())
        with open(frontend_json_dir / "probability_calibration.json", "w") as f:
            f.write(fig_prob_cal.to_json())
        print("Saved probability_calibration.html and .json (local + frontend)")

        # 5. Calibration Trends
        fig_trends = create_calibration_trends_visualization(bet_edge_df)
        fig_trends.update_layout(width=1200, height=600)
        fig_trends.write_html(output_dir / "calibration_trends.html")
        with open(output_dir / "calibration_trends.json", "w") as f:
            f.write(fig_trends.to_json())
        with open(frontend_json_dir / "calibration_trends.json", "w") as f:
            f.write(fig_trends.to_json())
        print("Saved calibration_trends.html and .json (local + frontend)")

        # 6. Consistency vs 7-Day Returns Correlation
        fig_consistency_7day = create_consistency_vs_7day_returns_chart(bet_edge_df, backend_data)
        fig_consistency_7day.update_layout(width=1100, height=700)
        fig_consistency_7day.write_html(output_dir / "consistency_vs_7day_returns.html")
        with open(output_dir / "consistency_vs_7day_returns.json", "w") as f:
            f.write(fig_consistency_7day.to_json())
        with open(frontend_json_dir / "consistency_vs_7day_returns.json", "w") as f:
            f.write(fig_consistency_7day.to_json())
        print("Saved consistency_vs_7day_returns.html and .json (local + frontend)")

        # 6b. Consistency vs Brier Score Correlation
        fig_consistency_brier = create_consistency_vs_brier_chart(bet_edge_df, backend_data)
        fig_consistency_brier.update_layout(width=1100, height=700)
        fig_consistency_brier.write_html(output_dir / "consistency_vs_brier_score.html")
        with open(output_dir / "consistency_vs_brier_score.json", "w") as f:
            f.write(fig_consistency_brier.to_json())
        with open(frontend_json_dir / "consistency_vs_brier_score.json", "w") as f:
            f.write(fig_consistency_brier.to_json())
        print("Saved consistency_vs_brier_score.html and .json (local + frontend)")

        # 6c. Brier Score vs Market Correlation
        fig_brier_market_corr = create_brier_vs_market_correlation_chart(bet_edge_df, backend_data)
        fig_brier_market_corr.update_layout(width=1100, height=700)
        fig_brier_market_corr.write_html(output_dir / "brier_vs_market_correlation.html")
        with open(output_dir / "brier_vs_market_correlation.json", "w") as f:
            f.write(fig_brier_market_corr.to_json())
        with open(frontend_json_dir / "brier_vs_market_correlation.json", "w") as f:
            f.write(fig_brier_market_corr.to_json())
        print("Saved brier_vs_market_correlation.html and .json (local + frontend)")

        # 6d. Brier Score vs 7-Day Returns
        fig_brier_7day = create_brier_vs_7day_returns_chart(bet_edge_df, backend_data)
        fig_brier_7day.update_layout(width=1100, height=700)
        fig_brier_7day.write_html(output_dir / "brier_vs_7day_returns.html")
        with open(output_dir / "brier_vs_7day_returns.json", "w") as f:
            f.write(fig_brier_7day.to_json())
        with open(frontend_json_dir / "brier_vs_7day_returns.json", "w") as f:
            f.write(fig_brier_7day.to_json())
        print("Saved brier_vs_7day_returns.html and .json (local + frontend)")

        # 7. Bet Amount vs Confidence (with edge threshold)
        edge_threshold = 0.20  # 20% edge threshold
        fig_bet_confidence = create_bet_amount_vs_confidence_chart(bet_edge_df, edge_threshold)
        fig_bet_confidence.update_layout(width=1000, height=700)
        fig_bet_confidence.write_html(output_dir / "bet_amount_vs_confidence.html")
        with open(output_dir / "bet_amount_vs_confidence.json", "w") as f:
            f.write(fig_bet_confidence.to_json())
        with open(frontend_json_dir / "bet_amount_vs_confidence.json", "w") as f:
            f.write(fig_bet_confidence.to_json())
        print("Saved bet_amount_vs_confidence.html and .json (local + frontend)")

        # 8. Decision Consistency Over Time
        fig_decision_time = create_decision_consistency_over_time_chart(bet_edge_df)
        fig_decision_time.update_layout(width=1400, height=900)
        fig_decision_time.write_html(output_dir / "decision_consistency_over_time.html")
        with open(output_dir / "decision_consistency_over_time.json", "w") as f:
            f.write(fig_decision_time.to_json())
        with open(frontend_json_dir / "decision_consistency_over_time.json", "w") as f:
            f.write(fig_decision_time.to_json())
        print("Saved decision_consistency_over_time.html and .json (local + frontend)")

        # 9. Decision Correlation Matrix
        fig_correlation_matrix = create_decision_correlation_matrix(bet_edge_df)
        fig_correlation_matrix.update_layout(width=1000, height=700)
        fig_correlation_matrix.write_html(output_dir / "decision_correlation_matrix.html")
        with open(output_dir / "decision_correlation_matrix.json", "w") as f:
            f.write(fig_correlation_matrix.to_json())
        with open(frontend_json_dir / "decision_correlation_matrix.json", "w") as f:
            f.write(fig_correlation_matrix.to_json())
        print("Saved decision_correlation_matrix.html and .json (local + frontend)")

        # 10. Individual Bet Amount vs Confidence Charts for Side-by-Side Display
        # GPT-5 chart
        fig_bet_confidence_gpt5 = create_bet_amount_vs_confidence_chart_single_model(bet_edge_df, "GPT-5", edge_threshold)
        with open(frontend_json_dir / "bet_amount_vs_confidence_gpt5.json", "w") as f:
            f.write(fig_bet_confidence_gpt5.to_json())
        print("Saved bet_amount_vs_confidence_gpt5.json (frontend)")

        # GPT-5 Mini chart
        fig_bet_confidence_gpt5_mini = create_bet_amount_vs_confidence_chart_single_model(bet_edge_df, "GPT-5 Mini", edge_threshold)
        with open(frontend_json_dir / "bet_amount_vs_confidence_gpt5_mini.json", "w") as f:
            f.write(fig_bet_confidence_gpt5_mini.to_json())
        print("Saved bet_amount_vs_confidence_gpt5_mini.json (frontend)")

        # 11. Individual Probability Calibration Charts for Side-by-Side Display
        # GPT-5 chart
        fig_prob_cal_gpt5 = create_probability_calibration_chart_single_model(bet_edge_df, "GPT-5")
        with open(frontend_json_dir / "probability_calibration_gpt5.json", "w") as f:
            f.write(fig_prob_cal_gpt5.to_json())
        print("Saved probability_calibration_gpt5.json (frontend)")

        # GPT-OSS 120B chart
        fig_prob_cal_gpt_oss = create_probability_calibration_chart_single_model(bet_edge_df, "GPT-OSS 120B")
        with open(frontend_json_dir / "probability_calibration_gpt_oss_120b.json", "w") as f:
            f.write(fig_prob_cal_gpt_oss.to_json())
        print("Saved probability_calibration_gpt_oss_120b.json (frontend)")

        # Print summary statistics
        overall_consistency = bet_edge_df["consistent"].mean()
        avg_edge = bet_edge_df["abs_edge"].mean()
        avg_kelly_deviation = bet_edge_df["bet_vs_kelly"].mean()

        print(f"Overall consistency rate: {overall_consistency:.2%}")
        print(f"Average absolute edge: {avg_edge:.3f}")
        print(f"Average Kelly deviation: {avg_kelly_deviation:.3f}")

        # Consistency by model
        model_consistency = bet_edge_df.groupby("model_name")["consistent"].agg(["mean", "count"])
        print("\nConsistency by model (excluding baseline):")
        for model_name, stats in model_consistency.iterrows():
            if stats["count"] >= 10:  # Only show models with enough data
                print(f"  {model_name}: {stats['mean']:.2%} (n={stats['count']})")

    print(f"\nAnalysis complete! Results saved to:")
    print(f"  HTML files: {output_dir}")
    print(f"  JSON files: {output_dir} and {frontend_json_dir}")


if __name__ == "__main__":
    main()