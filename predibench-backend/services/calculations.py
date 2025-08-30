from datetime import datetime
from functools import lru_cache

import numpy as np
import pandas as pd
from predibench.brier import BrierScoreCalculator
from predibench.pnl import PnlCalculator, get_pnls

from models.schemas import DataPoint, LeaderboardEntry
from services.data_loader import load_agent_choices, get_events_by_ids


@lru_cache(maxsize=1)
def extract_decisions_data():
    """Extract decisions data with odds and confidence from model results"""
    model_results = load_agent_choices()

    decisions = []

    # Working with Pydantic models from GCP
    for model_result in model_results:
        model_name = model_result.model_info.model_pretty_name
        date = model_result.target_date

        for event_decision in model_result.event_investment_decisions:
            for market_decision in event_decision.market_investment_decisions:
                decisions.append(
                    {
                        "date": date,
                        "market_id": market_decision.market_id,
                        "model_name": model_name,
                        "model_id": model_result.model_id,
                        "odds": market_decision.model_decision.odds,
                        "confidence": market_decision.model_decision.confidence,
                    }
                )

    return pd.DataFrame.from_records(decisions)


def get_pnl_wrapper(
    positions_df: pd.DataFrame,
    write_plots: bool = False,
    end_date: datetime | None = None,
) -> dict[str, PnlCalculator]:
    return get_pnls(positions_df, end_date, write_plots)


@lru_cache(maxsize=1)
def calculate_real_performance():
    """Calculate real Profit and performance metrics exactly like gradio app"""
    model_results = load_agent_choices()

    # Working with Pydantic models from GCP
    print(f"Loaded {len(model_results)} model results from GCP")

    positions = []
    for model_result in model_results:
        model_name = model_result.model_info.model_pretty_name
        date = model_result.target_date

        for event_decision in model_result.event_investment_decisions:
            for market_decision in event_decision.market_investment_decisions:
                positions.append(
                    {
                        "date": date,
                        "market_id": market_decision.market_id,
                        "choice": market_decision.model_decision.bet,
                        "model_name": model_name,
                    }
                )

    positions_df = pd.DataFrame.from_records(positions)
    print(f"Created {len(positions_df)} position records")

    # positions_df = positions_df.pivot(index="date", columns="market_id", values="bet")

    pnl_calculators = get_pnl_wrapper(
        positions_df, write_plots=False, end_date=datetime.today()
    )

    # Create BrierScoreCalculator instances for each agent
    decisions_df = extract_decisions_data()
    brier_calculators = {}
    for model_name, pnl_calculator in pnl_calculators.items():
        # Filter decisions for this agent
        agent_decisions = decisions_df[decisions_df["model_name"] == model_name]
        # Convert to pivot format
        decisions_pivot_df = agent_decisions.pivot(
            index="date", columns="market_id", values="odds"
        )
        # Align with PnL calculator's price data
        decisions_pivot_df = decisions_pivot_df.reindex(
            pnl_calculator.prices.index, method="ffill"
        )
        brier_calculators[model_name] = BrierScoreCalculator(
            decisions_pivot_df, pnl_calculator.prices
        )

    agents_performance = {}
    for model_name, pnl_calculator in pnl_calculators.items():
        brier_calculator = brier_calculators[model_name]
        daily_pnl = pnl_calculator.portfolio_daily_pnl

        # Generate performance history from cumulative Profit
        cumulative_pnl = pnl_calculator.portfolio_cumulative_pnl
        pnl_history = []
        for date_idx, pnl_value in cumulative_pnl.items():
            pnl_history.append(
                DataPoint(date=date_idx.strftime("%Y-%m-%d"), value=float(pnl_value))
            )

        # Calculate metrics exactly like gradio
        final_pnl = float(pnl_calculator.portfolio_cumulative_pnl.iloc[-1])
        sharpe_ratio = (
            float((daily_pnl.mean() / daily_pnl.std()) * np.sqrt(252))
            if daily_pnl.std() > 0
            else 0
        )

        agents_performance[model_name] = {
            "model_name": model_name,
            "final_cumulative_pnl": final_pnl,
            "annualized_sharpe_ratio": sharpe_ratio,
            "pnl_history": pnl_history,
            "daily_cumulative_pnl": pnl_calculator.portfolio_cumulative_pnl.tolist(),
            "dates": [
                d.strftime("%Y-%m-%d")
                for d in pnl_calculator.portfolio_cumulative_pnl.index.tolist()
            ],
            "avg_brier_score": brier_calculator.avg_brier_score,
        }

        print(
            f"Agent {model_name}: Profit={final_pnl:.3f}, Sharpe={sharpe_ratio:.3f}, Brier={brier_calculator.avg_brier_score:.3f}"
        )

    print(f"Calculated performance for {len(agents_performance)} agents")
    return agents_performance


@lru_cache(maxsize=1)
def get_leaderboard() -> list[LeaderboardEntry]:
    real_performance = calculate_real_performance()

    leaderboard = []
    for _, (model_name, metrics) in enumerate(
        sorted(
            real_performance.items(),
            key=lambda x: x[1]["final_cumulative_pnl"],
            reverse=True,
        )
    ):
        # Determine trend
        history = metrics["pnl_history"]
        if len(history) >= 2:
            recent_change = history[-1].value - history[-2].value
            trend = (
                "up"
                if recent_change > 0.1
                else "down"
                if recent_change < -0.1
                else "stable"
            )
        else:
            trend = "stable"

        entry = LeaderboardEntry(
            id=model_name,
            model=model_name,
            final_cumulative_pnl=metrics["final_cumulative_pnl"],
            trades=0,
            profit=0,
            lastUpdated=datetime.now().strftime("%Y-%m-%d"),
            trend=trend,
            pnl_history=metrics["pnl_history"],
            avg_brier_score=metrics["avg_brier_score"],
        )
        leaderboard.append(entry)

    return leaderboard


@lru_cache(maxsize=1)
def get_events_that_received_predictions():
    """Get events based that models ran predictions on"""
    # Load agent choices to see what markets they've been betting on
    data = load_agent_choices()

    # Working with Pydantic models from GCP
    event_ids = set()
    for model_result in data:
        for event_decision in model_result.event_investment_decisions:
            event_ids.add(event_decision.event_id)
    event_ids = tuple(event_ids)

    return get_events_by_ids(event_ids)