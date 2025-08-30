import os
from datetime import datetime
from functools import lru_cache

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from predibench.polymarket_api import (
    Event,
    _HistoricalTimeSeriesRequestParameters,
)

from models.schemas import LeaderboardEntry, Stats
from services.calculations import get_leaderboard, get_events_that_received_predictions
from services.data_loader import load_agent_choices, get_events_by_ids
from services.position_analytics import get_positions_df, get_all_markets_pnls

print("Successfully imported predibench modules")


app = FastAPI(title="Polymarket LLM Benchmark API", version="1.0.0")

# CORS for local development only
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


# Configuration
AGENT_CHOICES_REPO = "Sibyllic/predibench-3"



# API Endpoints
@app.get("/")
def root():
    return {"message": "Polymarket LLM Benchmark API", "version": "1.0.0"}


@app.get("/api/leaderboard", response_model=list[LeaderboardEntry])
def get_leaderboard_endpoint():
    """Get the current leaderboard with LLM performance data"""
    return get_leaderboard()


@app.get("/api/events", response_model=list[Event])
def get_events_endpoint(
    search: str = "",
    sort_by: str = "volume",
    order: str = "desc",
    limit: int = 50,
):
    """Get active Polymarket events with search and filtering"""
    events = get_events_that_received_predictions()

    # Apply search filter
    if search:
        search_lower = search.lower()
        events = [
            event
            for event in events
            if (search_lower in event.title.lower() if event.title else False)
            or (
                search_lower in event.description.lower()
                if event.description
                else False
            )
            or (search_lower in str(event.id).lower())
        ]

    # Apply sorting
    if sort_by == "volume" and hasattr(events[0] if events else None, "volume"):
        events.sort(key=lambda x: x.volume or 0, reverse=(order == "desc"))
    elif sort_by == "date" and hasattr(events[0] if events else None, "end_datetime"):
        events.sort(
            key=lambda x: x.end_datetime or datetime.min, reverse=(order == "desc")
        )

    # Apply limit
    return events[:limit]


@app.get("/api/stats", response_model=Stats)
def get_stats():
    """Get overall benchmark statistics"""
    leaderboard = get_leaderboard()

    return Stats(
        topFinalCumulativePnl=max(entry.final_cumulative_pnl for entry in leaderboard),
        avgPnl=sum(entry.final_cumulative_pnl for entry in leaderboard)
        / len(leaderboard),
        totalTrades=sum(entry.trades for entry in leaderboard),
        totalProfit=sum(entry.profit for entry in leaderboard),
    )


@app.get("/api/model/{model_id}", response_model=LeaderboardEntry)
@lru_cache(maxsize=16)
def get_model_details(model_id: str):
    """Get detailed information for a specific model"""
    leaderboard = get_leaderboard()
    model = next((entry for entry in leaderboard if entry.id == model_id), None)

    if not model:
        return {"error": "Model not found"}

    return model


@lru_cache(maxsize=16)
@app.get("/api/model/{agent_id}/pnl")
def get_model_investment_details(agent_id: str):
    """Get market-level position and PnL data for a specific model"""

    pnl_calculators = get_all_markets_pnls()

    # Get PnL calculator for this agent
    pnl_calculator = pnl_calculators[agent_id]

    # Filter for this specific agent
    positions_df = get_positions_df()
    agent_positions = positions_df[positions_df["model_name"] == agent_id]

    if agent_positions.empty:
        return {"markets": []}

    # Prepare market data with questions
    markets_data = {}

    # Get market questions from events
    events = get_events_that_received_predictions()
    market_dict = {}
    for event in events:
        for market in event.markets:
            market_dict[market.id] = market

    # Process each market this agent traded
    for market_id in agent_positions["market_id"].unique():
        # Get market question
        market_question = market_dict[market_id].question

        # Get price data if available
        price_data = []
        market_prices = pnl_calculator.prices[market_id].fillna(0)
        for date_idx, price in market_prices.items():
            price_data.append(
                {
                    "date": date_idx.strftime("%Y-%m-%d"),
                    "price": float(price),
                }
            )

        # Get position markers, ffill positions
        market_positions = agent_positions[agent_positions["market_id"] == market_id][
            ["date", "choice"]
        ]
        market_positions = pd.concat(
            [
                market_positions,
                pd.DataFrame({"date": [market_prices.index[-1]], "choice": [np.nan]}),
            ]
        )  # Add a last value to allow ffill to work
        market_positions["date"] = pd.to_datetime(market_positions["date"])
        market_positions["choice"] = market_positions["choice"].astype(float)
        market_positions = market_positions.set_index("date")
        market_positions = market_positions.resample("D").ffill(limit=7).reset_index()

        position_markers = []
        for _, pos_row in market_positions.iterrows():
            position_markers.append(
                {
                    "date": pos_row["date"].strftime("%Y-%m-%d"),
                    "position": pos_row["choice"],
                }
            )

        # Get market-specific Profit
        market_pnl = pnl_calculator.pnl[market_id].cumsum().fillna(0)
        pnl_data = []
        for date_idx, pnl_value in market_pnl.items():
            pnl_data.append(
                {"date": date_idx.strftime("%Y-%m-%d"), "pnl": float(pnl_value)}
            )
        markets_data[market_id] = {
            "market_id": market_id,
            "question": market_question,
            "prices": price_data,
            "positions": position_markers,
            "pnl_data": pnl_data,
        }

    return markets_data


@app.get("/api/event/{event_id}")
def get_event_details(event_id: str):
    """Get detailed information for a specific event including all its markets"""
    events_list = get_events_by_ids((event_id,))

    if not events_list:
        return {"error": "Event not found"}

    return events_list[0]


@app.get("/api/event/{event_id}/market_prices")
@lru_cache(maxsize=32)
def get_event_market_prices(event_id: str):
    """Get price history for all markets in an event"""
    events_list = get_events_by_ids((event_id,))

    if not events_list:
        return {}

    event = events_list[0]
    market_prices = {}

    # Get prices for each market in the event
    for market in event.markets:
        clob_token_id = market.outcomes[0].clob_token_id
        price_data = _HistoricalTimeSeriesRequestParameters(
            clob_token_id=clob_token_id,
        ).get_cached_token_timeseries()

        market_prices[market.id] = price_data

    return market_prices


@app.get(
    "/api/event/{event_id}/investment_decisions",
    response_model=list[dict],
)
def get_event_investment_decisions(event_id: str):
    """Get real investment choices for a specific event"""
    # Load agent choices data like in gradio app
    data = load_agent_choices()

    # Working with Pydantic models from GCP
    market_investments = []

    # Get the latest prediction for each agent for this specific event ID
    agent_latest_predictions = {}
    for model_result in data:
        model_name = model_result.model_info.model_pretty_name
        for event_decision in model_result.event_investment_decisions:
            if event_decision.event_id == event_id:
                # Use target_date as a proxy for "latest" (assuming newer dates are more recent)
                if (
                    model_name not in agent_latest_predictions
                    or model_result.target_date
                    > agent_latest_predictions[model_name][0].target_date
                ):
                    agent_latest_predictions[model_name] = (
                        model_result,
                        event_decision,
                    )

    # Extract market decisions from latest predictions
    for model_result, event_decision in agent_latest_predictions.values():
        for market_decision in event_decision.market_investment_decisions:
            market_investments.append(
                {
                    "market_id": market_decision.market_id,
                    "model_name": model_result.model_info.model_pretty_name,
                    "model_id": model_result.model_id,
                    "bet": market_decision.model_decision.bet,
                    "odds": market_decision.model_decision.odds,
                    "confidence": market_decision.model_decision.confidence,
                    "rationale": market_decision.model_decision.rationale,
                    "date": model_result.target_date,
                }
            )

    return market_investments


if __name__ == "__main__":
    import os

    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)