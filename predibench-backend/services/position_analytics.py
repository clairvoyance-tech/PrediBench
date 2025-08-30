from datetime import datetime
from functools import lru_cache

import pandas as pd
from predibench.pnl import PnlCalculator

from services.calculations import get_pnl_wrapper
from services.data_loader import load_agent_choices


@lru_cache(maxsize=1)
def get_positions_df():
    # Calculate market-level data
    data = load_agent_choices()

    # Working with Pydantic models from GCP
    positions = []
    for model_result in data:
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

    return pd.DataFrame.from_records(positions)


@lru_cache(maxsize=1)
def get_all_markets_pnls():
    positions_df = get_positions_df()
    pnl_calculators = get_pnl_wrapper(
        positions_df, write_plots=False, end_date=datetime.today()
    )
    return pnl_calculators