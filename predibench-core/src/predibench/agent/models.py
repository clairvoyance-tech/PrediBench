from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

from predibench.common import get_date_output_path
from predibench.storage_utils import write_to_storage
from pydantic import BaseModel, Field
from smolagents import Timing, TokenUsage

# NOTE: price ad odd of the 'yes' on any market should be equal, since normalized to 1


class SingleModelDecision(BaseModel):
    rationale: str = Field(..., description="Explanation for your decision and why you think this market is mispriced (or correctly priced if skipping). Write at least a few sentences. If you take a strong bet, make sure to highlight the facts you know/value that the market doesn't.")
    odds: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="The odds you think the market will settle at (your true probability estimate)",
    )
    bet: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="The amount in dollars that you bet on this market (can be negative if you want to buy the opposite of the market)",
    )
    confidence: int = Field(
        ..., ge=0, le=10, description="Your confidence in the odds and your bet. Should be between 0 (absolute uncertainty, you shouldn't bet if you're not confident) and 10 (absolute certainty, then you can bet high)."
    )
    
class MarketInvestmentDecisionRaw(BaseModel):
    market_id: str = Field(..., description="The market ID")
    model_decision: SingleModelDecision = Field(..., description="Model's decision for this market")

class MarketInvestmentDecision(MarketInvestmentDecisionRaw):
    market_question: str | None = None 


class EventInvestmentDecisions(BaseModel):
    event_id: str
    event_title: str
    event_description: str | None = None
    market_investment_decisions: list[
        MarketInvestmentDecision
    ]  # Multiple markets per event
    unallocated_capital: float
    token_usage: TokenUsage | None = None
    timing: Timing | None = None
    sources_google: list[str] | None = None
    sources_visit_webpage: list[str] | None = None

    def normalize_gains(self) -> None:
        """Normalize the bet amounts so that total allocated capital + unallocated capital = 1.0"""
        total_allocated = sum(
            abs(decision.model_decision.bet) 
            for decision in self.market_investment_decisions
        )
        
        total_capital = total_allocated + self.unallocated_capital
        
        if total_capital != 1.0 and total_capital > 0:
            normalization_factor = 1.0 / total_capital
            
            # Normalize all bet amounts
            for decision in self.market_investment_decisions:
                decision.model_decision.bet *= normalization_factor
            
            # Normalize unallocated capital
            self.unallocated_capital *= normalization_factor


class ModelInfo(BaseModel):
    model_id: str
    model_pretty_name: str
    inference_provider: str
    company_pretty_name: str
    open_weights: bool = False
    client: Any | None = None
    agent_type: Literal["code", "toolcalling", "deepresearch"] = "code"
    sdk: Literal["smolagents", "openai"] = "smolagents"
    
    @staticmethod
    def static_get_model_result_path(model_id: str, target_date: date) -> Path:
        """
        Get the path to the model result for a given model and target date.
        """
        date_output_path = get_date_output_path(target_date)
        model_result_path = date_output_path / model_id.replace("/", "--")
        model_result_path.mkdir(parents=True, exist_ok=True)
        return model_result_path

    def get_model_result_path(self, target_date: date) -> Path:
        """
        Get the path to the model result for a given model and target date.
        """
        return ModelInfo.static_get_model_result_path(model_id=self.model_id, target_date=target_date)


class ModelInvestmentDecisions(BaseModel):
    model_id: str
    model_info: ModelInfo
    target_date: date
    decision_datetime: datetime
    event_investment_decisions: list[EventInvestmentDecisions]

    def _save_model_result(self) -> None:
        """Save model result to file."""

        model_result_path = self.model_info.get_model_result_path(self.target_date)

        filename = "model_investment_decisions.json"
        filepath = model_result_path / filename

        content = self.model_dump_json(indent=2, exclude={"model_info": {"client"}})
        write_to_storage(filepath, content)

        print(f"Saved model result to {filepath}")
