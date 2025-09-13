
from datetime import date
from typing import Literal
import asyncio

from predibench.agent.models import ModelInfo, MarketInvestmentDecisionRaw
from predibench.agent.smolagents_utils import (
    CompleteMarketInvestmentDecisions,
)

from agents import Agent, FunctionTool, RunConfig, RunContextWrapper, Runner
from agents.extensions.models.litellm_provider import LitellmProvider
from agents.extensions.models.litellm_model import LitellmModel
from pydantic import BaseModel, Field, ConfigDict
from predibench.agent.smolagents_utils import GoogleSearchTool, VisitWebpageTool, final_answer

class SmolagentsToolsContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    smolagent_web_tool: GoogleSearchTool
    smolagent_visit_webpage_tool: VisitWebpageTool


class _WebSearchToolArgs(BaseModel):    
    query: str

async def web_search_tool_forward(ctx: RunContextWrapper[SmolagentsToolsContext], args: str) -> str:
    args_dict = _WebSearchToolArgs.model_validate_json(args)
    return ctx.context.smolagent_web_tool.forward(args_dict.query)

web_search_tool = FunctionTool(
    name=GoogleSearchTool.name,
    description=GoogleSearchTool.description,
    params_json_schema=_WebSearchToolArgs.model_json_schema(),
    on_invoke_tool=web_search_tool_forward,
)


class _VisitWebpageToolArgs(BaseModel):
    url: str

async def visit_webpage_tool_forward(ctx: RunContextWrapper[SmolagentsToolsContext], args: str) -> str:
    args_dict = _VisitWebpageToolArgs.model_validate_json(args)
    return ctx.context.smolagent_visit_webpage_tool.forward(args_dict.url)

visit_webpage_tool = FunctionTool(
    name=VisitWebpageTool.name,
    description=VisitWebpageTool.description,
    params_json_schema=_VisitWebpageToolArgs.model_json_schema(),
    on_invoke_tool=visit_webpage_tool_forward,
)

class _FinalAnswerToolArgs(BaseModel):
    market_investment_decisions: list[MarketInvestmentDecisionRaw] = Field(..., description="List of market decisions. Each decision should contain: market_id, rationale, odds, confidence, bet")
    unallocated_capital: float = Field(..., description="Fraction of capital not allocated to any bet (0.0 to 1.0)")

async def final_answer_tool_forward(ctx: RunContextWrapper[SmolagentsToolsContext], args: str) -> _FinalAnswerToolArgs:
    args_dict = _FinalAnswerToolArgs.model_validate_json(args)
    return args_dict

final_answer_tool = FunctionTool(
    name=final_answer.name,
    description=final_answer.description,
    params_json_schema=_FinalAnswerToolArgs.model_json_schema(),
    on_invoke_tool=final_answer_tool_forward,
)




def run_openai_tools_agent(
    model_info: ModelInfo,
    question: str,
    search_provider: Literal["serpapi", "bright_data", "serper"],
    max_steps: int,
) -> CompleteMarketInvestmentDecisions:
    smolagents_tools_context = SmolagentsToolsContext(
        smolagent_web_tool=GoogleSearchTool(provider=search_provider, cutoff_date=None),
        smolagent_visit_webpage_tool=VisitWebpageTool(),
    )
    agent = Agent(
        name = "openai_tools_agent",
        instructions=question,
        tools=[web_search_tool, visit_webpage_tool, final_answer_tool],
    )
    
    prompt = f"""
Use the final_answer tool to validate your output before providing the final answer.
The final_answer tool must contain the arguments rationale and decision.
"""
    config = RunConfig(model=model_info.model_id, )

    async def run_agent():
        result = await Runner.run(starting_agent=agent, input=prompt, context=smolagents_tools_context, run_config=config)
        return result
    result = asyncio.run(run_agent())
    
    


    return full_result

