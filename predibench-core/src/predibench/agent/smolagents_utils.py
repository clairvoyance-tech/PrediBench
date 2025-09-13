import json
import os
import logging
import textwrap
from datetime import date
from typing import Any, Literal
import urllib.parse

import numpy as np
import requests
from openai import OpenAI
from predibench.agent.models import (
    MarketInvestmentDecision,
    ModelInfo,
    SingleModelDecision,
)
from predibench.logger_config import get_logger
from predibench.storage_utils import write_to_storage, read_from_storage, file_exists_in_storage
from pydantic import BaseModel
from smolagents import (
    ChatMessage,
    CodeAgent,
    LiteLLMModel,
    TokenUsage,
    Tool,
    ToolCallingAgent,
    VisitWebpageTool,
    tool,
)
from tenacity import (
    retry,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random,
    before_sleep_log,
    after_log,
)
from markdownify import markdownify as md
from predibench.agent.tools_common import web_search_common, visit_webpage_scrapfly

logger = get_logger(__name__)

BET_DESCRIPTION = """1. market_id (str): The market ID
2. rationale (str): Explanation for your decision and why you think this market is mispriced (or correctly priced if skipping). Write at least a few sentences. If you take a strong bet, make sure to highlight the facts you know/value that the market doesn't.
3. odds (float, 0 to 1): The odds you think the market will settle at (your true probability estimate)
4. confidence (int, 0 to 10): Your confidence in the odds and your bet. Should be between 0 (absolute uncertainty, you shouldn't bet if you're not confident) and 10 (absolute certainty, then you can bet high).
5. bet (float, -1 to 1): The amount in dollars that you bet on this market (can be negative if you want to buy the opposite of the market)"""


class VisitWebpageToolSaveSources(VisitWebpageTool):
    def __init__(self):
        super().__init__()
        self.sources: list[str] = []

    def forward(self, url: str) -> str:
        content = super().forward(url)
        self.sources.append(url)
        self.sources = list(dict.fromkeys(self.sources))
        return content


class VisitWebpageToolWithSources(Tool):
    def __init__(self):
        super().__init__()
        self.sources: list[str] = []
    
    def _add_source(self, url: str) -> None:
        """Add a URL to the sources list, avoiding duplicates."""
        self.sources.append(url)
        self.sources = list(dict.fromkeys(self.sources))
    

class GoogleSearchTool(Tool):
    name = "web_search"
    description = """Performs Google web search and returns top results."""
    inputs = {
        "query": {"type": "string", "description": "The search query to perform."},
    }
    output_type = "string"
    

    def __init__(self, provider: Literal["serpapi", "bright_data", "serper"], cutoff_date: date | None):
        super().__init__()
        self.provider = provider
        if provider == "serpapi":
            self.organic_key = "organic_results"
            self.api_key = os.getenv("SERPAPI_API_KEY")
        elif provider == "bright_data":
            self.organic_key = "organic"
            self.api_key = os.getenv("BRIGHT_SERPER_API_KEY")
        elif provider == "serper":
            self.organic_key = "organic"
            self.api_key = os.getenv("SERPER_API_KEY")
        else:
            raise ValueError(f"Invalid provider: {provider}")
        
        self.cutoff_date = cutoff_date
        self.sources: list[str] = []

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_random(min=10, max=120),
        retry=retry_if_exception_type((requests.exceptions.RequestException,)),
        reraise=True,
    )
    def forward(self, query: str) -> str:
        markdown, sources = web_search_common(query=query, provider=self.provider, cutoff_date=self.cutoff_date)
        self.sources.extend(sources)
        self.sources = list(dict.fromkeys(self.sources))
        return markdown


class BrightDataVisitWebpageTool(VisitWebpageToolWithSources):
    name = "visit_webpage"
    description = (
        "Visits a webpage at the given url and reads its content as a markdown string. Use this to browse webpages."
    )
    inputs = {
        "url": {
            "type": "string",
            "description": "The url of the webpage to visit.",
        }
    }
    output_type = "string"

    def __init__(self, zone: str | None = None):
        super().__init__()
        # Expect a full Bright Data browser CDP endpoint in env
        self.endpoint = os.getenv("BRIGHT_DATA_BROWSER_ENDPOINT")
        if not self.endpoint:
            raise ValueError(
                "Missing BRIGHT_DATA_BROWSER_ENDPOINT environment variable with the full wss CDP URL."
            )

    def forward(self, url: str) -> str:
        # Create a fresh Playwright session and browser connection for each call
        from playwright.sync_api import sync_playwright

        p = sync_playwright().start()
        browser = None
        page = None
        html = ""
        try:
            browser = p.chromium.connect_over_cdp(self.endpoint)
            page = browser.new_page()
            page.goto(url, timeout=2 * 60_000, wait_until="load")
            html = page.content()
        finally:
            # Cleanup resources regardless of success/failure
            try:
                if page is not None:
                    page.close()
            except Exception:
                pass
            try:
                if browser is not None:
                    browser.close()
            except Exception:
                pass
            try:
                p.stop()
            except Exception:
                pass

        if not html:
            raise ValueError("Failed to retrieve page content via Bright Data Playwright.")

        markdown_content = md(html, heading_style="ATX")
        self._add_source(url)
        return markdown_content


class ScrapeDoVisitWebpageTool(VisitWebpageToolWithSources):
    name = "visit_webpage"
    description = (
        "Visits a webpage at the given url and reads its content as a markdown string. Use this to browse webpages."
    )
    inputs = {
        "url": {
            "type": "string",
            "description": "The url of the webpage to visit.",
        }
    }
    output_type = "string"

    def __init__(self, render: bool = True):
        super().__init__()
        self.api_key = os.getenv("SCRAPE_DO_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Missing SCRAPE_DO_API_KEY environment variable for Scrape.do API."
            )
        self.render = render

    def forward(self, url: str) -> str:
        encoded_target_url = urllib.parse.quote(url, safe="")
        render_param = "true" if self.render else "false"
        scrape_api_url = (
            f"http://api.scrape.do/?url={encoded_target_url}&token={self.api_key}"
            f"&output=markdown&render={render_param}"
        )

        response = requests.get(scrape_api_url)
        if response.status_code != 200:
            logger.error(
                f"Scrape.do error {response.status_code}: {response.text}"
            )
            raise ValueError(response.text)

        # Scrape.do already returns markdown when output=markdown
        self._add_source(url)
        return response.text


class ScrapflyVisitWebPageTool(VisitWebpageToolWithSources):
    name = "visit_webpage"
    description = (
        "Visits a webpage at the given url and reads its content as a markdown string. Use this to browse webpages."
    )
    inputs = {
        "url": {
            "type": "string",
            "description": "The url of the webpage to visit.",
        }
    }
    output_type = "string"

    def __init__(self, asp: bool = True, render_js: bool = True):
        super().__init__()
        self.asp = asp
        self.render_js = render_js

    def forward(self, url: str) -> str:
        markdown_content = visit_webpage_scrapfly(url, asp=self.asp, render_js=self.render_js)
        self._add_source(url)
        return markdown_content

def parse_market_decisions_and_unallocated(
    market_decisions: list[dict], unallocated_capital: float
) -> tuple[list[MarketInvestmentDecision], float]:
    """Validate and parse market decisions + unallocated capital into structured objects."""
    # Manual type checks for market_decisions
    if not isinstance(market_decisions, list):
        raise TypeError(
            f"market_decisions must be a list, even an empty list got {type(market_decisions).__name__}"
        )
    if len(market_decisions) == 0:
        return [], 1.0

    for i, decision in enumerate(market_decisions):
        if not isinstance(decision, dict):
            raise TypeError(
                f"market_decisions[{i}] must be a dict, got {type(decision).__name__}"
            )

    # Manual type checks for unallocated_capital
    if not isinstance(unallocated_capital, (int, float)):
        raise TypeError(
            f"unallocated_capital must be a float or int, got {type(unallocated_capital).__name__}"
        )

    try:
        unallocated_capital = float(unallocated_capital)
    except (ValueError, TypeError) as e:
        raise TypeError(f"unallocated_capital cannot be converted to float: {e}")

    validated_decisions = []
    total_allocated = 0.0
    assert unallocated_capital >= 0.0, "Unallocated capital cannot be negative"

    for decision_dict in market_decisions:
        # Check required fields
        assert "market_id" in decision_dict, (
            "A key 'market_id' is required for each market decision"
        )
        assert "rationale" in decision_dict, (
            "A key 'rationale' is required for each market decision"
        )
        assert "odds" in decision_dict, (
            "A key 'odds' is required for each market decision"
        )
        assert "bet" in decision_dict, (
            "A key 'bet' is required for each market decision"
        )
        assert "confidence" in decision_dict, (
            "A key 'confidence' is required for each market decision"
        )

        # Validate market_id is not empty
        if not decision_dict["market_id"] or decision_dict["market_id"].strip() == "":
            raise ValueError("Market ID cannot be empty or whitespace only")

        # Validate rationale is not empty
        if not decision_dict["rationale"] or decision_dict["rationale"].strip() == "":
            raise ValueError(
                f"Rationale cannot be empty for market {decision_dict['market_id']}"
            )
        assert -1.0 <= decision_dict["bet"] <= 1.0, (
            f"Your bet must be between -1.0 and 1.0, got {decision_dict['bet']} for market {decision_dict['market_id']}"
        )
        assert 0.0 <= decision_dict["odds"] <= 1.0, (
            f"Your estimated odds must be between 0.0 and 1.0, got {decision_dict['odds']} for market {decision_dict['market_id']}"
        )
        try:
            assert int(decision_dict["confidence"]) == float(
                decision_dict["confidence"]
            )
            decision_dict["confidence"] = int(decision_dict["confidence"])
            assert 0 <= decision_dict["confidence"] <= 10
        except Exception:
            raise TypeError(
                f"Your confidence must be between an integer 0 and 10, got {decision_dict['confidence']} for market {decision_dict['market_id']}"
            )

        model_decision = SingleModelDecision(
            rationale=decision_dict["rationale"],
            odds=decision_dict["odds"],
            bet=decision_dict["bet"],
            confidence=decision_dict["confidence"],
        )
        total_allocated += np.abs(decision_dict["bet"])

        market_decision = MarketInvestmentDecision(
            market_id=decision_dict["market_id"],
            model_decision=model_decision,
        )
        validated_decisions.append(market_decision)

    # assert total_allocated + unallocated_capital == 1.0, (
    #     f"Total capital allocation, calculated as the sum of all absolute value of bets, must equal 1.0, got {total_allocated + unallocated_capital:.3f} (allocated: {total_allocated:.3f}, unallocated: {unallocated_capital:.3f})"
    # ) @ NOTE: models like gpt-4.1 are too dumb to respect this constraint, let's just enforce it a-posteriori with rescaling if needed.
    # if total_allocated + unallocated_capital != 1.0:
    #     for decision in validated_decisions:
    #         decision.model_decision.bet = decision.model_decision.bet / (
    #             total_allocated + unallocated_capital
    #         )
    # NOTE: don't rescale in the end

    return validated_decisions, unallocated_capital


@tool
def final_answer(
    market_decisions: list[dict], unallocated_capital: float
) -> tuple[list[MarketInvestmentDecision], float]:
    """
    Use this tool to validate and return the final event decisions for all relevant markets.
    Provide decisions for all markets you want to bet on.

    Args:
        market_decisions (list[dict]): List of market decisions. Each dict should contain:
            1. market_id (str): The market ID
            2. rationale (str): Explanation for your decision and why you think this market is mispriced (or correctly priced if skipping). Write at least a few sentences. If you take a strong bet, make sure to highlight the facts you know/value that the market doesn't.
            3. odds (float, 0 to 1): The odds you think the market will settle at (your true probability estimate)
            4. confidence (int, 0 to 10): Your confidence in the odds and your bet. Should be between 0 (absolute uncertainty, you shouldn't bet if you're not confident) and 10 (absolute certainty, then you can bet high).
            5. bet (float, -1 to 1): The amount in dollars that you bet on this market (can be negative if you want to buy the opposite of the market)
        unallocated_capital (float): Fraction of capital not allocated to any bet (0.0 to 1.0)
    """
    return parse_market_decisions_and_unallocated(market_decisions, unallocated_capital)


class ListMarketInvestmentDecisions(BaseModel):
    market_investment_decisions: list[MarketInvestmentDecision]
    unallocated_capital: float


class CompleteMarketInvestmentDecisions(ListMarketInvestmentDecisions):
    full_response: Any
    token_usage: TokenUsage | None = None
    sources_google: list[str] | None = None
    sources_visit_webpage: list[str] | None = None

def _should_retry(exception: Exception) -> bool:
    """Check if the exception is a rate limit error."""
    error_str = str(exception).lower()
    return (
        "BadRequest".lower() in error_str
        or "Bad Request".lower() in error_str
        or "ValidationError".lower() in error_str
        or "ContextError".lower() in error_str
        or "maximum context length".lower() in error_str
        or "context window".lower() in error_str
    )


@retry(
    stop=stop_after_attempt(3),
    retry=retry_if_exception(_should_retry),
    before_sleep=before_sleep_log(logger, logging.ERROR),
    after=after_log(logger, logging.ERROR),
    reraise=False,
)
def run_smolagents(
    model_info: ModelInfo,
    question: str,
    cutoff_date: date | None,
    search_provider: Literal["serpapi", "bright_data", "serper"],
    max_steps: int,
) -> CompleteMarketInvestmentDecisions:
    """Run smolagent for event-level analysis with structured output."""
    model_client = model_info.client
    assert model_client is not None, "Model client is not set"

    prompt = f"""{question}
        
Use the final_answer tool to validate your output before providing the final answer.
The final_answer tool must contain the arguments rationale and decision.
"""
    if cutoff_date is not None:
        assert cutoff_date < date.today()

    google_search_tool = GoogleSearchTool(
        provider=search_provider, cutoff_date=cutoff_date
    )
    visit_webpage_tool = ScrapflyVisitWebPageTool()
    tools = [
        google_search_tool,
        visit_webpage_tool,
        final_answer,
    ]
    if model_info.agent_type == "code":
        agent = CodeAgent(
            tools=tools,
            model=model_client,
            max_steps=max_steps,
            return_full_result=True,
            additional_authorized_imports=["requests"],
        )
    else:  # toolcalling is default
        agent = ToolCallingAgent(
            tools=tools,
            model=model_client,
            max_steps=max_steps,
            return_full_result=True,
        )

    full_result = agent.run(prompt)
    return CompleteMarketInvestmentDecisions(
        market_investment_decisions=full_result.output[0],
        unallocated_capital=full_result.output[1],
        full_response=full_result.steps,
        token_usage=full_result.token_usage,
        sources_google=google_search_tool.sources if google_search_tool.sources else None,
        sources_visit_webpage=visit_webpage_tool.sources if visit_webpage_tool.sources else None,
    )


def _get_cached_research_result(model_info: ModelInfo, target_date: date, event_id: str) -> dict | None:
    """Try to load cached full_response from storage."""
    if model_info is None or target_date is None or event_id is None:
        return None
        
    model_result_path = model_info.get_model_result_path(target_date)
    cache_file_path = model_result_path / f"{event_id}_full_result_cache.json"
    
    if file_exists_in_storage(cache_file_path):
        logger.info(f"Loading cached research result from {cache_file_path}")
        return json.loads(read_from_storage(cache_file_path))
        
    return None

def _save_research_result_to_cache(
    full_response: dict, model_info: ModelInfo, target_date: date, event_id: str
) -> None:
    """Save full_response to cache storage as JSON."""
    if model_info is None or target_date is None or event_id is None:
        return
        
    model_result_path = model_info.get_model_result_path(target_date)
    cache_file_path = model_result_path / f"{event_id}_full_result_cache.json"
    
    write_to_storage(cache_file_path, json.dumps(full_response, indent=2))
    logger.info(f"Saved research result to cache: {cache_file_path}")


@retry(
    stop=stop_after_attempt(2),
    reraise=False,
)
def structure_final_answer(
    research_output: str,
    original_question: str,
    structured_output_model_id: str = "huggingface/fireworks-ai/Qwen/Qwen3-Coder-480B-A35B-Instruct",
) -> tuple[list[MarketInvestmentDecision], float]:
    structured_model = LiteLLMModel(model_id=structured_output_model_id)

    structured_prompt = textwrap.dedent(f"""
        Based on the following research output, extract the investment decisions for each market:
        


        **ORIGINAL QUESTION AND MARKET CONTEXT:**
        <original_question>
        {original_question}
        </original_question>

        **RESEARCH ANALYSIS OUTPUT:**
        <research_output>
        {research_output}
        </research_output>
                
        Your output should be list of market decisions. Each decision should include:


        {BET_DESCRIPTION}

        Make sure to directly use elements from the research output: return each market decision exactly as is, do not add or change any element, extract everything as-is.

        **OUTPUT FORMAT:**
        Provide a JSON object with:
        - "market_investment_decisions": Array of market decisions
        - "unallocated_capital": Float (0.0 to 1.0) for capital not allocated to any market

        **VALIDATION:**
        - All market IDs must match those in the original question's "AVAILABLE MARKETS" section
        - Sum of absolute bet values + unallocated_capital should equal 1.0
        - All rationales should reflect insights from the research analysis
        - Confidence levels should reflect the certainty of your analysis
        - If no good betting opportunities exist, you may return an empty market_investment_decisions array and set unallocated_capital to 1.0
        """)
    structured_output = structured_model.generate(
        [ChatMessage(role="user", content=structured_prompt)],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "schema": ListMarketInvestmentDecisions.model_json_schema(),
            },
        },
    )

    parsed_output = json.loads(structured_output.content)
    market_investment_decisions_json = parsed_output["market_investment_decisions"]
    unallocated_capital = parsed_output["unallocated_capital"]
    return (
        [
            MarketInvestmentDecision(**decision)
            for decision in market_investment_decisions_json
        ],
        float(unallocated_capital),
    )


def run_openai_deep_research(
    model_id: str,
    question: str,
    model_info: ModelInfo | None = None,
    target_date: date | None = None,
    event_id: str | None = None,
) -> CompleteMarketInvestmentDecisions:
    # Try to load cached result first
    cached_result = _get_cached_research_result(model_info, target_date, event_id)
    if cached_result is not None:
        full_response = cached_result
        research_output = full_response.get("output_text", "")
    else:
        client = OpenAI(timeout=3600)

        full_response = client.responses.create(
            model=model_id,
            input=question
            + "\n\nProvide your detailed analysis and reasoning, then clearly state your final decisions for each market you want to bet on.",
            tools=[
                {"type": "web_search_preview"},
                {"type": "code_interpreter", "container": {"type": "auto"}},
            ],
        )
        research_output = full_response.output_text
        
        # Save to cache before attempting structured output
        _save_research_result_to_cache(full_response.model_dump(), model_info, target_date, event_id)

    # Use structured output to get EventDecisions
    structured_market_decisions, unallocated_capital = structure_final_answer(
        research_output, question
    )
    return CompleteMarketInvestmentDecisions(
        market_investment_decisions=structured_market_decisions,
        unallocated_capital=unallocated_capital,
        full_response=full_response.model_dump() if hasattr(full_response, 'model_dump') else full_response,
        token_usage=TokenUsage(
            input_tokens=full_response.get("usage", {}).get("input_tokens", 0) if isinstance(full_response, dict) else full_response.usage.input_tokens,
            output_tokens=(full_response.get("usage", {}).get("output_tokens", 0) + full_response.get("usage", {}).get("output_tokens_details", {}).get("reasoning_tokens", 0)) if isinstance(full_response, dict) else (full_response.usage.output_tokens + full_response.usage.output_tokens_details.reasoning_tokens),
        ) if full_response is not None else None,
        sources_visit_webpage=full_response.get("citations", None) if isinstance(full_response, dict) else None,
    )


def run_perplexity_deep_research(
    model_id: str,
    question: str,
    model_info: ModelInfo | None = None,
    target_date: date | None = None,
    event_id: str | None = None,
    dummy: bool = False, # SET AT FALSE
) -> CompleteMarketInvestmentDecisions:
    # Try to load cached result first
    cached_result = _get_cached_research_result(model_info, target_date, event_id)
    if cached_result is not None:
        full_response = cached_result
        research_output = full_response["choices"][0]["message"]["content"]
    elif dummy:
        # Return minimal dummy response for testing
        research_output = "dummy"
        full_response = {"dummy": True}
        # Save dummy result to cache
        if model_info is not None and target_date is not None and event_id is not None:
            _save_research_result_to_cache(full_response, model_info, target_date, event_id)
    else:
        url = "https://api.perplexity.ai/chat/completions"

        payload = {
            "model": model_id,
            "messages": [
                {
                    "role": "user",
                    "content": question,
                }
            ],
        }
        headers = {
            "Authorization": f"Bearer {os.getenv('PERPLEXITY_API_KEY')}",
            "Content-Type": "application/json",
        }

        raw_response = requests.post(url, json=payload, headers=headers)
        raw_response.raise_for_status()
        full_response = raw_response.json()
        research_output = full_response["choices"][0]["message"]["content"]
        
        # Save to cache before attempting structured output
        _save_research_result_to_cache(full_response, model_info=model_info, target_date=target_date, event_id=event_id)

    # Handle dummy case with minimal data
    if dummy and research_output == "dummy":
        return CompleteMarketInvestmentDecisions(
            market_investment_decisions=[],
            unallocated_capital=1.0,
            full_response=full_response,
            token_usage=None,
            sources_visit_webpage=full_response.get("citations", None),
        )
    
    structured_market_decisions, unallocated_capital = structure_final_answer(
        research_output, question
    )
    return CompleteMarketInvestmentDecisions(
        market_investment_decisions=structured_market_decisions,
        unallocated_capital=unallocated_capital,
        full_response=full_response,
        token_usage=TokenUsage(
            input_tokens=full_response["usage"]["prompt_tokens"],
            output_tokens=full_response["usage"]["completion_tokens"],
        ) if full_response is not None else None,
        sources_visit_webpage=full_response.get("citations", None),
    )
