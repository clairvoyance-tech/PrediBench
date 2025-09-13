import os
from datetime import date, timedelta

from dotenv import load_dotenv
from huggingface_hub import login
from smolagents.models import LiteLLMModel

from predibench.agent.models import ModelInfo
from predibench.agent.runner import ModelInvestmentDecisions, run_agent_investments
from predibench.common import get_date_output_path
from predibench.logger_config import get_logger
from predibench.market_selection import choose_events
from predibench.polymarket_data import load_events_from_file
from predibench.retry_models import (
    InferenceClientModelWithRetry,
    LiteLLMModelWithRetry,
    OpenAIModel,
    add_retry_logic,
)
from predibench.storage_utils import file_exists_in_storage

load_dotenv(override=True)
login(os.getenv("HF_TOKEN"))

logger = get_logger(__name__)

OpenAIModelWithRetry = add_retry_logic(OpenAIModel, wait_time=120)
LiteLLMModelWithRetryWait = add_retry_logic(LiteLLMModel, wait_time=120)


def run_investments_for_specific_date(
    models: list[ModelInfo],
    max_n_events: int,
    target_date: date,
    time_until_ending: timedelta,
    force_rewrite: bool = False,
    filter_crypto_events: bool = True,
) -> list[ModelInvestmentDecisions]:
    """Run event-based investment simulation with multiple AI models."""
    logger.info(f"Running investment analysis for {target_date}")

    cache_file_path = get_date_output_path(target_date)
    cache_file_path = cache_file_path / "events.json"

    if file_exists_in_storage(
        cache_file_path, force_rewrite=False
    ):  # IT is very unconvenient to override and lose the events
        logger.info(f"Loading events from cache: {cache_file_path}")
        selected_events = load_events_from_file(cache_file_path)
        if len(selected_events) < max_n_events:
            logger.info(f"Selected {len(selected_events)} events from cache, but expected {max_n_events}")
            selected_events = choose_events(
                target_date=target_date,
                time_until_ending=time_until_ending,
                n_events=max_n_events,
                filter_crypto_events=filter_crypto_events,
                save_path=cache_file_path,
            )
    else:
        logger.info("Fetching events from API")
        selected_events = choose_events(
            target_date=target_date,
            time_until_ending=time_until_ending,
            n_events=max_n_events,
            filter_crypto_events=filter_crypto_events,
            save_path=cache_file_path,
        )

    logger.info(f"Selected {len(selected_events)} events:")
    for event in selected_events:
        logger.info(f"  - {event.title} (Volume: ${event.volume:,.0f})")

    for model in models:
        if model.inference_provider == "openai":
            model.client = OpenAIModelWithRetry(model_id=model.model_id)
        elif model.inference_provider == "anthropic":
            # NOTE: Anthropic allows max 5 requests per minute
            model.client = LiteLLMModelWithRetryWait(
                model_id="anthropic/" + model.model_id
            )
        elif model.inference_provider == "xai":
            model.client = LiteLLMModelWithRetry(model_id="xai/" + model.model_id)
        elif model.inference_provider == "google":
            model.client = OpenAIModelWithRetry(
                model_id=model.model_id,
                api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
                api_key=os.getenv("GEMINI_API_KEY"),
            )
        elif model.open_weights:
            model.client = InferenceClientModelWithRetry(
                model_id=model.model_id,
                provider=model.inference_provider,
            )

    # Having no models could be useful to cache the events
    if models:
        results = run_agent_investments(
            models=models,
            events=selected_events,
            target_date=target_date,
            force_rewrite=force_rewrite,
        )

        logger.info("Investment analysis complete!")

        return results


if __name__ == "__main__":
    # Test with random model to verify new output format
    models = [
        ModelInfo(
            model_id="gpt-5-mini",
            model_pretty_name="GPT-5 Mini",
            inference_provider="openai",
            company_pretty_name="OpenAI",
            sdk="openai",
        ),
    ]

    results = run_investments_for_specific_date(
        models=models,
        time_until_ending=timedelta(days=7 * 6),
        target_date=date.today(),
        max_n_events=10,
        force_rewrite=True,
    )
