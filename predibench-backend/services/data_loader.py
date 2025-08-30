from functools import lru_cache

from predibench.agent.dataclasses import ModelInvestmentDecisions
from predibench.polymarket_api import Event, EventsRequestParameters
from predibench.storage_utils import get_bucket


@lru_cache(maxsize=1)
def load_investment_choices_from_google() -> list[ModelInvestmentDecisions]:
    # Has bucket access, load directly from GCP bucket

    model_results = []
    bucket = get_bucket()
    blobs = bucket.list_blobs(prefix="")

    for blob in blobs:
        if (
            blob.name.endswith(".json")
            and "/" in blob.name
            and "events.json" not in blob.name
        ):
            parts = blob.name.split("/")
            if "events_cache" in blob.name:
                continue
            if (
                len(parts) == 3 and parts[-1] == "model_investment_decisions.json"
            ):  # NOTE: date/model/model_event_decisions.json format
                try:
                    json_content = blob.download_as_text()
                    model_result = ModelInvestmentDecisions.model_validate_json(
                        json_content
                    )
                    model_results.append(model_result)
                except Exception as e:
                    print(f"Error reading {blob.name}: {e}")
                    continue

    # Sort by target_date
    model_results.sort(key=lambda x: x.target_date)
    return model_results


@lru_cache(maxsize=1)
def load_agent_choices():
    """Load agent choices from GCP instead of HuggingFace dataset"""
    return load_investment_choices_from_google()


@lru_cache(maxsize=32)
def get_events_by_ids(event_ids: tuple[str, ...]) -> list[Event]:
    """Cached wrapper for EventsRequestParameters.get_events()"""
    events = []
    for event_id in event_ids:
        events_request_parameters = EventsRequestParameters(
            id=event_id,
            limit=1,
        )
        events.append(events_request_parameters.get_events()[0])
    return events