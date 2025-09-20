import json
import os
from datetime import datetime
from typing import Literal

from apscheduler.schedulers.background import (
    BackgroundScheduler,  # runs tasks in the background
)
from apscheduler.triggers.cron import (
    CronTrigger,  # allows us to specify a recurring time for execution
)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from predibench.agent.models import ModelInvestmentDecisions
from predibench.backend.data_loader import (
    get_data_for_backend,
    load_event_decision_details_from_bucket,
)
from predibench.backend.data_model import (
    BackendData,
    EventBackend,
    FullModelResult,
    LeaderboardEntryBackend,
    ModelPerformanceBackend,
)
from predibench.common import DATA_PATH
from predibench.storage_utils import read_from_storage, write_to_storage
from pydantic import ValidationError

print("Successfully imported predibench modules")


CACHED_DATA: list[
    BackendData
] = []  # A list is a common way to introduce a global variable


class ContactFormSubmission(BaseModel):
    email: str
    message: str


def load_backend() -> BackendData:
    """Load pre-computed backend data from cache or recompute if missing/outdated."""
    cache_file_path = DATA_PATH / "backend_cache.json"
    data = None
    try:
        json_content = read_from_storage(cache_file_path)
        return BackendData.model_validate(json.loads(json_content))
    except (FileNotFoundError, ValidationError, KeyError, json.JSONDecodeError) as e:
        raise e
        print(f"Cache invalid or missing, recomputing backend data: {e}")
        data = get_data_for_backend()
        try:
            write_to_storage(cache_file_path, data.model_dump_json(indent=2))
            print("✅ Wrote refreshed backend cache to storage")
        except Exception as w:
            print(f"⚠️ Could not write backend cache: {w}")
        return data


def update_cache() -> None:
    print("Updating cache")
    if len(CACHED_DATA) == 0:
        print("Cache is empty, loading new data")
        data = load_backend()
        CACHED_DATA.append(data)
    else:
        print("starting cache update")
        data = load_backend()
        CACHED_DATA[0] = data
    print("Cache update complete")


def load_backend_cache() -> BackendData:
    print("Loading backend cache")
    if len(CACHED_DATA) == 0:
        update_cache()
    return CACHED_DATA[0]


# Warm cache at startup (and print status)
update_cache()
print(
    f"✅ Loaded backend cache with {len(CACHED_DATA[0].leaderboard)} leaderboard entries"
)

scheduler = BackgroundScheduler()
trigger = CronTrigger(minute=0)  # every hour
scheduler.add_job(update_cache, trigger)
scheduler.start()
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


# API Endpoints
@app.get("/")
def root():
    return {"message": "Polymarket LLM Benchmark API", "version": "1.0.0"}


@app.get("/api/leaderboard", response_model=list[LeaderboardEntryBackend])
def get_leaderboard_endpoint():
    """Get the current leaderboard with LLM performance data"""
    return load_backend_cache().leaderboard


@app.get("/api/prediction_dates", response_model=list[str])
def get_prediction_dates_endpoint():
    return load_backend_cache().prediction_dates


@app.get("/api/models/ids", response_model=list[str])
def get_all_models_endpoint():
    """Get a list of all model IDs"""
    data = load_backend_cache()
    model_ids = set()
    model_ids.update(data.model_results_by_id.keys())
    return sorted(list(model_ids))


@app.get("/api/model_results", response_model=list[ModelInvestmentDecisions])
def get_model_results_endpoint():
    return load_backend_cache().model_decisions


@app.get("/api/model_results/by_id", response_model=list[ModelInvestmentDecisions])
def get_model_results_by_id_endpoint(model_id: str):
    data = load_backend_cache()
    results = data.model_results_by_id.get(model_id)
    if results is None:
        raise HTTPException(status_code=404, detail="model_id not found")
    return results


@app.get("/api/model_results/by_date", response_model=list[ModelInvestmentDecisions])
def get_model_results_by_date_endpoint(prediction_date: str):
    data = load_backend_cache()
    results = data.model_results_by_date.get(prediction_date)
    if results is None:
        raise HTTPException(status_code=404, detail="prediction_date not found")
    return results


@app.get("/api/model_results/by_id_and_date", response_model=ModelInvestmentDecisions)
def get_model_results_by_id_and_date_endpoint(model_id: str, prediction_date: str):
    data = load_backend_cache()
    by_id = data.model_results_by_id_and_date.get(model_id)
    if by_id is None:
        raise HTTPException(status_code=404, detail="model_id not found")
    result = by_id.get(prediction_date)
    if result is None:
        raise HTTPException(
            status_code=404, detail="prediction_date not found for model_id"
        )
    return result


@app.get("/api/model_results/by_event", response_model=list[ModelInvestmentDecisions])
def get_model_results_by_event_id_endpoint(event_id: str):
    data = load_backend_cache()
    results = data.model_results_by_event_id.get(event_id)
    if results is None:
        raise HTTPException(status_code=404, detail="event_id not found")
    return results


@app.get("/api/performance", response_model=list[ModelPerformanceBackend])
def get_performance_endpoint():
    """Return model performance by day."""
    data = load_backend_cache()
    return list(data.performance_per_model.values())


@app.get("/api/performance/by_model", response_model=ModelPerformanceBackend)
def get_performance_by_model_endpoint(model_id: str):
    """Return performance for a specific model, by day."""
    data = load_backend_cache()
    try:
        return data.performance_per_model[model_id]
    except KeyError as e:
        print(f"Model not found: {e}")
        raise HTTPException(status_code=404, detail="model_id not found")


@app.get("/api/events/by_id", response_model=EventBackend)
def get_event_endpoint(event_id: str):
    """Get a specific event"""
    data = load_backend_cache()
    event = data.event_details.get(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="event_id not found")
    return event


@app.get("/api/events", response_model=list[EventBackend])
def get_events_endpoint(
    search: str = "",
    sort_by: Literal["volume", "date"] = "volume",
    order: Literal["desc", "asc"] = "desc",
    limit: int = 50,
):
    """Get active Polymarket events with search and filtering"""
    events = load_backend_cache().events

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
    if events:
        if sort_by == "volume" and hasattr(events[0], "volume"):
            events.sort(key=lambda x: x.volume or 0, reverse=(order == "desc"))
        elif sort_by == "date" and hasattr(events[0], "end_datetime"):
            events.sort(
                key=lambda x: x.end_datetime or datetime.min, reverse=(order == "desc")
            )

    # Apply limit
    return events[:limit]


@app.get("/api/events/all", response_model=list[EventBackend])
def get_all_events_endpoint():
    """Get all events without filtering"""
    return load_backend_cache().events


@app.get(
    "/api/decision_details/by_model_and_event", response_model=FullModelResult | None
)
def get_decision_details_by_model_and_event_endpoint(
    model_id: str, event_id: str, target_date: str
):
    """Get decision details for a specific model and event"""
    return load_event_decision_details_from_bucket(model_id, event_id, target_date)


@app.post("/api/contact")
def submit_contact_form(submission: ContactFormSubmission):
    """Save contact form submission to storage"""
    try:
        timestamp = datetime.now().isoformat()
        contact_data = {
            "email": submission.email,
            "message": submission.message,
            "timestamp": timestamp,
        }

        filename = f"reach_out/{timestamp.replace(':', '-')}.json"
        file_path = DATA_PATH / filename

        write_to_storage(file_path, json.dumps(contact_data, indent=2))

        return {"status": "success", "message": "Contact form submitted successfully"}
    except Exception as e:
        print(f"Error saving contact form: {e}")
        raise HTTPException(status_code=500, detail="Failed to save contact form")


if __name__ == "__main__":
    import os

    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)

    # ce que j'ai besoin:
    # - récupérer les résultats par model_id, et par event_id
