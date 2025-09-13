from datetime import date, timedelta

from predibench.common import DATA_PATH
from predibench.invest import run_investments_for_specific_date
from predibench.models import ModelInfo

import time
import pytest

def test_invest():
    models = [
        ModelInfo(
            model_id="openai/gpt-oss-120b",
            model_pretty_name="GPT-OSS 120B",
            inference_provider="fireworks-ai",
            company_pretty_name="OpenAI",
            open_weights=True,
            agent_type="toolcalling",
        ),
    ]
    target_date = date(2025, 7, 16)

    result = run_investments_for_specific_date(
        time_until_ending=timedelta(days=21),
        max_n_events=1,
        models=models,
        target_date=target_date,
        force_rewrite=True,
    )

    models = [
        ModelInfo(
            model_id="openai/gpt-oss-120b",
            model_pretty_name="GPT-OSS 120B",
            inference_provider="fireworks-ai",
            company_pretty_name="OpenAI",
            open_weights=True,
            agent_type="toolcalling",
        ),
    ]
    start_time = time.time()
    results = run_investments_for_specific_date(
        models=models,
        time_until_ending=timedelta(days=7 * 6),
        max_n_events=2,
        target_date=date(2025, 8, 24),
    )
    end_time = time.time()
    assert end_time - start_time < 3
    assert isinstance(result, list)
    if len(result) > 0:
        assert hasattr(result[0], "model_id")
        assert hasattr(result[0], "target_date")


def test_invest_openai():
    models = [
        ModelInfo(
            model_id="gpt-5-mini",
            model_pretty_name="GPT-5 Mini",
            inference_provider="openai",
            company_pretty_name="OpenAI",
            sdk="openai",
            agent_type="toolcalling",
        ),
    ]
    target_date = date.today()
    result = run_investments_for_specific_date(
        models=models,
        time_until_ending=timedelta(days=7 * 6),
        max_n_events=2,
        target_date=target_date,
        force_rewrite=True,
    )

if __name__ == "__main__":
    test_invest_openai()
