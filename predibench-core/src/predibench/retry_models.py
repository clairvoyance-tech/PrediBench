import logging
import time
from typing import Generator, Type, TypeVar

import litellm
from smolagents import ChatMessage, ChatMessageStreamDelta, Tool
from smolagents.models import (
    AmazonBedrockModel,
    ApiModel,
    InferenceClientModel,
    LiteLLMModel,
    OpenAIModel,
)
from tenacity import (
    after_log,
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_fixed,
)

from predibench.logger_config import get_logger

litellm.drop_params = True
# NOTE: this is needed for grok 4, because it doesn't support "stop" parameter.

logger = get_logger(__name__)

T = TypeVar("T", bound=ApiModel)


def is_rate_limit_error(exception):
    """Check if the exception is a rate limit error."""
    error_str = str(exception).lower()
    return (
        "429" in error_str
        or "rate limit" in error_str
        or "too many requests" in error_str
        or "rate_limit" in error_str
    )

def remove_content_after_stop_sequences(content: str, stop_sequences: list[str]) -> str:
    """Remove content after any stop sequence is encountered."""
    if not content or not stop_sequences:
        return content
    for stop_seq in stop_sequences:
        split = content.split(stop_seq)
        if len(split) == 0 or len(split) == 1 or len(split) == 2:
            continue
        else:
            # more than 2 stop sequences, the model did not stop properly so we truncate the content
            content = stop_seq.join(split[:2])
    return content


def add_retry_logic(base_class: Type[T], wait_time: float = 0) -> Type[T]:
    """Factory function to add retry logic to any ApiModel class."""

    class ModelWithRetry(base_class):
        @retry(
            stop=stop_after_attempt(5),
            wait=wait_fixed(120),  # 2 minutes
            retry=retry_if_exception(is_rate_limit_error),
            reraise=True,
            before_sleep=before_sleep_log(logger, logging.INFO),
            after=after_log(logger, logging.INFO),
        )
        def generate(
            self,
            messages: list[ChatMessage | dict],
            stop_sequences: list[str] | None = None,
            response_format: dict[str, str] | None = None,
            tools_to_call_from: list[Tool] | None = None,
            **kwargs,
        ) -> ChatMessage:
            time.sleep(wait_time)
            result = super().generate(
                messages=messages,
                stop_sequences=stop_sequences,
                response_format=response_format,
                tools_to_call_from=tools_to_call_from,
                **kwargs,
            )
            result.content = remove_content_after_stop_sequences(content=result.content, stop_sequences=stop_sequences)
            return result

        @retry(
            stop=stop_after_attempt(5),
            wait=wait_fixed(120),  # 2 minutes
            retry=retry_if_exception(is_rate_limit_error),
            reraise=True,
            before_sleep=before_sleep_log(logger, logging.WARNING),
            after=after_log(logger, logging.WARNING),
        )
        def generate_stream(
            self,
            messages: list[ChatMessage | dict],
            stop_sequences: list[str] | None = None,
            response_format: dict[str, str] | None = None,
            tools_to_call_from: list[Tool] | None = None,
            **kwargs,
        ) -> Generator[ChatMessageStreamDelta, None, None]:
            time.sleep(wait_time)
            result = super().generate(
                messages=messages,
                stop_sequences=stop_sequences,
                response_format=response_format,
                tools_to_call_from=tools_to_call_from,
                **kwargs,
            )
            result.content = remove_content_after_stop_sequences(content=result.content, stop_sequences=stop_sequences)
            return result

    ModelWithRetry.__name__ = f"{base_class.__name__}WithRetry"
    ModelWithRetry.__doc__ = (
        f"{base_class.__name__} with tenacity retry logic for rate limiting."
    )

    return ModelWithRetry


InferenceClientModelWithRetry = add_retry_logic(InferenceClientModel)
OpenAIModelWithRetry = add_retry_logic(OpenAIModel)
LiteLLMModelWithRetry = add_retry_logic(LiteLLMModel)
AWSBedrockModelWithRetry = add_retry_logic(AmazonBedrockModel)
