"""Callback Handler that prints to std out."""

import threading
from typing import Any, Dict, List, Union

from data_utils.db_access import get_db_connection
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from .utils import record_llm_history

MODEL_COST_PER_1K_INPUT_TOKENS = {
    "claude-instant-1.2": 0.0008,
    "claude-2.0": 0.008,
    "claude-2.1": 0.008,
    "claude-3-sonnet-20240229": 0.003,
    "claude-3-opus-20240229": 0.075,
    "claude-3-5-haiku-20241022": 0.001,
    "claude-3-5-sonnet-20241022": 0.003,  ## Upgraded version
    "claude-3-5-sonnet-20240620": 0.003,  ## Previous version
    "claude-3-haiku-20240307": 0.00025,
}

MODEL_COST_PER_1K_OUTPUT_TOKENS = {
    "claude-instant-1.2": 0.0024,
    "claude-2.0": 0.024,
    "claude-2.1": 0.024,
    "claude-3-sonnet-20240229": 0.015,
    "claude-3-opus-20240229": 0.075,
    "claude-3-5-haiku-20241022": 0.005,
    "claude-3-5-sonnet-20241022": 0.015,  ## Upgraded version
    "claude-3-5-sonnet-20240620": 0.015,  ## Previous version
    "claude-3-haiku-20240307": 0.00125,
}


def _get_anthropic_claude_token_cost(prompt_tokens: int, completion_tokens: int, model: Union[str, None]) -> float:
    """Get the cost of tokens for the Claude model."""
    if model not in MODEL_COST_PER_1K_INPUT_TOKENS:
        raise ValueError(
            f"Unknown model: {model}. Please provide a valid Anthropic model name."
            "Known models are: " + ", ".join(MODEL_COST_PER_1K_INPUT_TOKENS.keys())
        )
    return (prompt_tokens / 1000) * MODEL_COST_PER_1K_INPUT_TOKENS[model] + (
        completion_tokens / 1000
    ) * MODEL_COST_PER_1K_OUTPUT_TOKENS[model]


class AnthropicTokenUsageCallbackHandler(BaseCallbackHandler):
    """Callback Handler that tracks anthropic info."""

    prompts: List[str]
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    successful_requests: int = 0
    total_cost: float = 0.0
    llm_history: Dict[str, Any]

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return (
            f"Tokens Used: {self.total_tokens}\n"
            f"\tPrompt Tokens: {self.prompt_tokens}\n"
            f"\tCompletion Tokens: {self.completion_tokens}\n"
            f"Successful Requests: {self.successful_requests}\n"
            f"Total Cost (USD): ${self.total_cost}"
        )

    @property
    def always_verbose(self) -> bool:
        """Whether to call verbose callbacks even if verbose is False."""
        return True

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """Print out the prompts."""
        self.prompts = prompts

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Print out the token."""

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Collect token usage."""
        try:
            generation = response.generations[0][0]
        except IndexError:
            generation = None
        if isinstance(generation, ChatGeneration):
            try:
                message = generation.message
                run_id = message.id
                if isinstance(message, AIMessage):
                    usage_metadata = message.usage_metadata
                    response_metadata = message.response_metadata
                else:
                    usage_metadata = None
                    response_metadata = None
            except AttributeError:
                usage_metadata = None
                response_metadata = None
        else:
            usage_metadata = None
            response_metadata = None
            run_id = None

        # compute tokens and cost for this request
        completion_tokens = (usage_metadata or {}).get("output_tokens", 0)
        prompt_tokens = (usage_metadata or {}).get("input_tokens", 0)
        total_tokens = (usage_metadata or {}).get("total_tokens", 0)
        model = (response_metadata or {}).get("model")
        total_cost = _get_anthropic_claude_token_cost(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=model,
        )
        
        with get_db_connection() as conn:
            self.llm_history = record_llm_history(conn, self.prompts, response, run_id)

        # update shared state behind lock
        with self._lock:
            self.total_cost += total_cost
            self.total_tokens += total_tokens
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            self.successful_requests += 1

    def __copy__(self) -> "AnthropicTokenUsageCallbackHandler":
        """Return a copy of the callback handler."""
        return self

    def __deepcopy__(self, memo: Any) -> "AnthropicTokenUsageCallbackHandler":
        """Return a deep copy of the callback handler."""
        return self
