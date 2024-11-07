"""Callback Handler that prints to std out."""

import threading
from typing import Any, Dict, List

from data_utils.db_access import get_db_connection
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from .utils import record_llm_history


class RecordTokenUsageCallbackHandler(BaseCallbackHandler):
    """Callback Handler that tracks llm model usage info."""

    prompts: List[str]
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    successful_requests: int = 0

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return (
            f"Tokens Used: {self.total_tokens}\n"
            f"\tPrompt Tokens: {self.prompt_tokens}\n"
            f"\tCompletion Tokens: {self.completion_tokens}\n"
            f"Successful Requests: {self.successful_requests}\n"
        )

    @property
    def always_verbose(self) -> bool:
        """Whether to call verbose callbacks even if verbose is False."""
        return True

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """Print out the prompts."""

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Print out the token."""

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Collect token usage."""

        try:
            if isinstance(response, AIMessage):
                usage_metadata = response.usage_metadata
            else:
                usage_metadata = None
        except AttributeError:
            usage_metadata = None


        # compute tokens and cost for this request
        completion_tokens = (usage_metadata or {}).get("output_tokens", 0)
        prompt_tokens = (usage_metadata or {}).get("input_tokens", 0)
        total_tokens = (usage_metadata or {}).get("total_tokens", 0)
        
        with get_db_connection() as conn:
            self.llm_history = record_llm_history(conn, self.prompts, response)

        # update shared state behind lock
        with self._lock:
            self.total_tokens += total_tokens
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            self.successful_requests += 1

    def __copy__(self) -> "RecordTokenUsageCallbackHandler":
        """Return a copy of the callback handler."""
        return self

    def __deepcopy__(self, memo: Any) -> "RecordTokenUsageCallbackHandler":
        """Return a deep copy of the callback handler."""
        return self
