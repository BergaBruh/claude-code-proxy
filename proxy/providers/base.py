from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator

from fastapi import Request


class BaseProvider(ABC):
    """Base class for all backend providers."""

    @abstractmethod
    def configure_request(self, litellm_request: dict, raw_request: Request) -> dict:
        """Add provider-specific keys, URLs, and headers to the litellm request dict.
        Returns the modified request dict."""
        ...

    @property
    def uses_litellm(self) -> bool:
        """Whether this provider goes through litellm for the actual API call."""
        return True

    async def generate(self, litellm_request: dict) -> Any:
        """For non-litellm providers: perform the API call and return a response."""
        raise NotImplementedError

    async def stream(self, litellm_request: dict) -> AsyncGenerator:
        """For non-litellm providers: perform streaming API call."""
        raise NotImplementedError
