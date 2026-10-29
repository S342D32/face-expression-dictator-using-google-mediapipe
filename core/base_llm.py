from abc import ABC, abstractmethod
from langchain_core.language_models import BaseChatModel


class BaseLLMProvider(ABC):
    """
    Contract every LLM provider must satisfy.
    Returns a LangChain BaseChatModel so LangGraph works with any backend.
    """

    @abstractmethod
    def get_model(self) -> BaseChatModel:
        """
        Return a LangChain-compatible chat model.
        The caller will bind tools via  model.bind_tools(ALL_TOOLS).
        """
