"""
Provider factory — the only place that knows which concrete class to build.

Usage:
    from core.provider_factory import get_stt, get_tts, get_llm
    stt = get_stt(settings)
    tts = get_tts(settings)
    llm = get_llm(settings)          # returns BaseLLMProvider
    model = llm.get_model()          # returns LangChain BaseChatModel

Swapping providers:  change LLM_PROVIDER / STT_PROVIDER / TTS_PROVIDER in .env.
No code changes ever needed.
"""

from __future__ import annotations
import logging

from config.settings import Settings
from core.base_stt import BaseSTT
from core.base_tts import BaseTTS
from core.base_llm import BaseLLMProvider

logger = logging.getLogger(__name__)


# ── STT ───────────────────────────────────────────────────────────────────────

def get_stt(settings: Settings) -> BaseSTT:
    provider = settings.stt_provider.lower()
    logger.info("Building STT provider: %s", provider)

    match provider:
        case "whisper_local":
            from providers.stt.whisper_local import WhisperLocalSTT
            return WhisperLocalSTT(settings)

        case "groq":
            from providers.stt.groq_stt import GroqSTT
            return GroqSTT(settings)

        case "sarvam":
            from providers.stt.sarvam_stt import SarvamSTT
            return SarvamSTT(settings)

        case _:
            raise ValueError(
                f"Unknown STT provider '{provider}'. "
                "Valid options: whisper_local | groq | sarvam"
            )


# ── TTS ───────────────────────────────────────────────────────────────────────

def get_tts(settings: Settings) -> BaseTTS:
    provider = settings.tts_provider.lower()
    logger.info("Building TTS provider: %s", provider)

    match provider:
        case "piper":
            from providers.tts.piper_tts import PiperTTS
            return PiperTTS(settings)

        case "groq":
            from providers.tts.groq_tts import GroqTTS
            return GroqTTS(settings)

        case "sarvam":
            from providers.tts.sarvam_tts import SarvamTTS
            return SarvamTTS(settings)

        case _:
            raise ValueError(
                f"Unknown TTS provider '{provider}'. "
                "Valid options: piper | groq | sarvam"
            )


# ── LLM ───────────────────────────────────────────────────────────────────────

def get_llm(settings: Settings) -> BaseLLMProvider:
    provider = settings.llm_provider.lower()
    logger.info("Building LLM provider: %s", provider)

    match provider:
        case "ollama":
            from providers.llm.ollama_llm import OllamaLLMProvider
            return OllamaLLMProvider(settings)

        case "groq":
            from providers.llm.groq_llm import GroqLLMProvider
            return GroqLLMProvider(settings)

        case "openai":
            from providers.llm.openai_llm import OpenAILLMProvider
            return OpenAILLMProvider(settings)

        case _:
            raise ValueError(
                f"Unknown LLM provider '{provider}'. "
                "Valid options: ollama | groq | openai"
            )


# ── LLM with automatic fallback chain ─────────────────────────────────────────

def get_llm_with_fallback(settings: Settings):
    """
    Returns a LangChain model with fallback chain.
    Primary provider → Groq (if key present) → OpenAI (if key present).
    Uses LangChain's built-in .with_fallbacks() so the graph never changes.
    """
    primary_provider = get_llm(settings)
    primary_model    = primary_provider.get_model()

    fallbacks = []

    # Add Groq as fallback if not already primary and key exists
    if settings.llm_provider != "groq" and settings.groq_api_key:
        try:
            from providers.llm.groq_llm import GroqLLMProvider
            fallbacks.append(GroqLLMProvider(settings).get_model())
            logger.info("LLM fallback registered: groq")
        except Exception:
            logger.warning("Could not register Groq fallback")

    # Add OpenAI as final fallback if key exists
    if settings.llm_provider != "openai" and settings.openai_api_key:
        try:
            from providers.llm.openai_llm import OpenAILLMProvider
            fallbacks.append(OpenAILLMProvider(settings).get_model())
            logger.info("LLM fallback registered: openai")
        except Exception:
            logger.warning("Could not register OpenAI fallback")

    if fallbacks:
        return primary_model.with_fallbacks(fallbacks)

    return primary_model
