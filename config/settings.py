"""
Central configuration — single source of truth.
All values come from .env (or environment variables in production).
Import `settings` anywhere in the project — never read os.environ directly.
"""

from __future__ import annotations
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Provider selection ──────────────────────────────────────────────────
    # Change these 3 lines in .env to swap any component — no code changes.
    llm_provider: str = Field("ollama",         description="ollama | groq | openai")
    stt_provider: str = Field("whisper_local",  description="whisper_local | groq | sarvam")
    tts_provider: str = Field("piper",          description="piper | groq | sarvam")

    # ── LLM ────────────────────────────────────────────────────────────────
    ollama_model:  str = Field("gemma4:31b-cloud")
    ollama_host:   str = Field("http://localhost:11434")
    groq_model:    str = Field("llama-3.3-70b-versatile")
    openai_model:  str = Field("gpt-4o")
    llm_temperature: float = Field(0.7)
    max_tool_calls:  int   = Field(3)

    # ── API Keys (repr=False keeps them out of logs) ────────────────────────
    groq_api_key:   str = Field("", repr=False)
    openai_api_key: str = Field("", repr=False)
    sarvam_api_key: str = Field("", repr=False)

    # ── STT ────────────────────────────────────────────────────────────────
    whisper_model:    str   = Field("small")   # tiny | base | small | medium | large-v3
    whisper_device:   str   = Field("cpu")     # cpu | cuda
    whisper_compute:  str   = Field("int8")    # int8 | float16 | float32
    vad_threshold:    float = Field(0.6)

    # ── TTS ────────────────────────────────────────────────────────────────
    piper_voice_en: str = Field("models/piper/en_US-lessac-medium.onnx")
    piper_voice_hi: str = Field("models/piper/hi_IN-hindi_ldcil-medium.onnx")
    tts_target_rate: int = Field(24000)

    # ── Server ─────────────────────────────────────────────────────────────
    ws_host:    str = Field("0.0.0.0")
    ws_port:    int = Field(8765)   # ESP32 raw WebSocket
    api_host:   str = Field("0.0.0.0")
    api_port:   int = Field(8766)   # App Socket.IO
    secret_key: str = Field("change-me-in-production", repr=False)

    # ── Rate limiting ───────────────────────────────────────────────────────
    rate_limit_requests: int = Field(30)   # max requests per window
    rate_limit_window:   int = Field(60)   # seconds

    # ── Hardware ───────────────────────────────────────────────────────────
    esp32_car_url: str = Field("http://robotcar.local")
    car_cmd_timeout: float = Field(0.5)    # seconds

    # ── Logging ────────────────────────────────────────────────────────────
    log_level:    str = Field("INFO")
    log_dir:      str = Field("logs")
    log_max_bytes:  int = Field(5_242_880)   # 5 MB per file
    log_backup_count: int = Field(5)

    # ── PostgreSQL (LangGraph checkpointer) ────────────────────────────────
    postgres_url: str = Field(
        "postgresql://user:password@localhost:5432/agentdb",
        repr=False,
        description="Full DSN — set in .env. Used by LangGraph checkpointer.",
    )
    postgres_enabled: bool = Field(
        False,
        description="Set True only when Postgres is available.",
    )

    # ── MediaPipe ──────────────────────────────────────────────────────────
    face_model_path: str = Field("models/face_landmarker.task")
    num_faces: int = Field(1)


# Singleton — import this everywhere, never instantiate Settings() again.
settings = Settings()
