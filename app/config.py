import os


class Settings:
    """Load config from environment variables with sensible defaults."""

    # OpenAI
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_realtime_model: str = os.getenv("OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview")
    openai_realtime_url: str = "wss://api.openai.com/v1/realtime"

    # Camera
    camera_index: int = int(os.getenv("CAMERA_INDEX", "0"))

    # Face recognition
    face_confidence_threshold: float = float(os.getenv("FACE_CONFIDENCE_THRESHOLD", "0.45"))
    face_detection_interval_ms: int = int(os.getenv("FACE_DETECTION_INTERVAL_MS", "500"))
    face_det_min_score: float = 0.70   # discard very low-confidence detections

    # Audio — sounddevice device index, -1 means system default
    audio_input_device: int = int(os.getenv("AUDIO_INPUT_DEVICE", "-1"))
    audio_output_device: int = int(os.getenv("AUDIO_OUTPUT_DEVICE", "-1"))
    audio_input_device_name: str = os.getenv("AUDIO_INPUT_DEVICE_NAME", "").strip()
    audio_output_device_name: str = os.getenv("AUDIO_OUTPUT_DEVICE_NAME", "").strip()
    audio_sample_rate: int = 24000     # OpenAI Realtime API expects 24 kHz
    audio_channels: int = 1
    audio_chunk_ms: int = 100          # 100 ms chunks → 2400 frames @ 24 kHz

    # Conversation
    conversation_max_seconds: int = int(os.getenv("CONVERSATION_MAX_SECONDS", "120"))
    cooldown_seconds: int = int(os.getenv("COOLDOWN_SECONDS", "30"))

    # Storage
    database_path: str = os.getenv("DATABASE_PATH", "/app/data/face.db")
    events_dir: str = os.getenv("EVENTS_DIR", "/app/data/events")
    models_dir: str = os.getenv("MODELS_DIR", "/app/models")
    retention_days: int = int(os.getenv("RETENTION_DAYS", "30"))

    # Server
    dashboard_port: int = int(os.getenv("DASHBOARD_PORT", "8000"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()
