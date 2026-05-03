"""Centralised settings loaded from environment / .env file."""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Kafka
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    KAFKA_TOPIC_RAW_PACKETS: str = "raw_packets"
    KAFKA_TOPIC_PROCESSED_FEATURES: str = "processed_features"
    KAFKA_TOPIC_ALERTS: str = "alerts"
    KAFKA_GROUP_ID: str = "nids-processor"
    KAFKA_AUTO_OFFSET_RESET: str = "latest"

    # Elasticsearch
    ELASTICSEARCH_HOST: str = "http://localhost:9200"
    ELASTICSEARCH_INDEX_ALERTS: str = "nids-alerts"
    ELASTICSEARCH_INDEX_FLOWS: str = "nids-flows"
    ELASTICSEARCH_INDEX_PROFILES: str = "nids-profiles"

    # Redis
    REDIS_URL: str = "redis://localhost:6379"
    REDIS_CACHE_TTL: int = 3600
    REDIS_PROFILE_TTL: int = 86400

    # Model paths
    MODEL_PATH: str = "./models"
    SEQUENCE_MODEL_PATH: str = "./models/cnn_gru.pt"
    TRANSFORMER_MODEL_PATH: str = "./models/transformer.pt"
    ANOMALY_MODEL_PATH: str = "./models/isolation_forest.joblib"
    AUTOENCODER_PATH: str = "./models/autoencoder.pt"

    # Fusion weights
    FUSION_WEIGHT_ML: float = 0.50
    FUSION_WEIGHT_ANOMALY: float = 0.30
    FUSION_WEIGHT_SIGNATURE: float = 0.20

    # Thresholds
    THRESHOLD_SUSPICIOUS: float = 0.40
    THRESHOLD_MALICIOUS: float = 0.70

    # Capture
    CAPTURE_INTERFACE: str = "eth0"
    CAPTURE_BPF_FILTER: str = "ip"

    # Alerting
    SLACK_BOT_TOKEN: str = ""
    SLACK_CHANNEL_ID: str = ""
    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USER: str = ""
    SMTP_PASSWORD: str = ""
    ALERT_EMAIL_TO: str = ""
    WEBHOOK_URL: str = ""

    # Threat intel
    ABUSEIPDB_API_KEY: str = ""
    VIRUSTOTAL_API_KEY: str = ""
    THREAT_FEED_UPDATE_INTERVAL: int = 3600

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_SECRET_KEY: str = "change-me"
    API_CORS_ORIGINS: str = "http://localhost:3000"

    # Adaptive learning
    DRIFT_DETECTION_WINDOW: int = 1000
    DRIFT_DETECTION_THRESHOLD: float = 0.002
    RETRAIN_INTERVAL_HOURS: int = 24
    MIN_SAMPLES_FOR_RETRAIN: int = 5000

    # Observability
    LOG_LEVEL: str = "INFO"

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.API_CORS_ORIGINS.split(",")]


settings = Settings()
