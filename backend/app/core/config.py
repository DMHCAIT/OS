"""
Application configuration and settings
"""

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings
    
from typing import List, Optional
import os
from pathlib import Path


class Settings(BaseSettings):
    """Application settings with comprehensive environment variable support"""
    
    # Application
    APP_NAME: str = "Advanced AI Sales Automation"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "production"
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4
    RELOAD: bool = False
    
    # API Configuration
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = "your-super-secret-key-change-this-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440
    ALGORITHM: str = "HS256"
    BCRYPT_ROUNDS: int = 12
    
    # Database Configuration
    MONGODB_URL: str = "mongodb://localhost:27017"
    MONGODB_DATABASE: str = "ai_sales_automation"
    MONGODB_MIN_POOL_SIZE: int = 5
    MONGODB_MAX_POOL_SIZE: int = 50
    DATABASE_NAME: str = "ai_sales_automation"  # Backward compatibility
    
    # Redis Configuration
    REDIS_URL: str = "redis://localhost:6379"
    REDIS_PASSWORD: Optional[str] = None
    REDIS_DB: int = 0
    CACHE_EXPIRE_SECONDS: int = 3600
    CACHE_TTL: int = 3600
    CACHE_MAX_SIZE: int = 10000
    
    # CORS Configuration
    ALLOWED_HOSTS: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    CORS_ORIGINS: str = "http://localhost:3000,https://yourdomain.com"
    CORS_METHODS: str = "GET,POST,PUT,DELETE,OPTIONS"
    CORS_HEADERS: str = "*"
    
    # AI/ML Service API Keys
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    HUGGINGFACE_API_KEY: Optional[str] = None
    GOOGLE_SPEECH_API_KEY: Optional[str] = None
    
    # Voice AI Services
    ELEVENLABS_API_KEY: Optional[str] = None
    AZURE_SPEECH_KEY: Optional[str] = None
    AZURE_SPEECH_REGION: Optional[str] = None
    AUDIO_SAMPLE_RATE: int = 16000
    AUDIO_CHANNELS: int = 1
    VOICE_MODEL_TYPE: str = "transformer"
    
    # Economic and Financial Data APIs
    FRED_API_KEY: Optional[str] = None
    ALPHA_VANTAGE_API_KEY: Optional[str] = None
    QUANDL_API_KEY: Optional[str] = None
    
    # News and Social Media APIs
    NEWS_API_KEY: Optional[str] = None
    TWITTER_BEARER_TOKEN: Optional[str] = None
    GOOGLE_NEWS_API_KEY: Optional[str] = None
    
    # Geographic Services
    GOOGLE_MAPS_API_KEY: Optional[str] = None
    MAPBOX_ACCESS_TOKEN: Optional[str] = None
    
    # Email Configuration
    SMTP_SERVER: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USERNAME: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    FROM_EMAIL: Optional[str] = None
    
    # Monitoring and Logging
    SENTRY_DSN: Optional[str] = None
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    
    # Performance Settings
    MAX_WORKERS: int = 10
    REQUEST_TIMEOUT: int = 30
    CONNECTION_TIMEOUT: int = 10
    RATE_LIMIT_REQUESTS: int = 1000
    RATE_LIMIT_WINDOW: int = 3600
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_PER_HOUR: int = 1000
    
    # ML Model Settings
    MODEL_CACHE_SIZE: int = 1000
    MODEL_UPDATE_INTERVAL: int = 3600
    FEATURE_EXTRACTION_BATCH_SIZE: int = 32
    
    # Intelligence Services Configuration
    MARKET_ANALYSIS_INTERVAL: int = 3600
    COMPETITOR_MONITORING_INTERVAL: int = 1800
    TERRITORY_OPTIMIZATION_SCHEDULE: str = "0 2 * * *"
    SEASONAL_ANALYSIS_SCHEDULE: str = "0 3 1 * *"
    
    # Data Storage Configuration
    MAX_CONVERSATION_HISTORY: int = 1000
    MAX_ANALYSIS_RETENTION_DAYS: int = 365
    BACKUP_RETENTION_DAYS: int = 30
    
    # WebSocket Settings
    WS_MAX_CONNECTIONS: int = 1000
    WS_HEARTBEAT_INTERVAL: int = 30
    
    # Background Tasks
    TASK_QUEUE_WORKERS: int = 4
    TASK_RETRY_ATTEMPTS: int = 3
    TASK_RETRY_DELAY: int = 60
    
    # Phone/SMS Services - FOR AUTOMATIC CALLING
    TWILIO_ACCOUNT_SID: Optional[str] = None
    TWILIO_AUTH_TOKEN: Optional[str] = None
    TWILIO_PHONE_NUMBER: Optional[str] = None
    
    # File Storage
    UPLOAD_DIR: str = "uploads"
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_FILE_TYPES: List[str] = [".csv", ".xlsx", ".json", ".pdf", ".mp3", ".wav"]
    
    # ML Model Configuration
    LEAD_SCORE_THRESHOLD: float = 0.7
    PREDICTION_CONFIDENCE_THRESHOLD: float = 0.8
    
    # Voice AI Configuration - FOR AUTOMATIC CALLING
    VOICE_PROVIDER: str = "elevenlabs"  # elevenlabs, azure, openai
    VOICE_ID: str = "bella"  # Default voice personality
    VOICE_SPEED: float = 1.0
    VOICE_STABILITY: float = 0.75
    VOICE_CLARITY: float = 0.75
    SUPPORTED_LANGUAGES: List[str] = ["en-US", "en-GB", "es-ES", "fr-FR"]
    
    # Feature Flags
    ENABLE_VOICE_AI: bool = True
    ENABLE_MARKET_ANALYSIS: bool = True
    ENABLE_TERRITORY_OPTIMIZATION: bool = True
    ENABLE_SEASONAL_PATTERNS: bool = True
    ENABLE_COMPETITIVE_INTELLIGENCE: bool = True
    ENABLE_REAL_TIME_COACHING: bool = True
    ENABLE_PERFORMANCE_MONITORING: bool = True
    
    # API Documentation
    DOCS_ENABLED: bool = True
    REDOC_ENABLED: bool = True
    
    # Automatic Calling Configuration
    AUTO_DIAL_ENABLED: bool = True
    MAX_CALLS_PER_DAY: int = 100
    CALL_BUSINESS_HOURS_ONLY: bool = True
    BUSINESS_HOURS_START: str = "09:00"
    BUSINESS_HOURS_END: str = "17:00"
    TIME_ZONE: str = "America/New_York"
    LEAD_CALLING_THRESHOLD: int = 70  # Only call leads scoring 70+
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 100
    RATE_LIMIT_PER_HOUR: int = 1000
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/app.log"
    
    # Security
    PASSWORD_MIN_LENGTH: int = 8
    MAX_LOGIN_ATTEMPTS: int = 5
    ACCOUNT_LOCKOUT_DURATION: int = 900  # 15 minutes
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Create settings instance
settings = Settings()

# Create upload directory if it doesn't exist
upload_path = Path(settings.UPLOAD_DIR)
upload_path.mkdir(exist_ok=True)

# Create logs directory if it doesn't exist
log_path = Path(settings.LOG_FILE).parent
log_path.mkdir(exist_ok=True)