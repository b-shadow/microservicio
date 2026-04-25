from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict
from typing import Optional


class Settings(BaseSettings):
    """Configuración global del microservicio."""
    
    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=("settings_",),
    )
    
    # App
    app_name: str = Field("vision-service", alias="APP_NAME")
    app_env: str = Field("development", alias="APP_ENV")
    host: str = Field("0.0.0.0", alias="HOST")
    port: int = Field(8001, alias="PORT")
    
    # Logging
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    
    # Model
    model_provider: str = Field("mock", alias="MODEL_PROVIDER")
    model_path: str = Field("models/best.pt", alias="MODEL_PATH")
    
    # Security
    service_token: Optional[str] = Field(None, alias="SERVICE_TOKEN")
    
    # Timeouts y límites
    request_timeout_seconds: int = Field(20, alias="REQUEST_TIMEOUT_SECONDS")
    max_image_size_mb: int = Field(10, alias="MAX_IMAGE_SIZE_MB")
    
    @property
    def max_image_size_bytes(self) -> int:
        """Retorna el tamaño máximo de imagen en bytes."""
        return self.max_image_size_mb * 1024 * 1024


# Instancia global
settings = Settings()
