from loguru import logger

from .inference_base import InferenceBase
from .inference_mock import InferenceMock
from .inference_ultralytics import InferenceUltralytics
from ..core import settings, ModelNotFoundError


def create_inference_engine() -> InferenceBase:
    """
    Factory para crear el motor de inferencia basado en configuración.
    
    Returns:
        Instancia de InferenceBase (Mock o Ultralytics)
    
    Raises:
        ModelNotFoundError: Si el proveedor es inválido o el modelo no se puede cargar
    """
    
    provider = settings.model_provider.lower()
    logger.info(f"Creando motor de inferencia: {provider}")
    
    if provider == "mock":
        return InferenceMock()
    
    elif provider in ("ultralytics", "torch", "pytorch"):
        return InferenceUltralytics()
    
    else:
        raise ModelNotFoundError(
            f"Proveedor de modelo desconocido: {provider}. "
            "Usa 'mock' o 'torch'"
        )


# Instancia global lazily initialized
_engine: InferenceBase = None


def get_inference_engine() -> InferenceBase:
    """Obtiene o crea la instancia global del motor de inferencia."""
    global _engine
    if _engine is None:
        _engine = create_inference_engine()
    return _engine
