import sys
from loguru import logger

from .config import settings


def setup_logging():
    """Configura loguru para el servicio."""
    
    # Eliminar handlers predeterminados
    logger.remove()
    
    # Configurar formato y nivel
    format_str = (
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    
    logger.add(
        sys.stdout,
        format=format_str,
        level=settings.log_level,
        colorize=True
    )
    
    # Opcional: agregar archivo de log
    logger.add(
        "logs/vision_service.log",
        format=format_str,
        level=settings.log_level,
        rotation="500 MB",
        retention="7 days"
    )
    
    return logger
