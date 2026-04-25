import io
from PIL import Image
import numpy as np
from loguru import logger

from ..core import InvalidImageException, ImageTooLargeException, settings


ALLOWED_FORMATS = {'JPEG', 'JPG', 'PNG', 'BMP', 'GIF', 'WEBP'}
ALLOWED_MIME_TYPES = {
    'image/jpeg', 'image/jpg', 'image/png', 'image/bmp', 'image/gif', 'image/webp'
}


def validate_image_bytes(
    image_bytes: bytes,
    max_size: int = settings.max_image_size_bytes
) -> Image.Image:
    """
    Valida y carga una imagen desde bytes.
    
    Args:
        image_bytes: Bytes de la imagen
        max_size: Tamaño máximo permitido en bytes
    
    Returns:
        Imagen PIL
    
    Raises:
        InvalidImageException: Si la imagen es inválida
        ImageTooLargeException: Si la imagen excede el tamaño máximo
    """
    
    # Validar que no esté vacía
    if not image_bytes:
        raise InvalidImageException("Archivo de imagen vacío")
    
    # Validar tamaño
    if len(image_bytes) > max_size:
        raise ImageTooLargeException(
            f"Imagen excede {settings.max_image_size_mb}MB"
        )
    
    try:
        # Intentar abrir la imagen
        img = Image.open(io.BytesIO(image_bytes))
        
        # Validar formato
        if img.format and img.format.upper() not in ALLOWED_FORMATS:
            raise InvalidImageException(
                f"Formato de imagen no soportado: {img.format}"
            )
        
        # Convertir a RGB si es necesario
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        logger.info(f"Imagen validada: {img.format} {img.size}")
        return img
        
    except InvalidImageException:
        raise
    except Exception as e:
        logger.error(f"Error validando imagen: {str(e)}")
        raise InvalidImageException(
            f"Error al procesar imagen: {str(e)}"
        )


def image_to_numpy(img: Image.Image) -> np.ndarray:
    """Convierte una imagen PIL a numpy array."""
    return np.array(img)


def get_image_dimensions(img: Image.Image) -> tuple:
    """Retorna (width, height) de la imagen."""
    return img.size


def get_image_format(img: Image.Image) -> str:
    """Retorna el formato de la imagen."""
    return img.format or "UNKNOWN"
