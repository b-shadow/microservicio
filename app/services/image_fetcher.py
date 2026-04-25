import httpx
from loguru import logger

from ..core import ImageDownloadException, ImageTooLargeException, settings
from ..utils import ALLOWED_MIME_TYPES


async def fetch_image_from_url(url: str) -> bytes:
    """
    Descarga una imagen desde una URL de forma segura.
    
    Args:
        url: URL de la imagen
    
    Returns:
        Bytes de la imagen
    
    Raises:
        ImageDownloadException: Si la descarga falla
        ImageTooLargeException: Si el archivo excede el límite
    """
    
    try:
        async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client:
            logger.info(f"Descargando imagen desde: {url}")
            
            response = await client.get(url, follow_redirects=True)
            response.raise_for_status()
            
            # Validar content-type
            content_type = response.headers.get('content-type', '').lower()
            if not any(mime in content_type for mime in ALLOWED_MIME_TYPES):
                raise ImageDownloadException(
                    f"Content-Type no es imagen válida: {content_type}"
                )
            
            # Validar tamaño
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > settings.max_image_size_bytes:
                raise ImageTooLargeException(
                    f"Imagen en URL excede {settings.max_image_size_mb}MB"
                )
            
            image_bytes = response.content
            
            # Validar tamaño de lo descargado
            if len(image_bytes) > settings.max_image_size_bytes:
                raise ImageTooLargeException(
                    f"Imagen descargada excede {settings.max_image_size_mb}MB"
                )
            
            logger.info(f"Imagen descargada exitosamente: {len(image_bytes)} bytes")
            return image_bytes
            
    except ImageDownloadException:
        raise
    except ImageTooLargeException:
        raise
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP Error descargando imagen: {e.response.status_code}")
        raise ImageDownloadException(
            f"Error HTTP {e.response.status_code} al descargar imagen"
        )
    except httpx.TimeoutException:
        logger.error("Timeout descargando imagen")
        raise ImageDownloadException("Timeout al descargar imagen")
    except Exception as e:
        logger.error(f"Error descargando imagen: {str(e)}")
        raise ImageDownloadException(f"Error descargando imagen: {str(e)}")
