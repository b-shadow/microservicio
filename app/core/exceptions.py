from fastapi import HTTPException, status


class VisionServiceException(HTTPException):
    """Excepción base del servicio de visión."""
    
    def __init__(
        self,
        detail: str,
        status_code: int = status.HTTP_400_BAD_REQUEST
    ):
        super().__init__(status_code=status_code, detail=detail)


class InvalidImageException(VisionServiceException):
    """Imagen inválida o no soportada."""
    
    def __init__(self, detail: str = "Imagen inválida o no soportada"):
        super().__init__(detail, status.HTTP_400_BAD_REQUEST)


class ImageTooLargeException(VisionServiceException):
    """Imagen excede el tamaño máximo permitido."""
    
    def __init__(self, detail: str = "Imagen excede el tamaño máximo permitido"):
        super().__init__(detail, status.HTTP_413_REQUEST_ENTITY_TOO_LARGE)


class ImageDownloadException(VisionServiceException):
    """Error al descargar imagen desde URL."""
    
    def __init__(self, detail: str = "Error al descargar imagen desde URL"):
        super().__init__(detail, status.HTTP_400_BAD_REQUEST)


class ModelNotFoundError(VisionServiceException):
    """Modelo no encontrado o no disponible."""
    
    def __init__(self, detail: str = "Modelo no disponible"):
        super().__init__(detail, status.HTTP_500_INTERNAL_SERVER_ERROR)


class UnauthorizedException(VisionServiceException):
    """Acceso no autorizado."""
    
    def __init__(self, detail: str = "Token de servicio inválido"):
        super().__init__(detail, status.HTTP_401_UNAUTHORIZED)
