from fastapi import Depends, Header
from typing import Optional
from .config import settings
from .exceptions import UnauthorizedException


async def verify_service_token(
    x_service_token: Optional[str] = Header(None)
) -> None:
    """
    Verifica el token de servicio si está configurado.
    Si SERVICE_TOKEN no está configurado en .env, permite acceso sin token.
    """
    if settings.service_token:
        if not x_service_token or x_service_token != settings.service_token:
            raise UnauthorizedException("Token de servicio inválido o faltante")
