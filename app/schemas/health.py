from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Esquema de respuesta para el endpoint de salud."""
    status: str
    service: str
    mode: str
