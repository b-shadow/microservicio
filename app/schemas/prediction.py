from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class PredictionResponse(BaseModel):
    """Esquema de respuesta para clasificación de imagen."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "ok": True,
                "source": "upload",
                "evidencia_id": "ev-123",
                "clase_predicha": "PINCHAZO_LLANTA",
                "confianza": 0.87,
                "especialidad_sugerida": "GOMERIA_LLANTAS",
                "servicio_sugerido": "CAMBIO_LLANTA",
                "nivel_urgencia_sugerido": "MEDIA",
                "modelo_utilizado": "mock-vision",
                "version_modelo": "0.1.0",
                "tiempo_inferencia_ms": 35,
                "observaciones": "Imagen clara mostrando llanta dañada"
            }
        }
    )
    
    ok: bool
    source: str = Field(..., description="'upload' o 'url'")
    evidencia_id: Optional[str] = None
    clase_predicha: str = Field(..., description="Clase predicha: COLISION_VISIBLE, PINCHAZO_LLANTA, etc")
    confianza: float = Field(..., ge=0.0, le=1.0, description="Confianza del modelo (0.0-1.0)")
    especialidad_sugerida: Optional[str] = Field(None, description="Especialidad sugerida basada en la clase")
    servicio_sugerido: Optional[str] = Field(None, description="Servicio sugerido basado en la clase")
    nivel_urgencia_sugerido: Optional[str] = Field(None, description="BAJA, MEDIA, ALTA")
    modelo_utilizado: str = Field(..., description="mock-vision o image-classifier")
    version_modelo: str = Field(..., description="Versión del modelo")
    tiempo_inferencia_ms: int = Field(..., ge=0, description="Tiempo de inferencia en ms")
    observaciones: Optional[str] = Field(None, description="Observaciones adicionales")
