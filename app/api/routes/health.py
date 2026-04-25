from fastapi import APIRouter, Depends

from ...schemas import HealthResponse
from ...core import verify_service_token, settings
from ...services import get_inference_engine

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check del servicio"
)
async def health_check(
    _: None = Depends(verify_service_token)
) -> HealthResponse:
    """
    Verifica el estado del microservicio.
    
    Retorna:
    - status: "ok"
    - service: nombre del servicio
    - mode: modo actual (mock o ultralytics)
    """
    
    engine = get_inference_engine()
    
    return HealthResponse(
        status="ok",
        service="vision-service",
        mode=settings.model_provider
    )
