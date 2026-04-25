from fastapi import APIRouter, UploadFile, File, Form, Depends
from typing import Optional
from loguru import logger

from ...schemas import PredictionResponse
from ...core import verify_service_token, InvalidImageException
from ...utils import validate_image_bytes
from ...services import get_inference_engine, fetch_image_from_url

router = APIRouter(tags=["prediction"])


@router.post(
    "/predict/image",
    response_model=PredictionResponse,
    summary="Predicción desde upload de imagen"
)
async def predict_image(
    image: UploadFile = File(...),
    evidencia_id: Optional[str] = Form(None),
    mime_type: Optional[str] = Form(None),
    _: None = Depends(verify_service_token)
) -> PredictionResponse:
    """
    Realiza predicción en una imagen cargada.
    
    Args:
        image: Archivo de imagen (multipart/form-data)
        evidencia_id: ID opcional de evidencia del backend
        mime_type: MIME type opcional
    
    Retorna:
        PredictionResponse con resultados de análisis
    """
    
    try:
        # Validar y cargar imagen
        logger.info(f"Procesando imagen: {image.filename}")
        
        image_bytes = await image.read()
        img = validate_image_bytes(image_bytes)
        
        # Ejecutar inferencia
        engine = get_inference_engine()
        result = await engine.predict(
            image=img,
            evidencia_id=evidencia_id
        )
        
        # Agregar fuente
        result["source"] = "upload"
        
        return PredictionResponse(**result)
        
    except Exception as e:
        logger.error(f"Error en predicción image: {str(e)}")
        raise


@router.post(
    "/predict/image-from-url",
    response_model=PredictionResponse,
    summary="Predicción desde URL de imagen"
)
async def predict_image_from_url(
    body: dict,
    _: None = Depends(verify_service_token)
) -> PredictionResponse:
    """
    Realiza predicción en una imagen descargada desde URL.
    
    Body JSON:
    {
        "image_url": "https://...",
        "evidencia_id": "string (opcional)"
    }
    
    Retorna:
        PredictionResponse con resultados de análisis
    """
    
    try:
        image_url = body.get("image_url")
        evidencia_id = body.get("evidencia_id")
        
        if not image_url:
            raise InvalidImageException("Falta 'image_url' en el body")
        
        logger.info(f"Descargando y procesando imagen desde URL: {image_url}")
        
        # Descargar imagen
        image_bytes = await fetch_image_from_url(image_url)
        
        # Validar imagen
        img = validate_image_bytes(image_bytes)
        
        # Ejecutar inferencia
        engine = get_inference_engine()
        result = await engine.predict(
            image=img,
            evidencia_id=evidencia_id
        )
        
        # Agregar fuente
        result["source"] = "url"
        
        return PredictionResponse(**result)
        
    except Exception as e:
        logger.error(f"Error en predicción image-from-url: {str(e)}")
        raise
