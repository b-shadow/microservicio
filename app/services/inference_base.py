from abc import ABC, abstractmethod
from typing import Optional
from PIL import Image
from loguru import logger


class InferenceBase(ABC):
    """Clase base para motores de clasificación de imágenes."""
    
    # Clases disponibles para clasificación
    AVAILABLE_CLASSES = [
        "COLISION_VISIBLE",
        "PINCHAZO_LLANTA",
        "HUMO_O_SOBRECALENTAMIENTO",
        "VEHICULO_INMOVILIZADO",
        "SIN_HALLAZGOS_CLAROS"
    ]
    
    def __init__(self):
        logger.info(f"Inicializando {self.__class__.__name__}")
    
    @abstractmethod
    async def predict(
        self,
        image: Image.Image,
        evidencia_id: Optional[str] = None
    ) -> dict:
        """
        Clasifica una imagen en una de las clases disponibles.
        
        Args:
            image: Imagen PIL
            evidencia_id: ID opcional de evidencia
        
        Returns:
            Diccionario con clase predicha y confianza
        """
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Nombre del modelo utilizado."""
        pass
    
    @property
    @abstractmethod
    def model_version(self) -> str:
        """Versión del modelo."""
        pass
    
    @staticmethod
    def _map_class_to_specialty(class_name: str) -> Optional[str]:
        """Mapea clase predicha a especialidad sugerida."""
        mapping = {
            "COLISION_VISIBLE": "CHAPERIA_CARROCERIA",
            "PINCHAZO_LLANTA": "GOMERIA_LLANTAS",
            "HUMO_O_SOBRECALENTAMIENTO": "MECANICA_GENERAL",
            "VEHICULO_INMOVILIZADO": "AUXILIO_VIAL_RESCATE",
            "SIN_HALLAZGOS_CLAROS": None,
        }
        return mapping.get(class_name)
    
    @staticmethod
    def _map_class_to_service(class_name: str) -> Optional[str]:
        """Mapea clase predicha a servicio sugerido."""
        mapping = {
            "COLISION_VISIBLE": "GRUA",
            "PINCHAZO_LLANTA": "CAMBIO_LLANTA",
            "HUMO_O_SOBRECALENTAMIENTO": "DIAGNOSTICO_MECANICO_EN_SITIO",
            "VEHICULO_INMOVILIZADO": "REMOLQUE_VEHICULO",
            "SIN_HALLAZGOS_CLAROS": None,
        }
        return mapping.get(class_name)
    
    @staticmethod
    def _map_class_to_urgency(class_name: str, confidence: float) -> str:
        """Mapea clase predicha a nivel de urgencia."""
        urgency_map = {
            "COLISION_VISIBLE": "ALTA",
            "PINCHAZO_LLANTA": "MEDIA",
            "HUMO_O_SOBRECALENTAMIENTO": "ALTA",
            "VEHICULO_INMOVILIZADO": "MEDIA",
            "SIN_HALLAZGOS_CLAROS": "BAJA",
        }
        return urgency_map.get(class_name, "MEDIA")
