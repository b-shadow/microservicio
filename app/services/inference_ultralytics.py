import json
import time
from pathlib import Path
from typing import Optional

from PIL import Image
from loguru import logger

from .inference_base import InferenceBase
from ..core import ModelNotFoundError, settings


class InferenceUltralytics(InferenceBase):
    """
    Motor de clasificacion real para modelos entrenados en este proyecto.

    Mantiene el nombre de clase para compatibilidad con configuraciones previas
    (`MODEL_PROVIDER=ultralytics`), pero internamente usa PyTorch/torchvision
    porque los modelos se guardan como `state_dict`.
    """

    IDX_TO_CLASS = {
        0: "COLISION_VISIBLE",
        1: "PINCHAZO_LLANTA",
        2: "HUMO_O_SOBRECALENTAMIENTO",
        3: "VEHICULO_INMOVILIZADO",
        4: "SIN_HALLAZGOS_CLAROS",
    }

    def __init__(self):
        super().__init__()
        self.model = None
        self.device = None
        self.arch = None
        self.transform = None
        self._version = "1.0.0"
        self._load_model()

    def _build_resnet50(self):
        import torch.nn as nn
        from torchvision import models

        model = models.resnet50(weights=None)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 5),
        )
        return model

    def _build_mobilenet(self):
        import torch.nn as nn
        from torchvision import models

        model = models.mobilenet_v2(weights=None)
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 5),
        )
        return model

    def _load_model(self):
        try:
            import torch
            from torchvision import transforms

            model_path = Path(settings.model_path)
            if not model_path.exists():
                raise ModelNotFoundError(
                    f"Modelo no encontrado en {model_path}. "
                    "Asegurate de entrenar el modelo primero."
                )

            self.arch = (
                "mobilenet_v2"
                if "mobile" in model_path.name.lower()
                else "resnet50"
            )
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            logger.info(
                f"Cargando modelo real desde {model_path} "
                f"(arquitectura={self.arch}, device={self.device})"
            )

            self.model = (
                self._build_mobilenet()
                if self.arch == "mobilenet_v2"
                else self._build_resnet50()
            )

            checkpoint = torch.load(
                model_path,
                map_location=self.device,
                weights_only=True,
            )
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

            self.model.load_state_dict(state_dict, strict=True)
            self.model = self.model.to(self.device)
            self.model.eval()

            self.transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )

            history_name = "history_mobile.json" if self.arch == "mobilenet_v2" else "history.json"
            history_path = model_path.parent / history_name
            if history_path.exists():
                with open(history_path, "r", encoding="utf-8") as f:
                    history = json.load(f)
                best_epoch = history.get("best_epoch")
                best_val_acc = history.get("best_val_acc")
                if best_epoch is not None and isinstance(best_val_acc, (float, int)):
                    self._version = f"{self.arch}-e{best_epoch}-va{best_val_acc:.4f}"

            logger.info("Modelo real cargado exitosamente")

        except ImportError as e:
            raise ModelNotFoundError(
                "Dependencias de inferencia no instaladas. "
                "Instala torch y torchvision para modo real."
            ) from e
        except Exception as e:
            logger.error(f"Error cargando modelo real: {str(e)}")
            raise ModelNotFoundError(f"Error cargando modelo: {str(e)}")

    async def predict(
        self,
        image: Image.Image,
        evidencia_id: Optional[str] = None,
    ) -> dict:
        if not self.model:
            raise ModelNotFoundError("Modelo no inicializado")

        start_time = time.time()

        try:
            import torch

            image_rgb = image.convert("RGB")
            tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                top1_prob, top1_idx = torch.max(probs, 1)

            class_idx = int(top1_idx.item())
            confianza = float(top1_prob.item())
            clase_predicha = self.IDX_TO_CLASS.get(class_idx, "SIN_HALLAZGOS_CLAROS")

            especialidad = self._map_class_to_specialty(clase_predicha)
            servicio = self._map_class_to_service(clase_predicha)
            urgencia = self._map_class_to_urgency(clase_predicha, confianza)

            elapsed = (time.time() - start_time) * 1000

            resultado = {
                "ok": True,
                "evidencia_id": evidencia_id,
                "clase_predicha": clase_predicha,
                "confianza": round(confianza, 4),
                "especialidad_sugerida": especialidad,
                "servicio_sugerido": servicio,
                "nivel_urgencia_sugerido": urgencia,
                "modelo_utilizado": self.model_name,
                "version_modelo": self.model_version,
                "tiempo_inferencia_ms": int(elapsed),
                "observaciones": f"Prediccion con {self.arch}",
            }

            logger.info(
                f"Clasificacion real completada: {clase_predicha} "
                f"(confianza: {confianza:.4f})"
            )
            return resultado

        except Exception as e:
            logger.error(f"Error en inferencia real: {str(e)}")
            raise

    @property
    def model_name(self) -> str:
        return f"torch-{self.arch}"

    @property
    def model_version(self) -> str:
        return self._version
