import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from app.main import app


client = TestClient(app)


def create_test_image(width=100, height=100, format='PNG'):
    """Helper para crear imágenes de test."""
    img = Image.new('RGB', (width, height), color='blue')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format=format)
    img_bytes.seek(0)
    return img_bytes


def test_predict_image_mock_valid():
    """Test de predicción en modo mock con imagen válida."""
    img_bytes = create_test_image()
    
    response = client.post(
        "/predict/image",
        files={"image": ("test.png", img_bytes, "image/png")}
    )
    
    assert response.status_code == 200
    
    data = response.json()
    assert data["ok"] is True
    assert data["source"] == "upload"
    assert data["modelo_utilizado"] == "mock-vision"
    assert data["version_modelo"] == "0.2.0"
    assert isinstance(data["tiempo_inferencia_ms"], int)
    assert "clase_predicha" in data
    assert data["clase_predicha"] in [
        "COLISION_VISIBLE",
        "PINCHAZO_LLANTA",
        "HUMO_O_SOBRECALENTAMIENTO",
        "VEHICULO_INMOVILIZADO",
        "SIN_HALLAZGOS_CLAROS"
    ]
    assert 0.0 <= data["confianza"] <= 1.0


def test_predict_image_with_evidencia_id():
    """Test de predicción con evidencia_id."""
    img_bytes = create_test_image()
    
    response = client.post(
        "/predict/image",
        files={"image": ("test.png", img_bytes, "image/png")},
        data={"evidencia_id": "ev-123"}
    )
    
    assert response.status_code == 200
    
    data = response.json()
    assert data["evidencia_id"] == "ev-123"


def test_predict_image_missing_file():
    """Test de predicción sin imagen."""
    response = client.post("/predict/image")
    
    # Debe fallar sin archivo
    assert response.status_code == 422  # Validation error


def test_predict_image_invalid_file():
    """Test de predicción con archivo inválido."""
    invalid_data = b"Este no es una imagen"
    
    response = client.post(
        "/predict/image",
        files={"image": ("invalid.txt", invalid_data, "text/plain")}
    )
    
    assert response.status_code >= 400


def test_predict_image_large_image():
    """Test con imagen grande (pero dentro del límite)."""
    # Crear imagen grande pero válida (2MB)
    img = Image.new('RGB', (2000, 2000), color='green')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    response = client.post(
        "/predict/image",
        files={"image": ("large.png", img_bytes, "image/png")}
    )
    
    # Debe funcionar si está dentro del límite (10MB por defecto)
    assert response.status_code == 200


def test_predict_response_structure():
    """Test de que la respuesta tiene la estructura correcta."""
    img_bytes = create_test_image()
    
    response = client.post(
        "/predict/image",
        files={"image": ("test.png", img_bytes, "image/png")}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Validar que tiene todos los campos requeridos
    required_fields = [
        "ok", "source", "modelo_utilizado", "version_modelo",
        "tiempo_inferencia_ms", "clase_predicha", "confianza"
    ]
    
    for field in required_fields:
        assert field in data, f"Falta campo requerido: {field}"


def test_predict_classification_result():
    """Test que la clasificación es válida y con mapeos correctos."""
    img_bytes = create_test_image()
    
    response = client.post(
        "/predict/image",
        files={"image": ("test.png", img_bytes, "image/png")}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Clases válidas
    valid_classes = [
        "COLISION_VISIBLE",
        "PINCHAZO_LLANTA",
        "HUMO_O_SOBRECALENTAMIENTO",
        "VEHICULO_INMOVILIZADO",
        "SIN_HALLAZGOS_CLAROS"
    ]
    assert data["clase_predicha"] in valid_classes
    
    # Verificar mapeos
    clase = data["clase_predicha"]
    
    # Si hay especialidad, debe ser válida
    if data["especialidad_sugerida"]:
        valid_specialties = [
            "CHAPERIA_CARROCERIA",
            "GOMERIA_LLANTAS",
            "MECANICA_GENERAL",
            "AUXILIO_VIAL_RESCATE"
        ]
        assert data["especialidad_sugerida"] in valid_specialties
    
    # Si hay urgencia, debe ser válida
    if data["nivel_urgencia_sugerido"]:
        assert data["nivel_urgencia_sugerido"] in ["BAJA", "MEDIA", "ALTA"]


def test_predict_confidence_range():
    """Test que confianza está en rango válido."""
    img_bytes = create_test_image()
    
    response = client.post(
        "/predict/image",
        files={"image": ("test.png", img_bytes, "image/png")}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert isinstance(data["confianza"], (int, float))
    assert 0.0 <= data["confianza"] <= 1.0

