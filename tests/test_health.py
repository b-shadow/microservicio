import pytest
from app.main import app
from fastapi.testclient import TestClient


client = TestClient(app)


def test_health_check():
    """Test del endpoint /health."""
    response = client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "vision-service"
    assert data["mode"] in ["mock", "ultralytics"]


def test_health_with_invalid_token():
    """Test del endpoint /health con token inválido (si está configurado)."""
    # Este test manejaría si SERVICE_TOKEN está configurado
    # Por ahora, solo verificamos que funciona sin token
    response = client.get("/health")
    assert response.status_code == 200


def test_root_endpoint():
    """Test del endpoint raíz."""
    response = client.get("/")
    assert response.status_code == 200
    
    data = response.json()
    assert "service" in data
    assert "version" in data
    assert "mode" in data
    assert data["service"] == "vision-service"
