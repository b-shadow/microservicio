import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    """Fixture para el cliente de test."""
    return TestClient(app)


@pytest.fixture
def sample_image_bytes():
    """Fixture que proporciona una imagen válida en bytes."""
    from PIL import Image
    import io
    
    # Crear imagen GIF mínima válida
    img = Image.new('RGB', (100, 100), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    return img_bytes.getvalue()


@pytest.fixture
def invalid_image_bytes():
    """Fixture que proporciona datos invalidos como imagen."""
    return b"Este no es una imagen valida"
