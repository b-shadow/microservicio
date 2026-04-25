# Vision Service Microservice

Microservicio ligero para **clasificación de imágenes de incidentes vehiculares**. Predice la clase de incidente visible en una foto (colisión, pinchazo, sobrecalentamiento, vehículo inmovilizado, sin hallazgos).

## 🎯 Características

- **Clasificación de imágenes**: Predice qué tipo de incidente muestra la imagen
- **5 Clases de incidentes**:
  - **COLISION_VISIBLE** → Sugiere Chapería, Grúa, Urgencia ALTA
  - **PINCHAZO_LLANTA** → Sugiere Goma, Cambio llanta, Urgencia MEDIA
  - **HUMO_O_SOBRECALENTAMIENTO** → Sugiere Mecánica, Diagnóstico, Urgencia ALTA
  - **VEHICULO_INMOVILIZADO** → Sugiere Auxilio vial, Remolque, Urgencia MEDIA
  - **SIN_HALLAZGOS_CLAROS** → Sugiere Sin servicio, Urgencia BAJA
- **Dos modos de operación**:
  - **Mock** (por defecto): Operativo inmediatamente sin modelos entrenados
  - **Real**: Listo para integrar modelos de clasificación (ResNet, EfficientNet, etc.)
- **Stateless**: Sin almacenamiento persistente ni base de datos
- **API REST**: FastAPI con documentación automática Swagger/ReDoc
- **Seguridad opcional**: Token de servicio configurable
- **Descargas de imágenes**: Soporte para procesamiento desde URL

## 🚀 Inicio Rápido

### 1. Clonar y preparar ambiente

```bash
# Navegar a la carpeta del proyecto
cd microservicio

# Crear entorno virtual
python -m venv venv_vision

# Activar entorno
# En Windows:
venv_vision\Scripts\activate
# En Linux/Mac:
source venv_vision/bin/activate
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Configurar variables de entorno

```bash
# Crear archivo .env desde el ejemplo
cp .env.example .env

# Verificar/editar .env según necesites
# Por defecto, corre en modo mock en puerto 8001
```

### 4. Ejecutar el servicio

```bash
# Modo desarrollo
uvicorn app.main:app --reload --host 0.0.0.0 --port 8001

# Modo producción
uvicorn app.main:app --host 0.0.0.0 --port 8001 --workers 4
```

## 📝 Endpoints

### GET /health
Verifica el estado del servicio.

```bash
curl http://localhost:8001/health
```

**Respuesta:**
```json
{
  "status": "ok",
  "service": "vision-service",
  "mode": "mock"
}
```

### POST /predict/image
Clasifica una imagen cargada.

**Request:**
```bash
curl -X POST "http://localhost:8001/predict/image" \
  -F "image=@/path/to/image.jpg" \
  -F "evidencia_id=ev-123"
```

**Response:**
```json
{
  "ok": true,
  "source": "upload",
  "evidencia_id": "ev-123",
  "clase_predicha": "PINCHAZO_LLANTA",
  "confianza": 0.87,
  "especialidad_sugerida": "GOMERIA_LLANTAS",
  "servicio_sugerido": "CAMBIO_LLANTA",
  "nivel_urgencia_sugerido": "MEDIA",
  "modelo_utilizado": "mock-vision",
  "version_modelo": "0.2.0",
  "tiempo_inferencia_ms": 35,
  "observaciones": "Imagen clara mostrando llanta dañada"
}
```

### POST /predict/image-from-url
Descarga y analiza una imagen desde URL.

**Request:**
```bash
curl -X POST "http://localhost:8001/predict/image-from-url" \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/image.jpg",
    "evidencia_id": "ev-456"
  }'
```

**Response:** Idéntica a `/predict/image` pero con `"source": "url"`

## 🔧 Configuración

Variables en `.env`:

```env
# Aplicación
APP_NAME=vision-service
APP_ENV=development
HOST=0.0.0.0
PORT=8001

# Logging
LOG_LEVEL=INFO

# Modelo
MODEL_PROVIDER=torch             # mock | torch
MODEL_PATH=models/best.pt

# Seguridad (opcional)
SERVICE_TOKEN=                   # Dejar vacío para desactivar

# Límites
REQUEST_TIMEOUT_SECONDS=20
MAX_IMAGE_SIZE_MB=10
```

## 📋 Modos de Operación

### Modo Mock (Predeterminado)

Operacional inmediatamente sin dependencias externas.

```bash
# Ya está configurado por defecto
MODEL_PROVIDER=mock
```

**Características:**
- Acepta imágenes válidas
- Devuelve clasificaciones simuladas pero realistas
- Pesos de probabilidad para cada clase:
  - SIN_HALLAZGOS_CLAROS: 35%
  - PINCHAZO_LLANTA: 25%
  - COLISION_VISIBLE: 15%
  - VEHICULO_INMOVILIZADO: 15%
  - HUMO_O_SOBRECALENTAMIENTO: 10%
- Sin requisito de modelos entrenados
- Ideal para integración inicial
- Version: 0.2.0

### Modo Real (Modelo Personalizado)

Preparado para modelos de clasificación reales.

```bash
# Editar .env
MODEL_PROVIDER=torch
MODEL_PATH=models/best.pt
```

**Pasos para usar modelo real:**

1. **Entrenar o descargar modelo** (ejemplo con PyTorch/FastAI/TensorFlow):
```bash
# Ejemplo: modelo ResNet entrenado
# python train_classifier.py --output models/best.pt
```

2. **Preparar el modelo** para que retorne clases en el mismo orden:
   - 0: COLISION_VISIBLE
   - 1: PINCHAZO_LLANTA
   - 2: HUMO_O_SOBRECALENTAMIENTO
   - 3: VEHICULO_INMOVILIZADO
   - 4: SIN_HALLAZGOS_CLAROS

3. **Copiar modelo**:
```bash
cp models/best.pt models/best.pt
```

4. **Configurar**:
```bash
MODEL_PROVIDER=torch
MODEL_PATH=models/best.pt
```

5. **Reiniciar servicio**:
```bash
uvicorn app.main:app --reload
```

**El servicio cargará automáticamente el modelo real.**

## 🔐 Seguridad

### Token de Servicio (Opcional)

Si necesitas proteger el acceso:

```bash
# En .env
SERVICE_TOKEN=tu_token_secreto_aqui
```

Luego incluye el header en requests:

```bash
curl -H "X-Service-Token: tu_token_secreto_aqui" \
  http://localhost:8001/health
```

Si `SERVICE_TOKEN` está vacío en `.env`, se permite acceso sin token.

## 🧪 Tests

Ejecutar suite de tests:

```bash
# Todos los tests
pytest

# Con cobertura
pytest --cov=app

# Solo tests específicos
pytest tests/test_health.py -v
pytest tests/test_predict_mock.py -v
```

## 📊 Documentación Interactiva

Una vez que el servicio esté corriendo:

- **Swagger UI**: http://localhost:8001/docs
- **ReDoc**: http://localhost:8001/redoc

## 🐳 Docker

### Construir imagen

```bash
docker build -t vision-service:latest .
```

### Ejecutar en Docker

```bash
docker run -p 8001:8001 \
  -e MODEL_PROVIDER=mock \
  -e LOG_LEVEL=INFO \
  vision-service:latest
```

### Docker Compose (opcional)

Crea `docker-compose.yml`:

```yaml
version: '3.8'

services:
  vision-service:
    build: .
    ports:
      - "8001:8001"
    environment:
      MODEL_PROVIDER: mock
      LOG_LEVEL: INFO
      HOST: 0.0.0.0
      PORT: 8001
```

Ejecuta:
```bash
docker-compose up
```

## ☁️ Despliegue en Render

### 1. Subir a GitHub

```bash
git init
git add .
git commit -m "Initial commit: vision-service"
git push origin main
```

### 2. Crear servicio en Render

1. Ve a https://dashboard.render.com
2. New → Web Service
3. Conecta tu repositorio de GitHub
4. Configura:
   - **Name**: vision-service
   - **Environment**: Python 3.12
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Environment Variables:
   - `MODEL_PROVIDER=mock`
   - `LOG_LEVEL=INFO`
   - `PORT=10000`

## 🔗 Integración con Backend Principal

El backend principal puede consumir este servicio:

```python
import httpx

async def classify_incident_image(image_path: str, incident_id: str):
    """Clasifica imagen de incidente usando el servicio de visión."""
    
    async with httpx.AsyncClient() as client:
        with open(image_path, 'rb') as f:
            files = {
                'image': ('image.jpg', f, 'image/jpeg'),
            }
            data = {
                'evidencia_id': incident_id
            }
            
            response = await client.post(
                "http://localhost:8001/predict/image",
                files=files,
                data=data
            )
    
    if response.status_code == 200:
        result = response.json()
        # Usar clase_predicha, especialidad_sugerida, etc.
        clase = result['clase_predicha']
        especialidad = result['especialidad_sugerida']
        urgencia = result['nivel_urgencia_sugerido']
        return {
            'clase': clase,
            'especialidad': especialidad,
            'urgencia': urgencia,
            'confianza': result['confianza']
        }
    else:
        raise Exception(f"Vision service error: {response.text}")
```

### Variables de entorno en backend

Agrega al `.env` del backend:

```env
VISION_SERVICE_URL=http://localhost:8001
# O en producción:
VISION_SERVICE_URL=https://vision-service.example.com
```

## 📚 Estructura del Proyecto

```
microservicio/
├── app/
│   ├── main.py              # Aplicación FastAPI
│   ├── api/
│   │   └── routes/
│   │       ├── health.py    # Endpoint /health
│   │       └── predict.py   # Endpoints de clasificación
│   ├── core/
│   │   ├── config.py        # Configuración (Pydantic Settings)
│   │   ├── exceptions.py    # Excepciones personalizadas
│   │   ├── logging.py       # Setup de loguru
│   │   └── security.py      # Validación de tokens
│   ├── schemas/
│   │   ├── health.py        # Schema de /health
│   │   └── prediction.py    # Schema de respuestas de clasificación
│   ├── services/
│   │   ├── image_fetcher.py        # Descarga de URLs
│   │   ├── inference_base.py       # Clase base de clasificadores
│   │   ├── inference_mock.py       # Clasificador mock
│   │   ├── inference_ultralytics.py # Clasificador real (pluggable)
│   │   └── inference_factory.py    # Factory pattern
│   └── utils/
│       └── image_utils.py  # Utilidades de imagen
├── models/
│   └── .gitkeep           # Carpeta para modelos (vacía inicialmente)
├── tests/
│   ├── conftest.py        # Configuración pytest
│   ├── test_health.py     # Tests de health
│   ├── test_predict_mock.py # Tests de clasificación
│   └── test_api.py        # Tests de integración
├── requirements.txt       # Dependencias Python
├── .env.example          # Variables de ejemplo
├── .gitignore           # Archivos a ignorar
├── Dockerfile           # Para despliegue en Docker
└── README.md            # Este archivo
```

## 🛠️ Troubleshooting

### El servicio no arranca

```bash
# Verificar que tengas Python 3.12+
python --version

# Verificar que todas las dependencias están instaladas
pip install -r requirements.txt

# Verificar la configuración de .env
cat .env
```

### Imagen rechazada como "no válida"

- Verifica que sea un formato soportado (PNG, JPG, BMP, GIF, WEBP)
- Verifica el tamaño (máximo 10MB por defecto)
- Asegúrate de no enviar un archivo corrupto

### "Modelo no encontrado" con modo real

```bash
# Asegúrate de que el archivo existe
ls -la models/best.pt

# Si no existe, coloca un modelo entrenado allí
```

### Timeout en descargas de URL

Aumenta timeout en `.env`:
```bash
REQUEST_TIMEOUT_SECONDS=30
```

## 📖 Recursos

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Ultralytics YOLO](https://docs.ultralytics.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Loguru Documentation](https://loguru.readthedocs.io/)

## 📄 Licencia

Este proyecto es parte del sistema de emergencias vehiculares.

## 👨‍💼 Autor

Creado como microservicio desacoplado para clasificación de imágenes de incidentes.

---

**Última actualización**: Abril 2026
