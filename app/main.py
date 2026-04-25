from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
from loguru import logger

from .core import settings, setup_logging, VisionServiceException
from .api.routes import health, predict

# Configurar logging
logger.remove()
setup_logger = setup_logging()


# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info(f"Iniciando {settings.app_name}")
    logger.info(f"Ambiente: {settings.app_env}")
    logger.info(f"Modo: {settings.model_provider}")
    logger.info(f"Escuchando en {settings.host}:{settings.port}")
    yield
    # Shutdown
    logger.info("Apagando servicio")


# Crear aplicación
app = FastAPI(
    title=settings.app_name,
    description="Microservicio ligero para análisis de imágenes de incidentes vehiculares",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)


# Handlers de excepciones personalizadas
@app.exception_handler(VisionServiceException)
async def vision_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"ok": False, "detail": exc.detail}
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={"ok": False, "detail": str(exc)}
    )


# Incluir routers
app.include_router(health.router)
app.include_router(predict.router)


# Root endpoint
@app.get("/", tags=["info"])
async def root():
    """Información básica del servicio."""
    return {
        "service": settings.app_name,
        "version": "1.0.0",
        "mode": settings.model_provider,
        "docs": "/docs",
        "redoc": "/redoc"
    }
