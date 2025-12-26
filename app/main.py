"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import get_settings
from app.models.qwen_vl import get_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown."""
    settings = get_settings()

    # Configure logging level
    logging.getLogger().setLevel(settings.log_level)

    logger.info("Starting Traffic Equipment Anomaly Detection Service...")
    logger.info(f"Model: {settings.model_name}")
    logger.info(f"Device: {settings.device}")

    # Pre-load model on startup (optional, can be loaded on first request)
    model = get_model()
    try:
        logger.info("Loading model...")
        model.load()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to pre-load model: {e}. Will load on first request.")

    yield

    # Cleanup on shutdown
    logger.info("Shutting down...")
    if model.is_loaded:
        model.unload()
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="交通设备异常检测服务",
        description="基于 Qwen3-VL 的交通设备异常检测 API 服务",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routes
    app.include_router(router)

    @app.get("/")
    async def root():
        """Root endpoint with service information."""
        return {
            "service": "Traffic Equipment Anomaly Detection",
            "version": "0.1.0",
            "model": settings.model_name,
            "docs": "/docs",
        }

    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        reload=settings.api_reload,
    )
