from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from backend.routes.predict import router as predict_router
from backend.routes.history import router as history_router
from backend.database import init_db

def create_app() -> FastAPI:
    app = FastAPI(
        title="DR Detection API",
        description="Diabetic Retinopathy Detection using EfficientNetV2-S",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(predict_router, prefix="/api")
    app.include_router(history_router, prefix="/api")

    # Ensure static directories exist
    Path("inference/heatmaps").mkdir(parents=True, exist_ok=True)
    Path("backend/reports").mkdir(parents=True, exist_ok=True)
    Path("backend/uploads").mkdir(parents=True, exist_ok=True)

    app.mount("/heatmaps", StaticFiles(directory="inference/heatmaps"), name="heatmaps")
    app.mount("/reports", StaticFiles(directory="backend/reports"), name="reports")
    app.mount("/uploads", StaticFiles(directory="backend/uploads"), name="uploads")

    @app.on_event("startup")
    async def startup_event() -> None:
        await init_db()

    @app.get("/health")
    async def health_check() -> dict:
        return {"status": "ok"}

    return app

app = create_app()
