from fastapi import FastAPI
from .api.v1.health import router as health_router
from .api.v1.analyze import router as analyze_router

app = FastAPI(title="SmartSpend API", version="1.0")

app.include_router(health_router, prefix="/api/v1")
app.include_router(analyze_router, prefix="/api/v1")
