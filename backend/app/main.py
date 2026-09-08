from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import CORS_ORIGINS
from app.routers import admin, fighters, models, predict

app = FastAPI(title="UFC Fight Predictor API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(fighters.router)
app.include_router(predict.router)
app.include_router(models.router)
app.include_router(admin.router)


@app.get("/api/health")
def health():
    return {"status": "ok"}
