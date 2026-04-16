from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings

app = FastAPI(
    title="WorldLimit API",
    version="0.1.0",
    description="Natural language to 3D terrain",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "environment": settings.environment,
    }


@app.get("/")
async def root():
    return {"message": "WorldLimit API — visit /docs for the API reference"}
