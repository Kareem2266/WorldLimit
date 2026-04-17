from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import create_pool, close_pool
from app.api.router import api_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    await create_pool()
    yield
    await close_pool()


app = FastAPI(
    title="WorldLimit API",
    version="0.1.0",
    description="Natural language to 3D terrain",
    lifespan=lifespan,
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

app.include_router(api_router, prefix="/api")


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "environment": settings.environment,
    }


@app.get("/")
async def root():
    return {"message": "WorldLimit API — visit /docs for the API reference"}
