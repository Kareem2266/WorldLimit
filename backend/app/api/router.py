from fastapi import APIRouter

from app.api.routes import generate, jobs, export

api_router = APIRouter()

api_router.include_router(generate.router)
api_router.include_router(jobs.router)
api_router.include_router(export.router)
