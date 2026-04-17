import uuid

from fastapi import APIRouter

from app.database import get_pool
from app.schemas.world import GenerateRequest, GenerateResponse

router = APIRouter()


@router.post("/generate", response_model=GenerateResponse, status_code=202)
async def generate(body: GenerateRequest) -> GenerateResponse:
    pool = get_pool()

    async with pool.acquire() as conn:
        async with conn.transaction():
            world = await conn.fetchrow(
                """
                INSERT INTO worlds (prompt, status, params)
                VALUES ($1, 'queued', $2)
                RETURNING id, status
                """,
                body.prompt,
                f'{{"seed": {body.seed}}}' if body.seed is not None else None,
            )

            job = await conn.fetchrow(
                """
                INSERT INTO jobs (world_id, status, progress)
                VALUES ($1, 'queued', 0)
                RETURNING id, status
                """,
                world["id"],
            )

    return GenerateResponse(
        job_id=job["id"],
        world_id=world["id"],
        status=job["status"],
    )
