import uuid

from fastapi import APIRouter, HTTPException

from app.database import get_pool
from app.schemas.job import JobOut

router = APIRouter()


@router.get("/jobs/{job_id}", response_model=JobOut)
async def get_job(job_id: uuid.UUID) -> JobOut:
    pool = get_pool()

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, world_id, status, progress, error, started_at, finished_at
            FROM jobs
            WHERE id = $1
            """,
            job_id,
        )

    if row is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return JobOut(
        id=row["id"],
        world_id=row["world_id"],
        status=row["status"],
        progress=row["progress"],
        error=row["error"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
    )
