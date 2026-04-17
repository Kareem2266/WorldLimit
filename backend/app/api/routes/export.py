import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.database import get_pool

router = APIRouter()

EXPORT_DIR = Path("/tmp/worldlimit/exports")


@router.get("/export/{job_id}")
async def export_world(job_id: uuid.UUID):
    pool = get_pool()

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT status, world_id FROM jobs WHERE id = $1",
            job_id,
        )

    if row is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if row["status"] != "done":
        raise HTTPException(
            status_code=409,
            detail=f"Job not ready for export (status: {row['status']})",
        )

    export_path = EXPORT_DIR / f"{row['world_id']}.zip"

    if not export_path.exists():
        raise HTTPException(status_code=404, detail="Export file not found")

    return FileResponse(
        path=str(export_path),
        media_type="application/zip",
        filename=f"world_{row['world_id']}.zip",
    )
