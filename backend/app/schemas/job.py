from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel


class JobOut(BaseModel):
    id: uuid.UUID
    world_id: uuid.UUID
    status: str
    progress: int
    error: str | None
    started_at: datetime | None
    finished_at: datetime | None
