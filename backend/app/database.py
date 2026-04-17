import asyncpg
from app.config import settings

pool: asyncpg.Pool | None = None


async def create_pool():
    """
    Opens the connection pool. Called once when the API starts.
    min_size=2: always keep 2 connections open (warm and ready)
    max_size=10: never open more than 10 simultaneous connections
    """
    global pool
    # asyncpg only understands "postgresql://" — not "postgresql+asyncpg://"
    # The +asyncpg suffix is a SQLAlchemy convention. We strip it here.
    dsn = settings.database_url.replace("postgresql+asyncpg", "postgresql")
    pool = await asyncpg.create_pool(
        dsn=dsn,
        min_size=2,
        max_size=10,
    )


async def close_pool():
    """Closes all connections cleanly when the API shuts down."""
    global pool
    if pool:
        await pool.close()


def get_pool() -> asyncpg.Pool:
    """Returns the pool. Raises clearly if called before startup."""
    if pool is None:
        raise RuntimeError("Database pool not initialised. Was create_pool() called?")
    return pool
