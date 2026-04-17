"""create_worlds_and_jobs_tables

Revision ID: 375bd1e478ae
Revises: 
Create Date: 2026-04-16 23:40:17.130305

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '375bd1e478ae'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute('CREATE EXTENSION IF NOT EXISTS "pgcrypto"')

    op.execute("""
        CREATE TABLE worlds (
            id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            prompt      TEXT NOT NULL,
            status      TEXT NOT NULL DEFAULT 'queued',
            params      JSONB,
            created_at  TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
        )
    """)

    op.execute("""
        CREATE TABLE jobs (
            id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            world_id     UUID NOT NULL REFERENCES worlds(id) ON DELETE CASCADE,
            status       TEXT NOT NULL DEFAULT 'queued',
            progress     INT NOT NULL DEFAULT 0,
            error        TEXT,
            started_at   TIMESTAMP WITH TIME ZONE,
            finished_at  TIMESTAMP WITH TIME ZONE
        )
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS jobs")
    op.execute("DROP TABLE IF EXISTS worlds")
