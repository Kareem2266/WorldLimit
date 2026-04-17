import os
from logging.config import fileConfig

from sqlalchemy import create_engine
from alembic import context

# Load logging config from alembic.ini
config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Read the DATABASE_URL from environment.
raw_url = os.environ.get("DATABASE_URL", "")
sync_url = raw_url.replace("postgresql+asyncpg", "postgresql+psycopg2")


def run_migrations():
    # create_engine opens a synchronous connection using psycopg2
    engine = create_engine(sync_url)
    with engine.connect() as connection:
        context.configure(connection=connection)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


run_migrations()
