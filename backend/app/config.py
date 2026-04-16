from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Pydantic reads each of these from environment variables
    # automatically. The variable name must match exactly (case-insensitive).
    # If a variable with no default is missing, the app crashes at startup
    # with a clear error — much better than a mysterious failure later.

    # Database
    database_url: str

    # Redis
    redis_url: str

    # App
    environment: str = "development"
    secret_key: str = "change_this_in_production"

    # Anthropic — optional for now, empty string is fine
    anthropic_api_key: str = ""

    class Config:
        # Tell pydantic where to find the .env file
        env_file = ".env"
        env_file_encoding = "utf-8"


# One shared instance — import this anywhere you need config:
# from app.config import settings
settings = Settings()
