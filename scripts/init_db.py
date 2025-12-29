from app.core.config import settings
from app.core.logging_db import DbLogger

if __name__ == "__main__":
    DbLogger(settings.database_url)
    print(f"DB initialized at: {settings.database_url}")
