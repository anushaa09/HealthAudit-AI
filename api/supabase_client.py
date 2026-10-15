
import os
from pathlib import Path

from dotenv import load_dotenv
from supabase import Client, create_client

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SECRET_KEY = os.getenv("SUPABASE_SECRET_KEY")

if not SUPABASE_URL or not SUPABASE_SECRET_KEY:
    raise RuntimeError(
        "Missing SUPABASE_URL or SUPABASE_SECRET_KEY "
        "in the project-root .env file"
    )

supabase: Client = create_client(
    SUPABASE_URL,
    SUPABASE_SECRET_KEY,
)

