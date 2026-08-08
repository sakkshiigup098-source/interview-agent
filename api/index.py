"""
Vercel entrypoint. Vercel's Python runtime auto-detects an ASGI app named
`app` exported from a file under /api, so this just re-exports the real
FastAPI app defined in app/main.py.
"""
from app.main import app  # noqa: F401
