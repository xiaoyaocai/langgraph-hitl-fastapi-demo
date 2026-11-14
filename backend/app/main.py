import sys
from pathlib import Path

# IDE Run/Debug puts this file's dir (backend/app) first on sys.path, which makes
# `import app` miss the project package and hit any site-packages named `app`.
# Always force backend/ to the front so `app.*` resolves to this project.
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_backend = str(_BACKEND_ROOT)
if _backend in sys.path:
    sys.path.remove(_backend)
sys.path.insert(0, _backend)

from dotenv import load_dotenv

# Load backend/.env before any module that constructs ChatOpenAI
load_dotenv(_BACKEND_ROOT / ".env")

from fastapi import FastAPI
from app.lesson_01_blocking import router as lesson_01_router
from app.lesson_02_streaming import router as lesson_02_router
from app.lesson_03_async_mcp import router as lesson_03_router
from app.cors_config import add_cors_middleware

app = FastAPI()
add_cors_middleware(app)

# Register lesson routers
app.include_router(lesson_01_router)
app.include_router(lesson_02_router)
app.include_router(lesson_03_router)

if __name__ == "__main__":
    import uvicorn

    # PyCharm/pydevd + reload on Windows re-spawns a child with a raw
    # backslash path (SyntaxError) and can leave port 8000 occupied.
    # Keep reload for normal Run; disable it under a debugger.
    is_debugging = "pydevd" in sys.modules or "debugpy" in sys.modules
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=not is_debugging,
    )