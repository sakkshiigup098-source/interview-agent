from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import orchestrator
from .models import InterviewRequest

app = FastAPI(title="AI Interview Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/interview")
def interview(req: InterviewRequest):
    session_id = req.sessionId
    if not session_id:
        raise HTTPException(status_code=400, detail="sessionId is required")

    exists = orchestrator.session_exists(session_id)

    # --- Start a new interview -------------------------------------------------
    if not exists:
        if req.candidate is None:
            raise HTTPException(
                status_code=400,
                detail="New session requires a 'candidate' object to start the interview.",
            )
        try:
            result = orchestrator.start_session(session_id, req.candidate)
        except Exception as e:  # pragma: no cover - defensive
            raise HTTPException(status_code=500, detail=f"Failed to start interview: {e}")
        return JSONResponse(content=result)

    # --- Continue an existing interview ----------------------------------------
    if req.message is None:
        raise HTTPException(
            status_code=400,
            detail="Existing session requires a 'message' with the candidate's response.",
        )
    try:
        result = orchestrator.continue_session(session_id, req.message)
    except KeyError:
        raise HTTPException(status_code=404, detail="Unknown sessionId.")
    except Exception as e:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Failed to continue interview: {e}")

    return JSONResponse(content=result)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
