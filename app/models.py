"""
Request/response models for the AI Interview Agent.

These mirror the contract defined in technical-spec.md exactly:

    POST /api/interview
    -> start:  {"sessionId": "...", "candidate": {...}}
    -> turn:   {"sessionId": "...", "message": "..."}
    <- reply:  {"reply": "...", "done": false}
    <- final:  {"reply": "...", "done": true, "feedback": {...}}
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class InterviewRequest(BaseModel):
    sessionId: str
    candidate: Optional[Dict[str, Any]] = None
    message: Optional[str] = None


class Feedback(BaseModel):
    summary: str
    strengths: List[str] = Field(default_factory=list)
    gaps: List[str] = Field(default_factory=list)
    next: List[str] = Field(default_factory=list)


class InterviewResponse(BaseModel):
    reply: str
    done: bool
    feedback: Optional[Feedback] = None
