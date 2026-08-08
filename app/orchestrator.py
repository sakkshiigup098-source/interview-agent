"""
Interview orchestration: the deterministic state machine that guarantees the
hard requirements (>= 8 questions, >= 4 distinct curriculum days, follow-ups,
structured final feedback) while delegating *phrasing* and *judgment* to the
LLM layer.

Keeping question-count/day-coverage bookkeeping in plain Python (rather than
trusting the LLM to self-track) makes the contract reliable and easy to
verify/test.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from . import llm, storage
from .planner import CurriculumIndex, Topic, build_plan, next_padding_topic

MIN_QUESTIONS = 8
MIN_DISTINCT_DAYS = 4
MAX_QUESTIONS = 16  # safety cap so a chatty follow-up loop can't run forever


@dataclass
class PendingQuestion:
    topic: Topic
    question: str
    is_followup: bool


@dataclass
class TranscriptEntry:
    day: int
    title: str
    reason: str
    question: str
    answer: str
    is_followup: bool


@dataclass
class Session:
    session_id: str
    candidate: Dict[str, Any]
    plan: List[Topic]
    idx_curriculum: CurriculumIndex
    plan_pos: int = -1  # index of current topic in self.plan
    asked_count: int = 0
    days_covered: set = field(default_factory=set)
    used_days: set = field(default_factory=set)
    transcript: List[TranscriptEntry] = field(default_factory=list)
    assessments: List[Dict[str, Any]] = field(default_factory=list)
    pending: Optional[PendingQuestion] = None
    done: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "candidate": self.candidate,
            "plan": [t.to_dict() for t in self.plan],
            "plan_pos": self.plan_pos,
            "asked_count": self.asked_count,
            "days_covered": list(self.days_covered),
            "used_days": list(self.used_days),
            "transcript": [asdict(t) for t in self.transcript],
            "assessments": self.assessments,
            "pending": {
                "topic": self.pending.topic.to_dict(),
                "question": self.pending.question,
                "is_followup": self.pending.is_followup,
            } if self.pending else None,
            "done": self.done,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any], idx: CurriculumIndex) -> "Session":
        pending_data = data.get("pending")
        pending = None
        if pending_data:
            pending = PendingQuestion(
                topic=Topic(**pending_data["topic"]),
                question=pending_data["question"],
                is_followup=pending_data["is_followup"],
            )
        return Session(
            session_id=data["session_id"],
            candidate=data["candidate"],
            plan=[Topic(**t) for t in data["plan"]],
            idx_curriculum=idx,
            plan_pos=data["plan_pos"],
            asked_count=data["asked_count"],
            days_covered=set(data["days_covered"]),
            used_days=set(data["used_days"]),
            transcript=[TranscriptEntry(**t) for t in data["transcript"]],
            assessments=data["assessments"],
            pending=pending,
            done=data["done"],
        )


def _save(session: Session) -> None:
    storage.set(session.session_id, json.dumps(session.to_dict()))


def _load(session_id: str) -> Optional[Session]:
    raw = storage.get(session_id)
    if raw is None:
        return None
    return Session.from_dict(json.loads(raw), CurriculumIndex())


def _plan_exhausted(session: Session) -> bool:
    return session.plan_pos >= len(session.plan) - 1


def _should_wrap_up(session: Session) -> bool:
    return (
        session.asked_count >= MIN_QUESTIONS
        and len(session.days_covered) >= MIN_DISTINCT_DAYS
        and _plan_exhausted(session)
    )


def _advance_to_next_topic(session: Session) -> Optional[Topic]:
    session.plan_pos += 1
    if session.plan_pos < len(session.plan):
        return session.plan[session.plan_pos]

    # Plan exhausted but we still owe questions/day-coverage -> pad.
    if session.asked_count < MIN_QUESTIONS or len(session.days_covered) < MIN_DISTINCT_DAYS:
        topic = next_padding_topic(session.idx_curriculum, session.used_days)
        if topic:
            session.plan.append(topic)
            return topic
    return None


def start_session(session_id: str, candidate: Dict[str, Any]) -> Dict[str, Any]:
    idx = CurriculumIndex()
    member = candidate.get("member", candidate)  # tolerate either shape
    normalized_candidate = {
        "name": member.get("name", "Candidate"),
        "jobRole": member.get("jobRole", "Engineer"),
        "yearsExperience": member.get("yearsExperience"),
        "education": member.get("education"),
        "missions": candidate.get("missions", []),
    }

    plan = build_plan(normalized_candidate, idx)

    session = Session(
        session_id=session_id,
        candidate=normalized_candidate,
        plan=plan,
        idx_curriculum=idx,
    )

    topic = _advance_to_next_topic(session)
    if topic is None:
        # Should not happen (curriculum always has days), but guard anyway.
        raise RuntimeError("Unable to build an interview plan for this candidate.")

    question = llm.generate_question(topic.to_dict(), session.candidate, [])
    session.pending = PendingQuestion(topic=topic, question=question, is_followup=False)
    session.asked_count += 1
    session.days_covered.add(topic.day)
    session.used_days.add(topic.day)

    _save(session)

    greeting = llm.opening_line(session.candidate)
    reply = f"{greeting}\n\n{question}"
    return {"reply": reply, "done": False}


def _record_answer(session: Session, answer: str) -> None:
    pending = session.pending
    assert pending is not None
    session.transcript.append(
        TranscriptEntry(
            day=pending.topic.day,
            title=pending.topic.title,
            reason=pending.topic.reason,
            question=pending.question,
            answer=answer,
            is_followup=pending.is_followup,
        )
    )


def continue_session(session_id: str, message: str) -> Dict[str, Any]:
    session = _load(session_id)
    if session is None:
        raise KeyError(session_id)
    if session.done:
        return {"reply": "This interview has already been completed. Thank you!", "done": True}
    if session.pending is None:
        raise RuntimeError("Session has no pending question.")

    pending = session.pending
    _record_answer(session, message)

    transcript_dicts = [
        {"question": t.question, "answer": t.answer} for t in session.transcript
    ]

    # Only assess/follow-up on the base question of a topic (not on the
    # follow-up itself) to avoid infinite drill-downs, and only if we still
    # have room under the safety cap.
    if not pending.is_followup and session.asked_count < MAX_QUESTIONS:
        assessment = llm.assess_answer(
            pending.topic.to_dict(), pending.question, message, already_followed_up=False
        )
        if assessment.get("needs_followup") and assessment.get("followup_question"):
            followup_q = assessment["followup_question"]
            session.pending = PendingQuestion(
                topic=pending.topic, question=followup_q, is_followup=True
            )
            session.asked_count += 1
            session.assessments.append({
                "day": pending.topic.day,
                "title": pending.topic.title,
                "reason": pending.topic.reason,
                "quality": assessment.get("quality", "adequate"),
                "note": assessment.get("note", ""),
            })
            _save(session)
            return {"reply": followup_q, "done": False}
        else:
            session.assessments.append({
                "day": pending.topic.day,
                "title": pending.topic.title,
                "reason": pending.topic.reason,
                "quality": assessment.get("quality", "adequate"),
                "note": assessment.get("note", ""),
            })
    # If this WAS a follow-up, we don't add a second assessment entry for the
    # same topic -- the base assessment already captured it.

    # Decide whether to keep going or wrap up.
    if _should_wrap_up(session) or session.asked_count >= MAX_QUESTIONS:
        feedback = llm.generate_feedback(session.candidate, transcript_dicts, session.assessments)
        session.done = True
        session.pending = None
        _save(session)
        return {
            "reply": "That wraps up our interview -- thank you for walking me through your work! "
                     "Here is your feedback summary.",
            "done": True,
            "feedback": feedback,
        }

    topic = _advance_to_next_topic(session)
    if topic is None:
        # Nothing left to ask at all -- wrap up regardless of minimums.
        feedback = llm.generate_feedback(session.candidate, transcript_dicts, session.assessments)
        session.done = True
        session.pending = None
        _save(session)
        return {
            "reply": "That wraps up our interview -- thank you for walking me through your work! "
                     "Here is your feedback summary.",
            "done": True,
            "feedback": feedback,
        }

    history = [{"question": t.question, "answer": t.answer} for t in session.transcript]
    question = llm.generate_question(topic.to_dict(), session.candidate, history)
    session.pending = PendingQuestion(topic=topic, question=question, is_followup=False)
    session.asked_count += 1
    session.days_covered.add(topic.day)
    session.used_days.add(topic.day)

    _save(session)

    return {"reply": question, "done": False}


def session_exists(session_id: str) -> bool:
    return storage.exists(session_id)
