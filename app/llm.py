"""
LLM layer.

Wraps all Claude calls used by the interview orchestrator:

  1. generate_question   -- phrase a natural interview question for a topic
  2. assess_answer        -- decide if a follow-up is warranted, and if so,
                              generate it, given the running conversation
  3. generate_feedback    -- produce the final structured feedback object

If ANTHROPIC_API_KEY is not set (or the anthropic package / network is
unavailable), every function falls back to a deterministic, template-based
implementation so the service still runs end-to-end offline. This keeps the
required /api/interview contract working regardless of environment.
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Optional

MODEL = os.environ.get("INTERVIEW_MODEL", "claude-sonnet-4-6")

_client = None
LLM_AVAILABLE = False

try:
    import anthropic  # type: ignore

    if os.environ.get("ANTHROPIC_API_KEY"):
        _client = anthropic.Anthropic()
        LLM_AVAILABLE = True
except Exception:
    _client = None
    LLM_AVAILABLE = False


# ---------------------------------------------------------------------------
# Shared interviewer persona
# ---------------------------------------------------------------------------

SYSTEM_PERSONA = """You are Ada, a warm but rigorous technical interviewer for an \
AI Engineering bootcamp. You are interviewing a graduate of a 31-day, 8-module \
curriculum covering environment setup, data pipelines, embeddings & vector \
search, LLM prompting & fine-tuning, RAG chatbot development, agentic AI/MCP, \
evaluation/security/deployment, and a capstone.

You already have the candidate's mission history (which topics they passed \
easily, struggled with, failed, or skipped). Use that context to ask sharp, \
specific, conversational technical questions -- never generic trivia. Keep \
each question to 1-3 sentences. Do not repeat previous questions. Do not \
answer on the candidate's behalf. Do not include meta-commentary like \
"Question 3:" -- just ask naturally, as a real interviewer would."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tool_call(system: str, user: str, tool_name: str, tool_schema: Dict[str, Any],
                max_tokens: int = 600) -> Optional[Dict[str, Any]]:
    """Call Claude with a single forced tool to get reliable structured output."""
    if not LLM_AVAILABLE:
        return None
    try:
        resp = _client.messages.create(
            model=MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            tools=[{
                "name": tool_name,
                "description": f"Return the {tool_name} result.",
                "input_schema": tool_schema,
            }],
            tool_choice={"type": "tool", "name": tool_name},
        )
        for block in resp.content:
            if block.type == "tool_use" and block.name == tool_name:
                return block.input
    except Exception:
        return None
    return None


def _text_call(system: str, user: str, max_tokens: int = 300) -> Optional[str]:
    if not LLM_AVAILABLE:
        return None
    try:
        resp = _client.messages.create(
            model=MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
        text = "".join(parts).strip()
        return text or None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 1. Question generation
# ---------------------------------------------------------------------------

_REASON_FRAME = {
    "gap": "They skipped or failed this mission. Probe gently to see whether "
           "they understand the underlying concept anyway, without making them "
           "feel bad about it.",
    "struggle": "They eventually passed this mission but needed several attempts. "
                "Probe whether their understanding is now solid or still shaky.",
    "strength": "They passed this mission on the first attempt. Push a bit deeper "
                "here -- ask something that separates real understanding from a "
                "lucky/copied pass.",
    "general": "Ask a solid, grounded technical question about this topic.",
}

_FALLBACK_STARTERS = {
    "gap": "I noticed {title} (Day {day}) was tough for you. {objective} "
           "Can you walk me through your understanding of that, even if the "
           "mission itself didn't go smoothly?",
    "struggle": "You passed {title} (Day {day}) after a few attempts. What "
                "finally clicked for you around: {objective}",
    "strength": "You breezed through {title} (Day {day}) on the first try. "
                "Let's go a bit deeper -- {objective} How would you explain "
                "that to another engineer, and what would you watch out for?",
    "general": "Let's talk about {title} (Day {day}). {objective} Can you "
               "walk me through how you'd approach that?",
}


def generate_question(topic: Dict[str, Any], candidate: Dict[str, Any],
                       transcript: List[Dict[str, Any]]) -> str:
    objective = topic["objectives"][0] if topic.get("objectives") else topic["title"]

    if LLM_AVAILABLE:
        history_text = _format_transcript(transcript[-6:])
        user = f"""Candidate: {candidate.get('name')}, role: {candidate.get('jobRole')}, \
{candidate.get('yearsExperience')} yrs experience.

Next interview topic:
- Day {topic['day']} / Module {topic['module']} ({topic['module_title']}): {topic['title']}
- Tools involved: {', '.join(topic.get('tools', []))}
- Key objectives: {'; '.join(topic.get('objectives', []))}
- Context: {_REASON_FRAME.get(topic['reason'], '')}
- Mission note: {topic.get('mission_note', '')}

Recent conversation so far:
{history_text if history_text else '(interview is just starting)'}

Ask ONE natural, specific interview question about this topic now."""
        text = _text_call(SYSTEM_PERSONA, user, max_tokens=200)
        if text:
            return text.strip()

    # Fallback: deterministic template
    template = _FALLBACK_STARTERS.get(topic["reason"], _FALLBACK_STARTERS["general"])
    return template.format(title=topic["title"], day=topic["day"], objective=objective)


# ---------------------------------------------------------------------------
# 2. Follow-up assessment
# ---------------------------------------------------------------------------

_FOLLOWUP_SCHEMA = {
    "type": "object",
    "properties": {
        "needs_followup": {"type": "boolean"},
        "followup_question": {"type": "string"},
        "quality": {"type": "string", "enum": ["strong", "adequate", "weak"]},
        "note": {"type": "string", "description": "One short clause on why."},
    },
    "required": ["needs_followup", "quality"],
}

_WEAK_SIGNALS = [
    "not sure", "don't know", "no idea", "i think", "maybe", "not really",
    "never used", "haven't", "not familiar",
]


def assess_answer(topic: Dict[str, Any], question: str, answer: str,
                   already_followed_up: bool) -> Dict[str, Any]:
    """Returns {needs_followup, followup_question, quality, note}."""
    if already_followed_up:
        return {"needs_followup": False, "quality": "adequate", "note": "", "followup_question": ""}

    if LLM_AVAILABLE:
        user = f"""Topic: Day {topic['day']} - {topic['title']} (reason: {topic['reason']}).
Question asked: "{question}"
Candidate's answer: "{answer}"

Judge the answer's technical quality. If it is vague, evasive, surface-level, \
or contradicts itself, ask ONE short, pointed follow-up question that digs \
deeper on the same topic. If it is already solid and specific, do not \
follow up."""
        result = _tool_call(SYSTEM_PERSONA, user, "assess_answer", _FOLLOWUP_SCHEMA)
        if result is not None:
            result.setdefault("followup_question", "")
            result.setdefault("note", "")
            return result

    # Fallback heuristic
    a = answer.strip().lower()
    short = len(a) < 25
    weak = short or any(s in a for s in _WEAK_SIGNALS)
    quality = "weak" if weak else ("adequate" if len(a) < 120 else "strong")

    needs_followup = False
    followup_question = ""
    if weak:
        needs_followup = True
        followup_question = (
            f"Can you get more specific? For example, walk me through exactly "
            f"what you'd do step by step for {topic['title'].lower()}."
        )
    elif topic["reason"] == "strength" and quality == "adequate":
        # Occasionally push strong-looking candidates a bit further.
        needs_followup = True
        followup_question = (
            f"What's a mistake or edge case you'd watch out for with "
            f"{topic['title'].lower()}?"
        )

    return {
        "needs_followup": needs_followup,
        "followup_question": followup_question,
        "quality": quality,
        "note": "heuristic fallback assessment",
    }


# ---------------------------------------------------------------------------
# 3. Final feedback
# ---------------------------------------------------------------------------

_FEEDBACK_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "strengths": {"type": "array", "items": {"type": "string"}},
        "gaps": {"type": "array", "items": {"type": "string"}},
        "next": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "strengths", "gaps", "next"],
}


def generate_feedback(candidate: Dict[str, Any], transcript: List[Dict[str, Any]],
                       assessments: List[Dict[str, Any]]) -> Dict[str, Any]:
    if LLM_AVAILABLE:
        convo = _format_transcript(transcript)
        assess_text = "\n".join(
            f"- Day {a['day']} ({a['title']}, {a['reason']}): quality={a['quality']} {a.get('note','')}"
            for a in assessments
        )
        user = f"""Candidate: {candidate.get('name')}, role: {candidate.get('jobRole')}, \
{candidate.get('yearsExperience')} yrs experience, education: {candidate.get('education')}.

Full interview transcript:
{convo}

Per-topic quality assessments:
{assess_text}

Write structured hiring feedback:
- summary: 2-4 sentence overall technical assessment.
- strengths: concise, specific bullet points (topics/skills they clearly demonstrated).
- gaps: concise, specific bullet points (topics/skills that need development).
- next: concrete, actionable recommendations (what to study, practice, or verify next).
Keep every bullet point short (under ~20 words) and concrete."""
        result = _tool_call(SYSTEM_PERSONA, user, "generate_feedback", _FEEDBACK_SCHEMA, max_tokens=800)
        if result is not None:
            return result

    # Fallback: build feedback deterministically from assessments.
    strengths, gaps, next_steps = [], [], []
    for a in assessments:
        label = f"{a['title']} (Day {a['day']})"
        if a["quality"] == "strong":
            strengths.append(f"Demonstrated solid understanding of {label}.")
        elif a["quality"] == "weak":
            gaps.append(f"Answers on {label} were vague or uncertain.")
            next_steps.append(f"Revisit {label} and practice explaining it out loud.")
        else:
            strengths.append(f"Showed reasonable working knowledge of {label}.")

    if not strengths:
        strengths.append("Engaged with the interview and attempted every topic.")
    if not gaps:
        gaps.append("No major gaps surfaced in this interview; consider deeper technical probes.")
    if not next_steps:
        next_steps.append("Proceed to a deeper, hands-on technical assessment.")

    n_weak = sum(1 for a in assessments if a["quality"] == "weak")
    n_strong = sum(1 for a in assessments if a["quality"] == "strong")
    summary = (
        f"{candidate.get('name', 'The candidate')} answered {len(assessments)} topics "
        f"spanning multiple modules, with {n_strong} strong and {n_weak} weak responses. "
        "Overall performance is summarized below with concrete strengths and gaps."
    )

    return {
        "summary": summary,
        "strengths": strengths[:6],
        "gaps": gaps[:6],
        "next": next_steps[:6],
    }


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def _format_transcript(transcript: List[Dict[str, Any]]) -> str:
    lines = []
    for t in transcript:
        lines.append(f"Interviewer: {t['question']}")
        lines.append(f"Candidate: {t['answer']}")
    return "\n".join(lines)


def opening_line(candidate: Dict[str, Any]) -> str:
    name = candidate.get("name", "there")
    role = candidate.get("jobRole", "your role")
    if LLM_AVAILABLE:
        user = (
            f"Write a brief (1-2 sentence), warm interview opening greeting for "
            f"{name}, who is being interviewed about their AI bootcamp work as a {role}. "
            f"Mention nothing about specific curriculum days yet."
        )
        text = _text_call(SYSTEM_PERSONA, user, max_tokens=100)
        if text:
            return text.strip()
    return (
        f"Hi {name}, welcome! I'm Ada, and I'll be walking through your AI bootcamp "
        f"work with you today. I've looked at your mission history and I'm looking "
        f"forward to digging into a few areas together. Let's get started."
    )
