"""
Interview planning.

Turns a candidate's mission history against the 31-day curriculum into a
prioritized list of interview topics. The goal is to make the interview feel
targeted rather than generic:

  - GAP     -> mission failed or skipped entirely. We probe whether the
              candidate actually understands the concept despite not
              finishing/passing the mission.
  - STRUGGLE-> mission eventually passed, but took 3+ attempts. We probe
              whether understanding "stuck" or if it was trial and error.
  - STRENGTH-> mission passed on the first (or second) attempt on a
              meaningfully hard day (module 3+). We probe for real depth,
              since first-try passes can also mean the task was easy or
              copied.
  - GENERAL -> fallback filler topics (e.g. day 1) used only if we need to
              pad out to the question minimum.

The resulting plan always spans at least 4 distinct curriculum days /
modules and has enough topics that, even with zero follow-up questions,
the interview would still hit 6 base questions -- comfortably reachable to
8 total once follow-ups are added.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

DATA_DIR = Path(__file__).parent / "data"

MIN_QUESTIONS = 8
MIN_DISTINCT_DAYS = 4
TARGET_TOPIC_COUNT = 6  # base topics chosen up front; follow-ups add more


@dataclass
class Topic:
    day: int
    title: str
    type: str
    tools: List[str]
    objectives: List[str]
    module: int
    module_title: str
    reason: str  # "gap" | "struggle" | "strength" | "general"
    mission_note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "day": self.day,
            "title": self.title,
            "type": self.type,
            "tools": self.tools,
            "objectives": self.objectives,
            "module": self.module,
            "module_title": self.module_title,
            "reason": self.reason,
            "mission_note": self.mission_note,
        }


def load_curriculum() -> Dict[str, Any]:
    with open(DATA_DIR / "curriculum.json", "r", encoding="utf-8") as f:
        return json.load(f)


class CurriculumIndex:
    """Fast lookups over curriculum.json."""

    def __init__(self, curriculum: Optional[Dict[str, Any]] = None):
        self.curriculum = curriculum or load_curriculum()
        self.days_by_number: Dict[int, Dict[str, Any]] = {
            d["day"]: d for d in self.curriculum["days"]
        }
        self.modules = self.curriculum["modules"]

    def module_for_day(self, day: int) -> Dict[str, Any]:
        for m in self.modules:
            lo, hi = m["days"][0], m["days"][1]
            if lo <= day <= hi:
                return m
        return {"n": 0, "title": "Unknown Module"}

    def day_info(self, day: int) -> Optional[Dict[str, Any]]:
        return self.days_by_number.get(day)

    def all_days_sorted(self) -> List[int]:
        return sorted(self.days_by_number.keys())


def _make_topic(idx: CurriculumIndex, day: int, reason: str, note: str = "") -> Optional[Topic]:
    info = idx.day_info(day)
    if not info:
        return None
    mod = idx.module_for_day(day)
    return Topic(
        day=day,
        title=info["title"],
        type=info.get("type", "BUILD"),
        tools=info.get("tools", []),
        objectives=info.get("objectives", []),
        module=mod.get("n", 0),
        module_title=mod.get("title", ""),
        reason=reason,
        mission_note=note,
    )


def build_plan(candidate: Dict[str, Any], idx: Optional[CurriculumIndex] = None) -> List[Topic]:
    idx = idx or CurriculumIndex()
    missions = candidate.get("missions", [])

    gaps: List[Topic] = []
    struggles: List[Topic] = []
    strengths: List[Topic] = []
    seen_days: set = set()

    for m in missions:
        day = m.get("day")
        if day is None or day in seen_days:
            continue
        passed = m.get("passed")
        skipped = m.get("skipped", False)
        attempts = m.get("attempts", 1)

        if skipped:
            t = _make_topic(idx, day, "gap", "Skipped this mission.")
            if t:
                gaps.append(t)
                seen_days.add(day)
        elif passed is False:
            t = _make_topic(
                idx, day, "gap", f"Did not pass after {attempts} attempt(s)."
            )
            if t:
                gaps.append(t)
                seen_days.add(day)
        elif passed is True and attempts and attempts >= 3:
            t = _make_topic(
                idx, day, "struggle", f"Passed after {attempts} attempts."
            )
            if t:
                struggles.append(t)
                seen_days.add(day)
        elif passed is True and attempts and attempts <= 1:
            t = _make_topic(idx, day, "strength", "Passed on the first attempt.")
            if t:
                strengths.append(t)
                seen_days.add(day)

    # Prefer harder / later modules for "strength" probes -- a first-try pass
    # on day 3 setup is less interesting than a first-try pass on day 23 MCP.
    strengths.sort(key=lambda t: t.day, reverse=True)
    # Prefer the most-attempted struggles and the most consequential gaps.
    struggles.sort(key=lambda t: t.day)
    gaps.sort(key=lambda t: t.day)

    plan: List[Topic] = []
    used_days: set = set()

    def add_from(pool: List[Topic], n: int):
        added = 0
        for t in pool:
            if added >= n:
                break
            if t.day in used_days:
                continue
            plan.append(t)
            used_days.add(t.day)
            added += 1

    # Balanced picks: up to 2 gaps, 2 struggles, 2 strengths first.
    add_from(gaps, 2)
    add_from(struggles, 2)
    add_from(strengths, 2)

    # Top up with whatever's left (more gaps > struggles > strengths) until
    # we hit the target topic count or run out of categorized missions.
    for pool in (gaps, struggles, strengths):
        if len(plan) >= TARGET_TOPIC_COUNT:
            break
        add_from(pool, TARGET_TOPIC_COUNT - len(plan))

    # Ensure minimum distinct-day diversity by pulling in any other
    # completed mission days not yet used.
    if len({t.day for t in plan}) < MIN_DISTINCT_DAYS or len(plan) < TARGET_TOPIC_COUNT:
        for m in missions:
            day = m.get("day")
            if day is None or day in used_days:
                continue
            t = _make_topic(idx, day, "general", "Additional completed mission.")
            if t:
                plan.append(t)
                used_days.add(t.day)
            if len(plan) >= TARGET_TOPIC_COUNT and len(used_days) >= MIN_DISTINCT_DAYS:
                break

    # Final safety net: pad with generic curriculum days the candidate has no
    # record for at all, so a very thin profile still yields a full plan.
    if len(plan) < TARGET_TOPIC_COUNT or len({t.day for t in plan}) < MIN_DISTINCT_DAYS:
        for day in idx.all_days_sorted():
            if day in used_days:
                continue
            t = _make_topic(idx, day, "general", "General curriculum topic.")
            if t:
                plan.append(t)
                used_days.add(t.day)
            if len(plan) >= TARGET_TOPIC_COUNT and len(used_days) >= MIN_DISTINCT_DAYS:
                break

    # Order the final plan roughly by curriculum day so the interview flows
    # chronologically through the program rather than jumping randomly.
    plan.sort(key=lambda t: t.day)
    return plan


def next_padding_topic(idx: CurriculumIndex, used_days: set) -> Optional[Topic]:
    """Used at runtime if we still need more questions after the plan is exhausted."""
    for day in idx.all_days_sorted():
        if day not in used_days:
            return _make_topic(idx, day, "general", "Extra topic to reach question minimum.")
    return None
