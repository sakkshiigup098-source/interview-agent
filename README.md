# AI Interview Agent

A conversational technical interview agent that quizzes a bootcamp graduate
on their own curriculum history, adapts its questions to their actual
strengths/gaps, asks follow-ups based on answer quality, and produces
structured hiring feedback at the end.

## Endpoint

```
POST /api/interview
```

Matches the technical spec exactly:

**Start a session**
```json
{ "sessionId": "abc-123", "candidate": { "member": {...}, "missions": [...] } }
→ { "reply": "...", "done": false }
```

**Continue a session**
```json
{ "sessionId": "abc-123", "message": "candidate's answer" }
→ { "reply": "...", "done": false }
```

**Final turn**
```json
{
  "reply": "...",
  "done": true,
  "feedback": { "summary": "...", "strengths": [...], "gaps": [...], "next": [...] }
}
```

## How it decides what to ask

`app/planner.py` reads the candidate's `missions[]` against `curriculum.json`
and buckets each completed/attempted day into:

- **gap** – failed or skipped → probe whether they understand the concept anyway
- **struggle** – passed but took 3+ attempts → probe whether it really stuck
- **strength** – passed on the first attempt → push deeper to separate real
  understanding from a lucky/copied pass
- **general** – fallback filler pulled from the wider curriculum if a
  candidate's history is too thin to hit the minimums on its own

It picks a balanced set of ~6 topics spanning multiple modules/days, so the
interview is always specific to that person, not generic trivia.

## How the conversation flows

`app/orchestrator.py` is a deterministic state machine (per `sessionId`, kept
in memory) that guarantees the hard requirements:

- **≥ 8 questions**, **≥ 4 distinct curriculum days**, always satisfied by
  construction (topic count + padding), even for thin profiles.
- After each **base** question, the answer is judged (`app/llm.py:
  assess_answer`) — vague/short/uncertain answers trigger exactly one
  targeted follow-up before moving on, so follow-ups are grounded in what
  the candidate actually said, not scripted.
- Full running transcript + per-topic quality assessments are kept in
  session state and fed into the final feedback generation.
- A `MAX_QUESTIONS` safety cap (16) prevents runaway follow-up loops.

## LLM layer

`app/llm.py` uses the Anthropic API (Claude) for:
1. Phrasing each question naturally, in context of the running conversation.
2. Judging answer quality and writing a follow-up question when warranted
   (via forced tool-use for reliable structured output).
3. Writing the final `summary` / `strengths` / `gaps` / `next` feedback.

If `ANTHROPIC_API_KEY` is not set (or the API is unreachable), every one of
these falls back to a deterministic, template-based implementation, so the
service still runs the full required flow offline. This was verified with
`simulate.py`.

## Running it locally

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-...   # optional — falls back to offline mode without it
uvicorn app.main:app --reload --port 8000
```

## Deploying to Vercel

Vercel runs Python as **serverless functions** — there's no long-lived
process, so session state can't just live in a plain Python dict (it may
not survive between requests). `app/storage.py` handles this: it
transparently switches from in-memory storage to **Upstash Redis** (REST
API, works great from serverless) whenever these env vars are present:

```
UPSTASH_REDIS_REST_URL
UPSTASH_REDIS_REST_TOKEN
```

**Setup:**
1. Create a free Redis database at https://upstash.com → copy the REST URL
   and token from its dashboard.
2. In your Vercel project → Settings → Environment Variables, add:
   - `UPSTASH_REDIS_REST_URL`
   - `UPSTASH_REDIS_REST_TOKEN`
   - `ANTHROPIC_API_KEY` (optional)
3. Push this repo to GitHub, then **Import Project** on Vercel and point it
   at the repo. `vercel.json` + `api/index.py` are already set up to route
   all traffic to the FastAPI app — no extra config needed.

Without the Upstash env vars, the app still deploys and runs, but session
state may not persist reliably across requests on Vercel's serverless
runtime — fine for quick manual testing, not for a real multi-turn
interview. Locally with `uvicorn` (a single persistent process), you never
need Upstash — the in-memory fallback is correct there.

Then:
```bash
curl -X POST localhost:8000/api/interview \
  -H "Content-Type: application/json" \
  -d '{"sessionId":"demo-1","candidate": <one entry from candidates.json>}'
```
Take the `reply` from each response, prompt the user, then POST
`{"sessionId":"demo-1","message":"<their answer>"}` on each subsequent turn
until `done: true`, at which point `feedback` is included.

## Offline verification (no server needed)

```bash
python simulate.py 0   # simulate a full interview for candidates[0]
```
This drives the orchestrator directly and asserts: ≥8 questions asked, ≥4
distinct days covered, and structured feedback is returned.

## Project layout

```
app/
  main.py          FastAPI app, exposes POST /api/interview
  models.py         Request/response pydantic models
  planner.py         Builds a candidate-specific topic plan from curriculum.json
  orchestrator.py     Session state machine (question count, day coverage, follow-ups)
  llm.py              Claude calls + offline fallback for question/follow-up/feedback generation
  data/
    curriculum.json
    candidates.json
simulate.py           Offline end-to-end test harness
```
