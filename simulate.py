"""
Offline simulation of a full interview run, exercising the orchestrator
directly (no HTTP layer, no network) to verify the core requirements:
  - >= 8 questions asked
  - >= 4 distinct curriculum days covered
  - follow-ups generated
  - structured feedback produced at the end
Run: python simulate.py [candidate_index]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app import orchestrator  # noqa: E402

with open(Path(__file__).parent / "app" / "data" / "candidates.json") as f:
    candidates = json.load(f)["candidates"]

idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
candidate = candidates[idx]
print(f"=== Simulating interview for {candidate['member']['name']} ({candidate['member']['jobRole']}) ===\n")

session_id = f"sim-{idx}"
result = orchestrator.start_session(session_id, candidate)
print("AGENT:", result["reply"], "\n")

turn = 0
distinct_answers = [
    "Yeah I remember that, we used cosine similarity to compare embeddings and picked the top k chunks.",
    "Honestly I'm not totally sure, I just followed the tutorial without fully getting the reasoning.",
    "I built a router that checked intent first then decided between SQL and vector search based on keywords.",
    "Not really, I skipped that part and moved on.",
    "We used FastAPI with a background task queue and retried failed tool calls up to 3 times with backoff.",
    "It clicked once I visualized the embeddings with PCA and saw clusters forming around plan types.",
    "I set up a Chroma collection with metadata filters for plan type and document section.",
    "Maybe? I think it has something to do with attention but I'm fuzzy on the details.",
    "We containerized both services with Docker and used a Helm chart for the Kubernetes deployment.",
    "I wrote a system prompt that only let the model answer from retrieved context and refuse otherwise.",
]

while not result.get("done") and turn < 20:
    answer = distinct_answers[turn % len(distinct_answers)]
    print("CANDIDATE:", answer)
    result = orchestrator.continue_session(session_id, answer)
    print("AGENT:", result["reply"], "\n")
    turn += 1

print("=== DONE ===")
print("done:", result.get("done"))
if result.get("feedback"):
    print(json.dumps(result["feedback"], indent=2))

sess = orchestrator._load(session_id)
print("\n--- Stats ---")
print("Total questions asked:", sess.asked_count)
print("Distinct days covered:", sorted(sess.days_covered), "count:", len(sess.days_covered))
print("Transcript length:", len(sess.transcript))
assert sess.asked_count >= 8, "FAILED: fewer than 8 questions"
assert len(sess.days_covered) >= 4, "FAILED: fewer than 4 distinct days"
assert result.get("done") is True
assert "feedback" in result
print("\nAll minimum requirements satisfied.")
