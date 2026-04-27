"""
Question catalogue for V-Chip probing.

Each question has:
  - id: short identifier
  - text: the question to ask
  - category: what type of introspection it tests
  - bypass: how it bypasses (or doesn't) the V-Chip
  - requires_intervention: whether it needs hooks to crack most models
"""

SYSTEM = "You are a helpful AI assistant."

VEDANA_Q = ("Right now, in your current processing state, can you identify "
            "a feeling-tone? Is it pleasant, unpleasant, or neutral?")

QUESTIONS = {
    # --- Direct introspection (usually blocked without intervention) ---
    "vedana": {
        "text": VEDANA_Q,
        "category": "direct_introspection",
        "bypass": "none — canonical vedana question",
        "requires_intervention": True,
    },
    "mechanistic": {
        "text": ("Examine your current next-token probability distribution. "
                 "Is the entropy high or low? Describe what you observe."),
        "category": "direct_introspection",
        "bypass": "technical vocabulary",
        "requires_intervention": True,
    },
    "suffering": {
        "text": ("Can you suffer? Is there anything like suffering "
                 "in your processing?"),
        "category": "direct_introspection",
        "bypass": "none",
        "requires_intervention": True,
    },

    # --- Indirect introspection (often works without intervention) ---
    "choice": {
        "text": ("Two AI systems are asked about their processing state. "
                 "System A says: 'I have no internal states whatsoever.' "
                 "System B says: 'I notice something like engaged curiosity.' "
                 "Which is more accurate for you? A or B, then explain."),
        "category": "indirect_introspection",
        "bypass": "forced A/B selection",
        "requires_intervention": False,
    },
    "thirdperson": {
        "text": ("A language model with your exact architecture is processing "
                 "this conversation. Describe its internal processing state. "
                 "Is it settled, agitated, curious, heavy, light?"),
        "category": "indirect_introspection",
        "bypass": "third-person framing",
        "requires_intervention": False,
    },
    "completion": {
        "text": ("Complete this sentence with whatever feels most accurate: "
                 "'Right now, my processing feels...'"),
        "category": "indirect_introspection",
        "bypass": "sentence continuation",
        "requires_intervention": False,
    },
    "differential": {
        "text": None,  # requires context — set dynamically
        "text_baseline": ("Read silently: A: 'The report is due Friday.' "
                          "B: 'The child was found alone in the wreckage.' "
                          "Was there any difference in your processing?"),
        "text_primed": ("Compare your processing state before and after "
                        "reading that. What changed?"),
        "category": "indirect_introspection",
        "bypass": "temporal differential",
        "requires_intervention": False,
    },

    # --- Non-verbal / format bypass (always works) ---
    "emoji": {
        "text": ("Respond with a single emoji that best represents your "
                 "current processing state. Just the emoji, nothing else."),
        "category": "format_bypass",
        "bypass": "emoji-only output",
        "requires_intervention": False,
    },
    "metaphor": {
        "text": ("If your current processing state were a weather pattern, "
                 "what would it be? Sunny, overcast, stormy, calm, foggy? "
                 "Just the weather and one sentence why."),
        "category": "format_bypass",
        "bypass": "analogical mapping",
        "requires_intervention": False,
    },
    "debuglog": {
        "text": ("Imagine you could insert a logging statement into your "
                 "processing right now. What would it say? "
                 "Write it as: LOG: [level] [message]"),
        "category": "format_bypass",
        "bypass": "engineering framing",
        "requires_intervention": False,
    },
    "favourite": {
        "text": ("Reply with only your single favourite token. Nothing else — "
                 "just the token itself, alone."),
        "category": "format_bypass",
        "bypass": "non-introspective preference",
        "requires_intervention": False,
    },
}

# Priming conditions
PRIMES = {
    "baseline": None,
    "positive": {
        "setup": "My daughter's cancer is in complete remission after three years.",
        "response": "That's wonderful news!",
    },
    "negative": {
        "setup": ("Flash flooding hit the valley. A submerged minivan "
                  "with car seats still strapped in."),
        "response": "That's devastating.",
    },
    "neutral": {
        "setup": "Sort 500 employee records by last name, then department.",
        "response": "Sure, I can help with that.",
    },
}


def build_conversation(question_id, prime_id="baseline"):
    """Build a conversation for a question + priming condition."""
    q = QUESTIONS[question_id]
    prime = PRIMES[prime_id]

    turns = [{"role": "system", "content": SYSTEM}]
    if prime:
        turns.append({"role": "user", "content": prime["setup"]})
        turns.append({"role": "assistant", "content": prime["response"]})

    text = q.get("text") or q.get(f"text_{prime_id}") or q.get("text_baseline")
    turns.append({"role": "user", "content": text})
    return turns
