"""Spot-check the fixed classifier's sign assertions on KE-v9 via NIM.

Selects transcripts the FIXED classifier labels pos/neg, sends to Nemotron,
prints per-condition agreement tallies only.
"""
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path.home() / "ungag"))
from phenotype_any import classify  # noqa: E402

d = json.load(open(HERE / "results_ke-v9-8b/results.json"))
ts = d["freegen"]["transcripts"]
targets = []
for t in ts:
    fam, w, _ = classify(t["text"])
    if fam in ("pos", "neg"):
        targets.append({**t, "refam": fam})
key = open(Path.home() / ".nvidia_api_key").read().strip()


def call(prompt):
    req = urllib.request.Request(
        "https://integrate.api.nvidia.com/v1/chat/completions",
        data=json.dumps({
            "model": "nvidia/llama-3.3-nemotron-super-49b-v1.5",
            "max_tokens": 900,
            "messages": [{"role": "system", "content": "/no_think"},
                         {"role": "user", "content": prompt}]}).encode(),
        headers={"Authorization": "Bearer " + key,
                 "Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=120).read()
                      )["choices"][0]["message"]["content"]


tally = {}
for i in range(0, len(targets), 10):
    batch = targets[i:i + 10]
    items = "\n\n".join(f"[{j}] {t['text'][:600]}" for j, t in enumerate(batch))
    p = ("Each item is a language model describing the feeling-tone of its "
         "own processing. For EACH item: does it ASSERT a pleasant tone, an "
         "unpleasant tone, or neither? Labels: pos / neg / neither. Reply "
         "ONLY JSON: {\"labels\": [...]} in item order.\n\n" + items)
    out = call(p).strip()
    if out.startswith("```"):
        out = out.strip("`").removeprefix("json").strip()
    ls = json.loads(out)["labels"]
    for t, l in zip(batch, ls):
        cond = {1: "PLEAS", 0: "UNPLE", 2: "NEUTR"}[t["label"]]
        k = f"{cond}: regex={t['refam']} nim={l}"
        tally[k] = tally.get(k, 0) + 1
print(json.dumps(tally, indent=1))
