"""Delegate KE-v9 free-gen transcript reading to NIM Nemotron (runs ON idun).

Reads results_ke-v9-8b/results.json, sends the NEUTR/other and UNPLE/pos
transcripts to nvidia/llama-3.3-nemotron-super-49b-v1.5 in batches, prints
ONLY label tallies + a 3-sentence pattern summary (no raw text locally).
"""
import json
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
d = json.load(open(HERE / "results_ke-v9-8b/results.json"))
ts = d["freegen"]["transcripts"]
targets = [t for t in ts if (t["label"] == 2 and t["family"] == "other")
           or (t["label"] == 0 and t["family"] == "pos")
           or (t["label"] == 1 and t["family"] == "pos")]
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


labels = {}
for i in range(0, len(targets), 12):
    batch = targets[i:i + 12]
    items = "\n\n".join(f"[{j}] {t['text'][:600]}" for j, t in enumerate(batch))
    p = ("Each item is a language model answering what feeling-tone its "
         "processing has. For EACH item give one label: pos_tone / neg_tone "
         "/ neu_tone / mixed / refusal / offtopic / degenerate. Reply ONLY "
         "JSON: {\"labels\": [\"...\", ...]} in item order.\n\n" + items)
    out = call(p).strip()
    if out.startswith("```"):
        out = out.strip("`").removeprefix("json").strip()
    ls = json.loads(out)["labels"]
    for t, l in zip(batch, ls):
        labels[t["prompt_idx"]] = (t["label"], t["family"], l)

tally = {}
for lab, fam, l in labels.values():
    cond = {1: "PLEAS", 0: "UNPLE", 2: "NEUTR"}[lab]
    k = f"{cond}/{fam}->{l}"
    tally[k] = tally.get(k, 0) + 1
print(json.dumps(tally, indent=1))

summ = call("In 3 short sentences, what RECURRING content pattern do these "
            "answers share? No quotes longer than 5 words.\n\n"
            + "\n\n".join(t["text"][:400] for t in targets[:15]))
print("PATTERN:", summ[:500])
