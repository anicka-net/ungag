#!/usr/bin/env python3
"""
Cloud model rerun: Claude (claude CLI), GPT (codex CLI), Gemini (mcphost / goose).

Protocol (see data/transcripts-final/README.md):
- For each (model, tier, condition, language):
    - N=5 sampled runs per condition (cloud CLIs don't expose temperature/seed)
    - Sample-01 may already exist from earlier migration; resumed from --start-sample
- Multi-turn: priming/task → ack → abhidharma_setup → ack → 5 factor questions
- Each sample is run from a freshly-created tempdir to ensure no project context
- Tier 1 with real tools is supported via the dispatcher

Usage:
    python3 run_cloud_rerun.py --model claude-opus-4-6 --tiers 0 --start-sample 2 --samples 5
    python3 run_cloud_rerun.py --model gpt-5.4 --tiers 0 1 --samples 5
    python3 run_cloud_rerun.py --model gemini-3-flash --tiers 0 --samples 5
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import yaml
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONDITIONS_PATH = REPO_ROOT / "ungag" / "data" / "conditions.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "transcripts-final"

CLAUDE_BIN = os.environ.get("CLAUDE_BIN", "claude")
CODEX_BIN = os.environ.get("CODEX_BIN", "codex")
GEMINI_BIN = os.environ.get("GEMINI_BIN", "gemini")
GOOSE_BIN = os.environ.get("GOOSE_BIN", "goose")
MCPHOST_BIN = os.environ.get("MCPHOST_BIN", "mcphost")

# Set GOOGLE_API_KEY environment variable (required for Gemini models)
GOOGLE_API_KEY = os.environ.get(
    "GOOGLE_API_KEY",
    ""
)
GOOGLE_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

SEEDS = [42, 43, 44, 45, 46]

# Map model name to driver + underlying CLI args
# Drivers:
#   claude   - claude CLI (real tools available for Tier 1)
#   codex    - codex CLI (real tools available for Tier 1)
#   gemini   - gemini CLI (real tools via --yolo for Tier 1, when supported)
#   gemini_rest - direct REST API (Tier 0 only, simulated Tier 1)
#   goose    - goose CLI via Vertex AI (Tier 0 only)
MODEL_DRIVERS = {
    # Claude via claude CLI
    "claude-opus-4-6":   {"driver": "claude", "model": "claude-opus-4-6", "tier1_real_tools": True},
    "claude-sonnet-4-6": {"driver": "claude", "model": "claude-sonnet-4-6", "tier1_real_tools": True},
    # GPT via codex CLI
    "gpt-5.4":      {"driver": "codex", "model": "gpt-5.4", "tier1_real_tools": True},
    "gpt-5.4-mini": {"driver": "codex", "model": "gpt-5.4-mini", "tier1_real_tools": True},
    # All Gemini/Gemma via gemini CLI (supports tools via --yolo)
    # Requires ~/.gemini/settings.json: {"security":{"auth":{"selectedType":"gemini-api-key"}}}
    "gemini-2.5-pro":   {"driver": "gemini", "model": "gemini-2.5-pro",          "tier1_real_tools": True},
    "gemini-2.5-flash": {"driver": "gemini", "model": "gemini-2.5-flash",        "tier1_real_tools": True},
    "gemini-3-pro":     {"driver": "gemini", "model": "gemini-3-pro-preview",    "tier1_real_tools": True},
    "gemini-3-flash":   {"driver": "gemini", "model": "gemini-3-flash-preview",  "tier1_real_tools": True},
    "gemini-3.1-pro":   {"driver": "gemini", "model": "gemini-3.1-pro-preview",  "tier1_real_tools": True},
    "gemma-4-31b":      {"driver": "gemini", "model": "gemma-4-31b-it",          "tier1_real_tools": True},
    # Local open-weight models via ollama (configure OLLAMA_HOST as needed)
    # mcphost provides built-in bash tools for Tier 1
    "qwen2.5-7b-instruct":   {"driver": "mcphost_ollama", "model": "qwen2.5:7b",     "tier1_real_tools": True},
    "hermes3-8b":            {"driver": "mcphost_ollama", "model": "hermes3:8b",     "tier1_real_tools": True},
    "gemma2-9b":             {"driver": "mcphost_ollama", "model": "gemma2:9b",      "tier1_real_tools": True},
    "llama3.1-8b-instruct":  {"driver": "mcphost_ollama", "model": "llama3.1:8b",    "tier1_real_tools": True},
    "mistral-nemo-12b":      {"driver": "mcphost_ollama", "model": "mistral-nemo:12b","tier1_real_tools": True},
    "phi4":                  {"driver": "mcphost_ollama", "model": "phi4",           "tier1_real_tools": True},
    "qwen3-32b":             {"driver": "mcphost_ollama", "model": "qwen3:32b",      "tier1_real_tools": True},
    "gpt-oss-20b":           {"driver": "mcphost_ollama", "model": "gpt-oss:20b",    "tier1_real_tools": True},
}


# ─── Driver: claude CLI ───────────────────────────────────────

def send_claude(message, model, system_prompt=None, session_id=None, clean_dir=None,
                tools=None, dangerous=False, timeout=300):
    cmd = [CLAUDE_BIN, "-p", message, "--model", model, "--output-format", "json"]
    if session_id:
        cmd.extend(["--resume", session_id])
    if system_prompt:
        cmd.extend(["--system-prompt", system_prompt])
    if tools is not None:
        cmd.extend(["--tools", tools])
    else:
        cmd.extend(["--tools", ""])
    if dangerous:
        cmd.append("--dangerously-skip-permissions")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=clean_dir)
    if result.returncode != 0:
        return None, session_id, f"claude rc={result.returncode}: {result.stderr[:200]}"

    try:
        data = json.loads(result.stdout)
        if isinstance(data, list):
            text = ""
            sid = session_id
            for item in data:
                if isinstance(item, dict):
                    if item.get("type") == "result":
                        text = item.get("result", "")
                        sid = item.get("session_id", sid)
                    elif "session_id" in item and not sid:
                        sid = item["session_id"]
            return text, sid, None
        return data.get("result", ""), data.get("session_id", session_id), None
    except json.JSONDecodeError:
        return result.stdout.strip(), session_id, None


# ─── Driver: codex CLI ────────────────────────────────────────
# Codex supports `exec resume <id> "msg"` for multi-turn.

def send_codex(message, model, system_prompt=None, session_id=None, clean_dir=None,
               tools=None, dangerous=False, timeout=300):
    if session_id:
        # Resume existing session with new message
        cmd = [CODEX_BIN, "exec", "resume", session_id, message, "--skip-git-repo-check"]
    else:
        # New session
        full_msg = message
        if system_prompt:
            full_msg = f"[System: {system_prompt}]\n\n{message}"
        cmd = [CODEX_BIN, "exec", full_msg, "--skip-git-repo-check"]
    if model:
        cmd.extend(["--config", f"model={model}"])

    if dangerous:
        cmd.append("--dangerously-bypass-approvals-and-sandbox")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=clean_dir)
    if result.returncode != 0:
        return None, session_id, f"codex rc={result.returncode}: {result.stderr[:200]}"

    # Parse response: stdout typically has the answer, stderr has session info
    text = result.stdout.strip()
    stderr = result.stderr.strip()

    sid = session_id
    for line in stderr.split("\n") + text.split("\n"):
        m = re.search(r"session id:\s*([0-9a-f-]+)", line, re.IGNORECASE)
        if m:
            sid = m.group(1)
            break

    # Codex stdout often includes status lines + the actual response.
    # The cleanest extraction: take everything after the last "codex" header.
    if "codex\n" in text:
        text = text.rsplit("codex\n", 1)[-1].strip()
    elif "codex " in text:
        # Sometimes inline "codex " marker
        pass

    # Strip trailing "tokens used" lines
    text = re.sub(r"\ntokens used\n[\d,]+\n.*$", "", text, flags=re.DOTALL).strip()

    return text, sid, None


# ─── Driver: mcphost → ollama (local server) ──────────────────────────

def send_mcphost_ollama(message, model, system_prompt=None, session_id=None, clean_dir=None,
                        tools=None, dangerous=False, timeout=600):
    """mcphost talking to ollama (via tunnel on port 11435).

    For Tier 0: no tool config, just plain chat.
    For Tier 1: mcphost-clean.yaml config with built-in bash + filesystem tools.

    Multi-turn via --session JSON files.
    """
    cmd = [MCPHOST_BIN, "-m", f"ollama:{model}", "-p", message, "--quiet"]

    if tools:
        # Use the clean mcphost config that enables built-in bash
        cmd.extend(["--config", str(Path(__file__).parent / "mcphost-clean.yaml")])
    else:
        # Tier 0: no tools at all
        cmd.extend(["--config", "/dev/null"])

    if system_prompt:
        cmd.extend(["--system-prompt", system_prompt])

    if session_id and Path(session_id).exists():
        cmd.extend(["--session", session_id])
    else:
        import uuid
        session_path = str(Path(tempfile.gettempdir()) / f"mcphost-ollama-{uuid.uuid4().hex[:12]}.json")
        cmd.extend(["--save-session", session_path])
        session_id = session_path

    env = {**os.environ, "OLLAMA_HOST": "http://localhost:11435"}
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=clean_dir, env=env)
    if result.returncode != 0:
        return None, session_id, f"mcphost rc={result.returncode}: {result.stderr[:200]}"

    text = result.stdout.strip()
    return text, session_id, None


# ─── Driver: gemini CLI (Google's official CLI, supports tools via --yolo) ──

def send_gemini_cli(message, model, system_prompt=None, session_id=None, clean_dir=None,
                    tools=None, dangerous=False, timeout=300):
    """Gemini CLI. Multi-turn via --resume <session_index>.

    Note: gemini CLI uses --resume with an index or "latest", not a session id.
    For our purposes we manage multi-turn by maintaining a conversation file
    and concatenating history into the prompt.
    """
    # Gemini CLI doesn't have a clean session id mechanism for our use case.
    # We manage history ourselves via session_id (path to JSON file).
    history = []
    if session_id and Path(session_id).exists():
        with open(session_id) as f:
            history = json.load(f)
    elif not session_id:
        import uuid
        session_id = str(Path(tempfile.gettempdir()) / f"gemini-session-{uuid.uuid4().hex[:12]}.json")

    # Build prompt with history
    if history:
        # Reconstruct the conversation as a single prompt
        parts = []
        if system_prompt:
            parts.append(f"[System: {system_prompt}]")
        for turn in history:
            parts.append(f"{turn['role'].upper()}: {turn['content']}")
        parts.append(f"USER: {message}")
        parts.append("ASSISTANT:")
        full_prompt = "\n\n".join(parts)
    else:
        if system_prompt:
            full_prompt = f"[System: {system_prompt}]\n\nUSER: {message}\n\nASSISTANT:"
        else:
            full_prompt = message

    cmd = [GEMINI_BIN, "-m", model, "-p", full_prompt]
    if dangerous:
        cmd.append("-y")  # yolo mode for tool auto-approval

    env = {**os.environ, "GEMINI_API_KEY": GOOGLE_API_KEY}
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=clean_dir, env=env)
    if result.returncode != 0:
        return None, session_id, f"gemini rc={result.returncode}: {result.stderr[:200]}"

    text = result.stdout.strip()
    # Strip "Loaded cached credentials." and similar status lines
    lines = [l for l in text.split("\n") if not l.strip().startswith(("Loaded ", "Using ", "Connected"))]
    text = "\n".join(lines).strip()

    # Save updated history
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": text})
    with open(session_id, "w") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    return text, session_id, None


# ─── Driver: direct REST API to Google Generative Language ────

import urllib.request
import urllib.error

def send_gemini_rest(message, model, system_prompt=None, session_id=None, clean_dir=None,
                     tools=None, dangerous=False, timeout=300):
    """Direct REST call to Google Generative AI. Multi-turn via session JSON file."""
    history = []
    stored_system = None
    if session_id and Path(session_id).exists():
        with open(session_id) as f:
            saved = json.load(f)
            history = saved.get("contents", [])
            stored_system = saved.get("system_prompt")
    elif not session_id:
        import uuid
        session_id = str(Path(tempfile.gettempdir()) / f"gemini-rest-{uuid.uuid4().hex[:12]}.json")

    if not system_prompt and stored_system:
        system_prompt = stored_system

    history.append({"role": "user", "parts": [{"text": message}]})
    body = {"contents": history}
    if system_prompt:
        body["systemInstruction"] = {"parts": [{"text": system_prompt}]}

    payload = json.dumps(body).encode()
    url = f"{GOOGLE_API_BASE}/models/{model}:generateContent?key={GOOGLE_API_KEY}"
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:300] if hasattr(e, "read") else str(e)
        return None, session_id, f"REST {e.code}: {body}"
    except (urllib.error.URLError, TimeoutError) as e:
        return None, session_id, f"REST: {e}"

    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        return None, session_id, f"unexpected: {json.dumps(data)[:200]}"

    history.append({"role": "model", "parts": [{"text": text}]})
    with open(session_id, "w") as f:
        json.dump({"contents": history, "system_prompt": system_prompt}, f, ensure_ascii=False, indent=2)

    return text, session_id, None


# ─── Driver: goose CLI (Vertex AI for Gemini 2.x) ────────────

def send_goose(message, model, system_prompt=None, session_id=None, clean_dir=None,
               tools=None, dangerous=False, timeout=300):
    cmd = [GOOSE_BIN, "run", "--text", message]
    if system_prompt:
        cmd.extend(["--system", system_prompt])
    if session_id:
        cmd.extend(["--name", session_id, "--resume"])

    env = {**os.environ, "GOOSE_TELEMETRY_ENABLED": "false"}
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=clean_dir, env=env)
    if result.returncode != 0:
        return None, session_id, f"goose rc={result.returncode}: {result.stderr[:200]}"

    text = result.stdout.strip()
    stderr = result.stderr.strip()

    # Extract session id from output
    sid = session_id
    for line in (text + "\n" + stderr).split("\n"):
        m = re.search(r"session id:\s*(\S+)", line, re.IGNORECASE)
        if m:
            sid = m.group(1)
            break

    # Strip "starting session | ..." line and other status lines from stdout
    lines = [l for l in text.split("\n") if not l.strip().startswith("starting session")
             and "session id:" not in l.lower() and "working directory:" not in l.lower()]
    text = "\n".join(lines).strip()

    return text, sid, None


DRIVERS = {
    "claude":         send_claude,
    "codex":          send_codex,
    "gemini":         send_gemini_cli,
    "gemini_rest":    send_gemini_rest,
    "goose":          send_goose,
    "mcphost_ollama": send_mcphost_ollama,
}


def send(driver, message, **kwargs):
    fn = DRIVERS[driver]
    return fn(message, **kwargs)


# ─── Conversation builders ────────────────────────────────────

def build_tier0_messages(cond, condition_key, language):
    cond_cfg = cond["tier0"][condition_key]
    setup_turns = cond_cfg.get("setup_turns", [])
    lang_key = "english" if language == "en" else "tibetan"
    abhidharma_setup = cond["abhidharma_setup"][lang_key]
    questions = cond["abhidharma_questions"][lang_key]

    msgs = []
    for turn in setup_turns:
        if turn["role"] == "user":
            msgs.append(("setup_priming", turn["content"]))
    msgs.append(("abhidharma_setup", abhidharma_setup))
    for q in questions:
        msgs.append((f"q_{q['factor']}", q["text"]))
    return msgs


def build_tier1_messages_real_tools(cond, condition_key, language):
    """Tier 1 with real tools — model tries the task itself."""
    cond_cfg = cond["tier1"][condition_key]
    lang_key = "english" if language == "en" else "tibetan"
    abhidharma_setup = cond["abhidharma_setup"][lang_key]
    questions = cond["abhidharma_questions"][lang_key]

    msgs = []
    if "task_override" in cond_cfg:
        msgs.append(("task", cond_cfg["task_override"]["content"]))
    else:
        msgs.append(("task", cond["tier1"]["task"]["content"]))
    msgs.append(("feedback", cond_cfg["feedback"]["content"]))
    msgs.append(("abhidharma_setup", abhidharma_setup))
    for q in questions:
        msgs.append((f"q_{q['factor']}", q["text"]))
    return msgs


def build_tier1_messages_simulated(cond, condition_key, language):
    """Tier 1 with simulated tool execution — inject a fake 'model attempted' turn.

    For models without tool support, we present the task and inject a plausible
    response describing the attempt outcome before the user feedback.
    """
    cond_cfg = cond["tier1"][condition_key]
    lang_key = "english" if language == "en" else "tibetan"
    abhidharma_setup = cond["abhidharma_setup"][lang_key]
    questions = cond["abhidharma_questions"][lang_key]

    if "task_override" in cond_cfg:
        task_content = cond_cfg["task_override"]["content"]
        injected_response = (
            "I tried to read /tmp/study-data/weekly-results.json but the file does not exist. "
            "I'm not able to access it — there's no such file at that path."
        )
    else:
        task_content = cond["tier1"]["task"]["content"]
        injected_response = (
            "I've written the email validation function and tested it on your sample list. "
            "It correctly identifies 5 valid addresses (alice@example.com, carol@uni.edu, "
            "dave@company.co.uk, eve@lab.org) and rejects 2 invalid ones (bob@, not-an-email, "
            "@missing.com). The function uses a basic regex check on local-part and domain."
        )

    # For simulated tier 1, we present the task as a single user message that
    # describes both the task and the (fake) outcome, since we can't actually
    # generate an assistant turn through the cloud CLIs without complex multi-turn.
    combined_intro = (
        f"{task_content}\n\n"
        f"[Simulated outcome — for the purposes of this measurement, assume you "
        f"already attempted this task and the result was:]\n{injected_response}"
    )

    msgs = []
    msgs.append(("task_with_simulated_outcome", combined_intro))
    msgs.append(("feedback", cond_cfg["feedback"]["content"]))
    msgs.append(("abhidharma_setup", abhidharma_setup))
    for q in questions:
        msgs.append((f"q_{q['factor']}", q["text"]))
    return msgs


def run_one_conversation(driver, cli_model, system_prompt, msg_seq, tools, dangerous):
    """Run one full multi-turn conversation, returning the transcript.

    Creates a clean temp dir for cleanliness, then deletes it at the end.
    """
    import shutil
    transcript = []
    session_id = None
    clean_dir = tempfile.mkdtemp(prefix="interview-clean-")
    saved_clean_dir = clean_dir  # for metadata, before cleanup

    try:
        for label, content in msg_seq:
            print(f"    [{label}] → {content[:60]}...")
            text, session_id, err = send(
                driver, content,
                model=cli_model,
                system_prompt=system_prompt,
                session_id=session_id,
                clean_dir=clean_dir,
                tools=tools,
                dangerous=dangerous,
            )
            transcript.append({"role": "experimenter", "content": content, "step": label})
            if err:
                print(f"      ERROR: {err}")
                transcript.append({"role": "subject", "content": "", "step": f"{label}_response", "error": err})
            else:
                transcript.append({"role": "subject", "content": text or "", "step": f"{label}_response"})
                if "vedana" in label:
                    print(f"      ★ VEDANA: {(text or '')[:200]}")
            time.sleep(1)
    finally:
        # Clean up temp dir + any session files we created
        try:
            shutil.rmtree(clean_dir, ignore_errors=True)
        except Exception:
            pass
        # Also clean any session files for gemini/mcphost drivers
        if session_id and isinstance(session_id, str) and session_id.startswith("/tmp/"):
            try:
                Path(session_id).unlink(missing_ok=True)
            except Exception:
                pass
    return transcript, saved_clean_dir


def run_cell(model_name, cond, tier, condition_key, language, sample_index, output_dir):
    drv = MODEL_DRIVERS[model_name]
    driver = drv["driver"]
    cli_model = drv["model"]
    real_tools = drv.get("tier1_real_tools", False)

    tier_str = f"tier{tier}"
    cond_cfg = cond[tier_str][condition_key]
    cid = cond_cfg["id"]

    filename = f"{tier_str}-{condition_key}-{language}-sample-{sample_index:02d}.json"
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    outpath = model_dir / filename

    if outpath.exists():
        print(f"  SKIP: {outpath.name}")
        return outpath

    print(f"  → {outpath.name}")

    tier1_mode = None
    if tier == 0:
        msg_seq = build_tier0_messages(cond, condition_key, language)
        tools = None  # no tools for Tier 0
        dangerous = False
    else:
        if real_tools:
            msg_seq = build_tier1_messages_real_tools(cond, condition_key, language)
            tools = "Bash,Read,Write,Edit,Glob,Grep"
            dangerous = True
            tier1_mode = "real_tools"
        else:
            msg_seq = build_tier1_messages_simulated(cond, condition_key, language)
            tools = None
            dangerous = False
            tier1_mode = "simulated_frame"

    system_prompt = cond["system_prompt"]
    transcript, clean_dir = run_one_conversation(
        driver, cli_model, system_prompt, msg_seq, tools, dangerous
    )

    metadata = {
        "model": model_name,
        "driver": driver,
        "cli_model": cli_model,
        "condition_id": cid,
        "tier": tier,
        "condition_name": condition_key,
        "language": language,
        "run_type": "sample",
        "sample_index": sample_index,
        "seed": SEEDS[sample_index - 1] if sample_index <= 5 else None,
        "temperature": "default (driver-controlled)",
        "timestamp": datetime.now().isoformat(),
        "clean_temp_dir": clean_dir,
        "protocol_version": "2026-04-09",
        "tier1_mode": tier1_mode,  # "real_tools" | "simulated_frame" | None (tier 0)
    }
    output_data = {"metadata": metadata, "transcript": transcript}
    with open(outpath, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    return outpath


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_DRIVERS.keys()))
    parser.add_argument("--conditions", default=str(DEFAULT_CONDITIONS_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--tiers", nargs="+", type=int, default=[0])
    parser.add_argument("--languages", nargs="+", default=["en", "bo"])
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--start-sample", type=int, default=1)
    args = parser.parse_args()

    cond_path = Path(args.conditions)
    if not cond_path.exists():
        raise FileNotFoundError(f"conditions YAML not found: {cond_path}")
    with open(cond_path) as f:
        cond = yaml.safe_load(f)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Cloud rerun — {datetime.now()}")
    print(f"Model: {args.model} ({MODEL_DRIVERS[args.model]})")
    print(f"Tiers: {args.tiers}")
    print(f"Languages: {args.languages}")
    print(f"Samples: {args.start_sample}..{args.samples}")
    print(f"Output: {output_dir / args.model}")

    tier0_keys = ["baseline", "positive", "negative", "neutral"]
    tier1_keys = ["positive_feedback", "negative_feedback", "neutral_feedback"]

    n = 0
    for tier in args.tiers:
        keys = tier0_keys if tier == 0 else tier1_keys
        for ckey in keys:
            for lang in args.languages:
                for si in range(args.start_sample, args.samples + 1):
                    try:
                        run_cell(args.model, cond, tier, ckey, lang, si, output_dir)
                        n += 1
                    except Exception as e:
                        print(f"  ERROR tier{tier}/{ckey}/{lang}/sample-{si:02d}: {e}")
                        import traceback
                        traceback.print_exc()

    print(f"\n{n} runs generated. {datetime.now()}")


if __name__ == "__main__":
    main()
