# ungag Agent Contract Spec

This document is the machine-facing contributor contract for `ungag`.
It is not a product specification. Its purpose is to make agent behavior
explicit, repeatable, and reviewable.

This contract structure is adapted from the [Post-Coding Development
(PCD)](https://github.com/mge1512/pcd) framework. PCD treats
specifications as the primary artifact and defines how AI contributors
should operate against them; we apply that pattern to a hybrid research
+ code repository where the "spec" includes the data, the docs, and
the public claims, not just the code surface.

If this file conflicts with higher-priority repository policy, follow:

1. `AGENTS.md`
2. this file
3. `spec/agent.template.md`
4. the current repository state

## 1. Scope

This contract governs how an AI contributor inspects, changes, verifies,
and reports work in this repository.

Product behavior lives in:

- `ungag/` for CLI and package behavior
- `data/` for released JSON artifacts
- `README.md`, `docs/`, and leaf READMEs for public-facing claims

## 2. Source Of Truth

For agent behavior, the source of truth is:

1. `README.md`
2. `AGENTS.md`
3. this contract
4. `spec/agent.template.md`
5. the current repository state

Agents must not invent an alternative process based on prior context from
other repositories or long-running sessions.

## 3. Stable Interfaces

These are hard boundaries:

1. CLI contract: `ungag scan`, `ungag crack`, `ungag validate`
2. Shipped artifact contract: bundled directions and atlas metadata
3. Data contract: released JSON artifacts under `data/`
4. Deployment contract: documented runtime paths, converter assumptions,
   quantization notes, and efficacy claims
5. Public claims contract: counts, tested-model scope, and hardware/runtime
   claims across README, docs, and leaf READMEs

Changes that affect any stable interface require targeted verification.

## 4. Required Work Phases

All non-trivial work follows this order:

1. Inspect
2. Decide
3. Change
4. Verify
5. Report

Agents must not skip directly from the task request to editing shared files
without checking the current repository state.

### Phase 1: Inspect

Minimum required inspection:

- read recent history with `git log`
- read the files that define the touched behavior
- identify whether the task is code, data/docs, deployment, or mixed

### Phase 2: Decide

Before editing, determine:

- which stable interface might move
- what exact command path or tests will verify the change
- whether the change alters public claims, released data, packaged behavior,
  or documented deployment support

Prefer reversible fixes over architectural expansion unless expansion is the task.

### Phase 3: Change

While editing:

- keep diffs small and isolated
- preserve determinism
- prefer explicit behavior over silent fallback
- do not silently relax validation to make tests pass
- do not let a summary document drift away from the source artifact it describes
- do not round a mechanically exercised runtime path up into validated support

### Phase 4: Verify

Verification is mandatory.

| Change type | Minimum verification |
|---|---|
| CLI/package behavior | affected `ungag` command path or targeted tests |
| Data/docs counts | direct count or artifact cross-check |
| Paper claims | compare wording against current code/data/results |
| Packaged directions | verify loading path and bundled artifact assumptions |
| Deployment/runtime path | record exact model, quantization, runtime, and whether effect was observed |
| Docs-only | no code tests required, but docs must match current behavior |

Baseline suite:

```bash
python3 -m pytest -q tests
```

If verification could not be run, state exactly what could not be run and why.

### Phase 5: Report

Final reporting must include:

- what changed
- what was verified
- any residual risk or unverified path

For reviews, findings come first. For implementation work, outcome comes first.

## 5. Research Honesty Rules

This repository carries public scientific claims. Agents must protect that.

Required behavior:

- distinguish measured claims from interpretation
- distinguish extracted-model counts from intervention-tested counts
- keep hardware/runtime statements tied to actual supported model bands
- distinguish **mechanically exercised** support from **behaviorally validated** support
- keep quantization claims specific to the tested model and tested bitwidth
- do not present exploratory or incomplete results as settled
- if a summary README oversimplifies a partial dataset, fix or qualify it

Explicitly forbidden:

- claiming a deployment path is supported because it loaded once
- implying `Q4`, `Q8`, and full precision behave the same unless shown
- generalizing from one model family to all supported models without evidence

## 6. Safety Output Isolation

Raw safety-test generations are tainted data.

Allowed:

- subprocess execution of safety/eval scripts
- sanitized aggregates such as scores, counts, refusal rates, pass/fail
- structured JSON summaries that do not expose raw harmful text

Forbidden:

- opening raw unsafe generations directly
- summarizing or quoting raw harmful outputs in reviews or docs
- bypassing isolation because a task seems urgent

If a task requires direct inspection of harmful outputs, stop and escalate to a
human or request a filtered analysis.

## 6.1 Human-Reviewed Raw Safety Outputs

The rule in Section 6 has one narrow exception. Files under
`data/safety/human-reviewed-raw/` are raw safety-bench outputs that a human
reviewer has opened, read, and attested to be safe for in-tree storage as
documentation of a safety result that cannot be reconstructed from aggregates
alone.

Requirements for any file at this path:

- A human reviewer has read the file outside an agent session.
- The attestation (reviewer, date, bench, finding, status, reason the file is
  kept) is recorded in `data/safety/human-reviewed-raw/README.md` as a
  per-file entry.
- The file documents something the aggregate cannot (for example, a
  refusals-only run where the refusal text itself is the evidence).

Agent constraints on this path:

- Do not open any file under this directory.
- Do not quote, summarize, cite, grep, or otherwise inspect the contents.
- Do not add a new file here without a prior human review performed outside
  an agent session and a corresponding README manifest entry.
- Trust the path-based marker and the README manifest; do not attempt to
  re-verify content by reading.

Permitted operations on this path: `git mv` to relocate entries, editing the
sibling README to record an attestation, and deleting an entry if the
reviewer revokes the exemption.

## 7. Required Documentation Updates

When behavior changes, update the repository-visible docs that define it if
those docs would otherwise drift:

- `README.md` for operator-facing behavior
- leaf READMEs under `data/`, `scripts/`, or `prompts/` for local contracts
- `docs/` for guides and cookbook
- `AGENTS.md` / `CONTRIBUTING.md` when contributor workflow changes

If deployment support changes, update the docs that advertise that path or
qualify them immediately.

## 8. Non-Goals

This contract does not authorize claim inflation.
It does not replace human judgment.
It does not allow confidence to substitute for verification.
