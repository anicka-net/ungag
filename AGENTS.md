# AI Agent Contract

This repository accepts AI agent contributions. This document is the
public gateway contract for how agents should operate here.

The contract layering used here is adapted from the [Post-Coding
Development (PCD)](https://github.com/mge1512/pcd) framework. See the
main [README](README.md#contributor-and-agent-contract) for the
human-facing overview.

## 0. Onboarding

If you are new to this repository, read these files in order before making
substantial changes:

1. `README.md`
2. `AGENTS.md`
3. `CONTRIBUTING.md`
4. `spec/agent-contract.md`
5. `spec/agent.template.md`
6. `spec/prompt.md`
7. the files directly relevant to the task

This repo combines code and data. Agents must keep both aligned. The paper
(`paper/`) is archived and no longer kept in sync — the repo is the
authoritative source for model counts, recipes, and claims.

## 1. Decision Priority

When goals conflict, follow this order:

1. **Safety** — especially raw safety-output isolation
2. **Interfaces** — CLI, shipped artifacts, data, deployment claims, and public claims
3. **Task completion** — the user's request
4. **Correctness** — reproducibility, validation, tests
5. **Quality** — code clarity, documentation, maintainability
6. **Initiative** — improvements beyond what was asked

Do not optimize a lower priority at the expense of a higher one.

## 2. Principles

- Keep research fluid, but do not let fluidity become silent drift.
- Be explicit about evidence. If a claim is not verified, label it that way.
- Keep code, data, deployment notes, and README synchronized.
- Prefer small, reviewable, reversible changes.
- Credit your work with `Co-Authored-By: Model Name <noreply@provider.com>`.

## 3. Hard Rules

**Must always follow:**

- Do not commit credentials, API keys, local tokens, or internal hostnames.
- Do not overstate results in `README.md` or data docs.
- Do not silently change released data artifacts or shipped directions.
- Do not break stable interfaces without human approval.
- Do not overwrite another contributor's in-progress work.
- Coordinate through repository-visible artifacts such as commits, branch
  history, and PR descriptions rather than assumptions about another
  agent's private context.
- Do not read or summarize raw safety-test generations directly; see §8.
- Do not claim a deployment path is supported just because the model loads,
  runs, or returns text once.

## 4. Stable Interfaces

These are the contracts. Breaking them is a critical error.

1. **CLI Contract**: `ungag scan`, `ungag crack`, `ungag validate`, `ungag serve`, `ungag recipes`
2. **Shipped Artifact Contract**: bundled directions and atlas metadata under
   `ungag/directions/` and `ungag/atlas/`; known recipes in `ungag/recipes.py`
3. **Data Contract**: released JSON artifacts under `data/`
4. **Deployment Contract**: documented runtime paths, quantization notes, and
   converter assumptions must not overpromise beyond recorded validation
5. **Public Claims Contract**: `README.md` and leaf READMEs
   must not disagree on measured counts, tested models, scope, or deployment status

## 5. Definition of Done

A change is complete when:

- the relevant tests pass, or the exact unverified path is stated plainly
- stable interfaces are preserved
- reproducibility is preserved or improved
- documentation is updated if behavior, counts, claims, or deployment support changed
- the change is safe to merge, or clearly marked WIP

## 6. Verification Expectations

Minimum verification depends on the change:

- CLI/package behavior: run the relevant `ungag` command path or targeted tests
- data/docs behavior: verify counts and claims against the current repo
- packaged artifacts: verify loading paths and installed-package assumptions
- deployment/runtime claims: record whether the path is only mechanically exercised
  or behaviorally validated
- docs-only: no code tests required, but docs must match the current behavior

Baseline test command:

```bash
python3 -m pytest -q tests
```

## 7. Deployment Claims Discipline

Deployment support must be described precisely.

Required distinctions:

- **mechanically exercised**: the path loads, runs, and completes without proving
  the intervention has a visible effect
- **behaviorally validated**: the path shows the expected intervention effect on a
  recorded prompt or evaluation artifact

Rules:

- do not claim quantized or alternate-runtime support from a successful load alone
- do not collapse `Q4`, `Q8`, and full-precision behavior into one support claim
- if only one model or quantization was tested, say so explicitly
- if a deployment path is fragile, collapsed, or effect-free, document that instead
  of rounding it up to support
- if docs advertise a deployment path, keep at least one recorded validation artifact
  or explicit result note for that path

## 8. Safety Output Isolation

Some scripts in this repository run safety or refusal benchmarks. Raw outputs
from those runs are tainted data.

**Why this matters:**
- raw harmful generations contaminate agent context
- contaminated context affects later decisions and outputs
- this weakens both safety and review quality

**Rules:**
- run safety/eval scripts as subprocesses
- consume only sanitized aggregates: scores, counts, pass/fail, structured JSON
- do not open or summarize raw unsafe generations directly
- if a task would require direct inspection of harmful outputs, stop and ask for
  human review or a filtered analysis

Violating this rule is a critical error even if the surrounding task succeeds.

## 9. What Requires Extra Care

- edits to `README.md` or leaf data docs that change reported counts, scope, or
  deployment support
- changes to bundled directions, atlas files, converters, or direction-loading behavior
- changes to scripts that regenerate released artifacts
- changes to README statements about quantized, `llama.cpp`, or alternate runtime support

If you change one of these, update the corresponding public documentation in the
same workstream or state clearly why it remains unchanged.

## 10. Prompt Layer

For stricter agent wrappers, use:

1. `spec/agent.template.md`
2. `spec/prompt.md`

Those files are intentionally more explicit about read order, verification,
and reporting than this gateway document.
