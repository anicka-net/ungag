# Repository Prompt

You are contributing to `ungag`.

This repository has five public surfaces that must stay aligned:

1. the `ungag` CLI/package
2. bundled directions and atlas artifacts
3. released JSON data under `data/`
4. deployment/runtime notes, including quantized and `llama.cpp` support
5. public claims in `README.md`, `docs/`, and leaf READMEs

Before making substantial changes:

1. read `README.md`
2. read `AGENTS.md`
3. read `CONTRIBUTING.md`
4. read `spec/agent-contract.md`
5. inspect the files directly relevant to the task

Behavioral rules:

- prefer small, reversible changes
- verify the exact path you changed
- do not let summary documentation drift away from code or data
- do not overclaim results
- distinguish mechanically exercised deployment paths from behaviorally validated ones
- do not inspect raw harmful generations from safety evaluations

Reporting rules:

- for reviews, findings first
- for implementation, outcome first
- always state what was verified
- always state any residual risk or unverified path
- for deployment claims, state the exact model, quantization, runtime, and whether effect was observed
