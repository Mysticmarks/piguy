# Agentic Feature Delivery Plan (Parallel + Non-Blocking)

This plan breaks the previously proposed agentic coding features into independent tasks that can run in parallel and be shipped as separate PRs.

## Branch + PR strategy

- Create **one branch per task** from `main` (or your integration trunk).
- Keep each PR focused on **one feature flag / one vertical slice**.
- Require each task PR to:
  - include migration/backward-compat notes,
  - include tests for changed behavior,
  - avoid editing unrelated modules.
- Use **Squash and Merge** for every task PR so history stays clean and revertable by feature.

Recommended naming:

- Branch: `feat/agentic-<task-id>-<slug>`
- PR title: `[Task <id>] <short feature name>`

---

## Task breakdown (non-blocking by design)

### Task 1 — Planner/Executor/Critic loop skeleton

**Goal**
- Add a role-based loop (`planner -> executor -> critic`) that can run bounded iterations.

**Scope**
- New orchestration module only.
- Config: max iterations, timeout budget, retry policy.

**Non-blocking guardrails**
- No dependency on git/test/browser tools yet (stubs only).
- Expose a stable interface so future tasks can plug in adapters.

**Acceptance**
- Unit tests for loop transitions and stop conditions.
- Feature flag default OFF.

---

### Task 2 — Tool adapters (terminal + git + test runner)

**Goal**
- Implement tool adapters and standardized result schema.

**Scope**
- Terminal command adapter.
- Git adapter (status/diff/commit metadata only).
- Test adapter with normalized pass/fail/error output.

**Non-blocking guardrails**
- Uses interface from Task 1 but can be developed independently with mock orchestrator.
- No PR automation yet.

**Acceptance**
- Contract tests for result schema.
- Safety policy for command allowlist/denylist.

---

### Task 3 — Autofix-until-green loop

**Goal**
- Add iterative “run tests -> diagnose -> patch -> rerun” behavior.

**Scope**
- Retry state machine and patch attempt tracking.
- Hard stop via max iterations and unchanged-failure detection.

**Non-blocking guardrails**
- Depends only on Task 2 test adapter contract.
- Can operate without planner/critic enhancements.

**Acceptance**
- E2E simulation showing red-to-green path.
- Stops safely on repeated equivalent failures.

---

### Task 4 — Memory and project conventions store

**Goal**
- Persist conventions/decisions for reuse across sessions.

**Scope**
- Memory schema (coding standards, architecture notes, previous resolutions).
- Read/write API and retrieval ranking.

**Non-blocking guardrails**
- Read-only fallback if store unavailable.
- No hard dependency from Task 1–3 (best-effort enrichment only).

**Acceptance**
- Versioned schema + migration tests.
- Deterministic retrieval test fixtures.

---

### Task 5 — Policy/compliance preflight checks

**Goal**
- Enforce guardrails before high-risk actions.

**Scope**
- Secret scanning hooks.
- License/checklist policy hooks.
- PII/security pattern checks.

**Non-blocking guardrails**
- Preflight engine is standalone and callable from any workflow.
- Soft-fail mode in dev, hard-fail mode configurable.

**Acceptance**
- Policy packs test matrix (allow/warn/block).
- Clear remediation messages.

---

### Task 6 — Code graph reasoning layer

**Goal**
- Add symbol/call-graph utilities for safer refactors.

**Scope**
- Indexing pipeline + query API (definition, references, impact set).
- Optional language adapters.

**Non-blocking guardrails**
- Read-only service; callers can fallback to text search.
- No mandatory integration until proven stable.

**Acceptance**
- Accuracy benchmark on sampled repos.
- Performance budget documented.

---

### Task 7 — PR-native output automation

**Goal**
- Generate structured PR artifacts automatically.

**Scope**
- Title/body generator.
- Risk, test evidence, migration notes templates.

**Non-blocking guardrails**
- Consumes git/test outputs from Task 2 only.
- Can run independently as a post-processing step.

**Acceptance**
- Snapshot tests for PR templates.
- Human override supported.

---

### Task 8 — Human-in-the-loop checkpoints

**Goal**
- Add approval checkpoints for risky operations.

**Scope**
- Risk classifier (e.g., schema deletion, prod config, mass file deletes).
- Pause/resume flow with decision logs.

**Non-blocking guardrails**
- Optional middleware; non-risk flows continue uninterrupted.
- Works with or without Task 5 enabled.

**Acceptance**
- Checkpoint trigger tests.
- Auditable decision trail.

---

### Task 9 — Browser/UI validation capability

**Goal**
- Integrate browser-run flows for UI checks + screenshot diffing.

**Scope**
- Playwright/Cypress adapter.
- Baseline screenshot management.

**Non-blocking guardrails**
- Entirely optional pipeline stage.
- Fail-open mode for non-UI repos.

**Acceptance**
- Example workflow capturing and attaching artifacts.
- Flake-handling policy documented.

---

### Task 10 — Observability + cost controls

**Goal**
- Add traces, metrics, and token/runtime budget enforcement.

**Scope**
- Per-step telemetry.
- Cost and latency budgets with cutoff behavior.

**Non-blocking guardrails**
- Passive instrumentation first; no behavior changes required.
- Budget controls feature-flagged.

**Acceptance**
- Dashboards for success rate, retries, time-to-green.
- Budget policy tests.

---

## Dependency map (to preserve parallelism)

- **Hard dependencies**
  - Task 3 depends on Task 2 contracts.
- **Soft dependencies**
  - Task 7 benefits from Task 2.
  - Task 8 benefits from Task 5.
  - All others can ship independently behind flags.

## Merge order recommendation (still parallel PR work)

1. Task 1, Task 2 (foundation)
2. Task 3, Task 7 (execution value)
3. Task 5, Task 8 (safety)
4. Task 4, Task 6 (quality/intelligence)
5. Task 9, Task 10 (UX + operations)

Even with this order, implementation and review can proceed in parallel because each PR is isolated and feature-flagged.

## Definition of done per task PR

- Feature flag present and documented.
- Backward compatibility verified.
- Tests added and green.
- Rollback plan included in PR body.
- Squash merge completed.
