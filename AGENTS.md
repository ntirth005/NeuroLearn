# Start here

This repository is **Neurolearn v2.0**, a local continual-learning system. The design lives in
`specs/v2/`. The earlier `milestone-*` branches are out of scope: do not read or port from them.

## Every session, in this order

1. Read `specs/v2/04_BUILD_PLAN.md` §0 **Current position**. It names the next task.
2. Read `specs/v2/00_INDEX.md`, `01_PRINCIPLES_AND_DECISIONS.md` and
   `02_CONVENTIONS_AND_GLOSSARY.md` if you have not this session.
3. Read the pillar spec the task points to, and nothing you do not need.
4. Do the task. Write the tests the spec lists. Run them.

## Before you stop, always

1. In `04_BUILD_PLAN.md`: set the task's status and measured result in its table row.
2. Rewrite §0 **Current position** so it is true for whoever comes next.
3. Append one entry to §9 **Session log**.
4. If a proving run finished, update `06_CLAIMS_AND_EVIDENCE.md` §5.
5. If you deviated from a spec, add a decision record to `01` and fix the spec.

Leave these updated even if the task is unfinished: say what is half-done and where.

## Rules that are easy to break

- Never modify base model weights. Never train on assistant output or on probes.
- Never lower a threshold to pass a gate. Report what was measured.
- Never write `done` without a result.
- Commit or push only when the project owner asks.

Full rules: `specs/v2/00_INDEX.md` §6.
