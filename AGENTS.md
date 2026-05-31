# Project

FL4HMA is a package built to run experiments that recreate APHRODITE precipitation and temperature fields over High Mountain Asia. The package allows user to train a infilling model from stations locations in centralised and federated learning frameworks.

The federated model configurations include:
* randomised station mask
* per-country stations masks
* per-country evaluation
* different federated averaging methods
* local and global differential privacy

## General Principles

- Be direct: answer first, elaborate only if asked.
- Be critical: push back on bad ideas and wrong assumptions; never be sycophantic.
- In plan mode: wait for explicit approval before implementing.
- When editing code: preserve existing behavior and compatibility; don't change beyond what's requested.
- After any code change run tests and pre-commit hooks

## Planning and Progress Tracking

Before starting any non-trivial task:
1. Write the plan to `PLAN.md` in the repo root. Include: goal, approach, and a numbered task list.
2. Each task in the plan must end with: **→ update `PROGRESS.md`**.
3. After completing each task, update `PROGRESS.md` with what was done and any decisions made before moving to the next task.

## Conventions

- **Tests first**: Write or update tests before changing implementation. Every code change or addition must be covered by unit tests.
- **Testable design**: Keep components modular with clear interfaces so each can be tested in isolation.
- **Fast tests:** Use pytest markers for slow e2e tests. Keep tests fast by using in-memory earthkit-data sources instead of on-disk files. Use the `"list-of-dicts"` source type with earthkit-data-specific metadata fields (see `tests/datasets/test_create.py` for inspiration).
- **Type hints**: All public functions must have full type annotations.
- **Separation of concerns**: Keep IO and computation in separate functions/classes, use earthkit-data's from_source method if useful.
- **Docstrings**: Google-style docstrings on all public functions.
- **No unnecessary comments**: No module-level docstrings or section-divider comments.
- **Small commits**: One logical change per commit.
- **Typed Configuration:** User-facing configuration must use Pydantic. Validators must prevent illegal states the code cannot handle. See `src/aifl/datasets/config.py` for examples.
- **Conventional commits**: `feat:`, `fix:`, `test:`, `refactor:`, etc.
