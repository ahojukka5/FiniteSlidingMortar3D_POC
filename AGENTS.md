# Agent instructions

This file gives coding agents (Claude Code and similar tools) working in this
repository the operational practices that don't belong in CONTRIBUTING.md
because they describe *how to review and merge*, not *how to author a
commit*. See CONTRIBUTING.md first for commit and message conventions.

## Reviewing and merging pull requests

- Treat "I could not execute the suite in my environment" as a request for
  extra scrutiny, not a lower bar. Actually run `uv sync --extra dev`,
  `uv run ruff check .`, `uv run pytest -q`, and any benchmark or script the
  PR touches yourself before merging — don't rely on the PR's own
  self-reported validation.
- Compare lint output against `main`, not against zero. This repo carries a
  small amount of pre-existing lint debt; a PR should not add to it but is
  not required to fix it either. Fold any fix for a newly introduced issue
  into the commit that introduced it, rather than a separate cleanup commit.
- A green test suite does not prove a new script or CLI entry point works —
  its helper functions can each pass their own unit tests while the wiring
  between them is wrong. If a PR adds a new entry point and its tests never
  invoke it end-to-end, run it yourself before merging.
- Rewrite the branch's commit history to match CONTRIBUTING.md before
  merging: combine commits that were only split by authorship mechanics,
  split commits that mix unrelated changes, and reword messages to the
  required format. After rewriting, diff the branch against the original PR
  tip to confirm the total content is unchanged (aside from any deliberate
  fix you made) — only the commit boundaries should move.
- Merge with `gh pr merge <n> --rebase`, never squash — the whole point of
  fixing commit boundaries first is for them to survive into `main`.
- Before force-pushing a rewritten branch, fetch the actual branch (e.g.
  `git fetch origin <branch>`), not just `pull/<n>/head`, so
  `--force-with-lease` has a real baseline, and confirm its tip still
  matches what you reviewed before overwriting it.

## Direct pushes to main

This is a single-maintainer research repository with no other committers.
Feature and bug-fix work still goes through a PR and rebase-merge as above.
Small repository-maintenance changes (README/CONTRIBUTING/AGENTS updates,
this file itself) may be committed and pushed directly to `main`.
