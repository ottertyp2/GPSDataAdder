# AGENTS.md

## CRITICAL delivery rule

Do not treat "code changed locally" as finished work in this repository.

- If a task is ready to share, carry it through `tests -> git status review -> commit -> push`.
- The push must go through `origin` so the same branch/commit reaches both GitHub and GitLab.
- The default delivery branch is `main`; do not create, switch to, or push a separate branch unless the user explicitly asks for one.
- Never silently stop after local edits or tests while claiming the work is delivered.
- If you intentionally do not commit or do not push, say that explicitly and explain why.

## Mandatory startup and finish checklist

Every agent instance working in this repository must do these checks explicitly:

- Read this `AGENTS.md` before making changes.
- Confirm you are on `main`. If not, return to `main` before editing unless the user explicitly requested another branch.
- Before finishing a shareable task, review `git status` and `git remote -v`.
- If the task is ready to share, commit only the intended files.
- Push `main` through `origin` so both GitHub and GitLab receive the exact same commit.
- If the worktree is dirty with unrelated changes, do not silently skip sync; report the situation clearly.
- If one remote push fails, say which host failed and which host succeeded.

## Communication style

- Keep progress updates and final summaries concise.
- Lead with what changed, what was tested, and whether it was pushed.
- Avoid long explanations unless the user asks for detail.

## Purpose

This repository hosts an offline GPS L1 C/A data adder GUI for decoder testing. It reads local little-endian `complex64` IQ recordings, synthesizes one plausible GPS satellite channel, and writes a new offline IQ recording with that synthetic channel added.

Agents working here should preserve the project's core priorities:

- offline-only processing of local files
- no SDR, transmitter, RF, or live-stream control
- readable GPS L1 C/A DSP implementation
- responsive PySide6 GUI for large local recordings
- deterministic synthetic PRN, navigation-bit, carrier, and metadata generation
- windowed/chunked processing so large recordings do not need to fit into RAM
- compatibility with the Fraunhofer_FHR decoder baseline format: little-endian `complex64`
- small, focused tests for C/A code generation, LNAV parity, and file augmentation

## Project structure

- `app/main.py`: GUI entry point and optional command-line mode
- `app/gui/`: main window and worker thread plumbing
- `app/dsp/`: PRN generation, LNAV bit generation, synthetic channel generation, and file mixing
- `app/tests/`: unit and smoke tests
- `tools/check_git_sync.ps1`: quick check for branch, status, and dual-push remote setup

## Management rules for agents

- Keep the GUI runnable with `python -m app.main`.
- Keep the CLI fallback runnable with `python -m app.main --cli`.
- Keep signal-processing logic out of the command-line parser.
- Treat `complex64` little-endian IQ as the baseline input and output format.
- Do not commit local recordings, generated `.bin`/`.dat`/`.iq` files, or metadata sidecars from test runs.
- Preserve deterministic defaults so decoder tests can be repeated exactly.
- Add or update tests for DSP behavior whenever practical.

## Git and release hygiene

- Keep `README.md`, `requirements.txt`, and this file aligned with major feature changes.
- Prefer small, reviewable commits with clear messages.
- Treat GitHub and GitLab as first-class remotes for this project and keep them in sync.
- Prefer keeping `origin` configured with both push URLs so one clean `git push origin main` updates GitHub and GitLab together.
- Do not create feature, task, or `codex/` branches unless the user explicitly requests branch-based work.
- Do not push partial, broken, or unrelated work just to satisfy the sync rule; stage and commit only the files that belong to the task being delivered.
- If one remote push fails, report it clearly so the repository state can be corrected instead of assuming both hosts were updated.
- Use `powershell -ExecutionPolicy Bypass -File tools/check_git_sync.ps1` as a quick pre-push verification when needed.

## Large-file policy

Recorded IQ captures and generated outputs are local analysis assets and should stay out of Git by default.
