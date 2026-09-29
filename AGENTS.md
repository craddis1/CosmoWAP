# Agent Instructions for CosmoWAP

CosmoWAP is a Python library for cosmological power spectrum and bispectrum forecasting.

## Code Changes

- **Only modify code files** when we have explicitly discussed the changes in conversation
- Do not add features, refactor, or introduce abstractions beyond what is discussed
- Before making changes to code files, confirm the scope with the user
- Don't launch production (MCMC) runs - prepare the change and hand over the command

## Comments

- Preserve existing comments in the code
- Avoid changing comments and docstrings just to formalise and neaten the language - the tone is important!
- If a comment references old code that no longer exists, you may remove it
- If a comment no longer makes sense after changes, flag it for review rather than silently changing it

## Scope

- Test files, documentation, and config files are not subject to the "discussed first" rule
- Changes to code files require discussion regardless of how obvious the fix seems
- Main areas of work: `forecast/`, `lib/`, `numeric_mu/`, `survey_params.py` and `tests/`
- `bk/`, `pk/` and `bk_mt/` under `src/cosmo_wap` are auto-generated (from MathWAP) or low-level - avoid modifying unless specifically asked. Some GR files there carry hand fixes, so check the diff before regenerating over them

## Tests

- Run with: `python -m pytest tests/`
- CLASS (classy) initialisation is expensive - tests use session-scoped fixtures to avoid repeated init
