# Claude Instructions for CosmoWAP

## Code Changes

- **Only modify code files** when we have explicitly discussed the changes in conversation
- Do not add features, refactor, or introduce abstractions beyond what is discussed
- Before making changes to code files, confirm the scope with the user

## Comments

- Preserve existing comments in the code
- If a comment references old code that no longer exists, you may remove it
- If a comment no longer makes sense after changes, flag it for review rather than silently changing it

## Scope

- Test files, documentation, and config files are not subject to the "discussed first" rule
- Changes to code files require discussion regardless of how obvious the fix seems
