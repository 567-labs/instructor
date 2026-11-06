---
title: Documentation Agent Guide
description: Internal guide for maintaining and improving Instructor documentation
---

# AGENT.md - Documentation

## Commands
- Serve docs locally: `uv run mkdocs serve`
- Build docs: `./build_mkdocs.sh` or `uv run mkdocs build`
- Install doc deps: `uv pip install -e ".[docs]"`
- Test examples: `uv run pytest docs/ --examples`

## Structure
- **Core docs**: `concepts/`, `integrations/`, `examples/`
- **Learning path**: `getting-started.md` → `learning/` → `tutorials/`
- **API reference**: Auto-generated from docstrings via `mkdocstrings`
- **Blog**: `blog/posts/` for announcements and deep-dives
- **Templates**: `templates/` for new docs (provider, concept, cookbook)

## Writing Guidelines
- **Reading level**: Grade 10 (from .cursor/rules)
- **Code examples**: Must be runnable with complete imports
- **Progressive complexity**: Simple → advanced concepts
- **Provider docs**: Follow `templates/` patterns
- **Navigation**: Update `mkdocs.yml` for new pages

## Pull Request (PR) Formatting

Use **Conventional Commits** formatting for PR titles so they are consistent and easy to scan. Treat the PR title as the message we would use for a squash merge commit.

### PR Title Format

Use:

`<type>(<scope>): <short summary>`

Rules:
- Keep it under ~70 characters when you can.
- Use the imperative mood (for example, “add”, “fix”, “update”).
- Do not end with a period.
- If it includes a breaking change, add `!` after the type or scope (for example, `feat(docs)!:`).

Good examples:
- `docs(agents): add conventional commit PR title guidelines`
- `docs(mkdocs): fix broken link in validation tutorial`
- `docs(examples): update youtube clips snippet`
- `chore(docs): refresh docs build commands`

Common types:
- `docs`: documentation-only changes
- `fix`: bug fix
- `feat`: new feature
- `test`: add or update tests
- `chore`: maintenance work (build scripts, tooling, repo hygiene)
- `ci`: CI pipeline changes

Suggested docs scopes:
- `docs`, `mkdocs`, `blog`, `examples`, `integrations`, `tutorials`, `agents`

### PR Description Guidelines

Keep PR descriptions short and actionable:
- **What**: What changed, in 1–3 sentences.
- **Why**: Why this change is needed (link issues when possible).
- **Changes**: 3–7 bullet points with the main edits.
- **Testing**: What you ran (or why you did not run anything).
- **Docs impact**: Call out page moves, redirects, or nav updates.

If the PR was authored by Cursor, include:
- `This PR was written by [Cursor](https://cursor.com)`

## Key Files
- `mkdocs.yml` - Site configuration and navigation
- `hooks/` - Custom processing (hide_lines.py removes `# <%hide%>` markers)
- `overrides/` - Custom theme elements
- `javascripts/` - Client-side enhancements
