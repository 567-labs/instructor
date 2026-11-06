# Cursor Rules

Cursor rules are configuration files that help guide AI-assisted development in the Cursor IDE. They provide structured instructions for how the AI should behave in specific contexts or when working with certain types of files.

## What is Cursor?

[Cursor](https://cursor.sh) is an AI-powered IDE that helps developers write, understand, and maintain code more efficiently. It integrates AI capabilities directly into the development workflow, providing features like:

- AI-assisted code completion
- Natural language code generation
- Intelligent code explanations
- Automated refactoring suggestions

## Understanding Cursor Rules

Cursor rules are defined in `.mdc` files within the `.cursor/rules` directory. Each rule file follows a specific naming convention: lowercase names with the `.mdc` extension (e.g., `simple-language.mdc`).

Each rule file contains:

1. **Metadata Header**: YAML frontmatter that defines:
   ```yaml
   ---
   description: when to apply this rule
   globs: file patterns to match (e.g., "*.py", "*.md", or "*" for all files)
   alwaysApply: true/false  # whether to apply automatically
   ---
   ```

2. **Rule Content**: Markdown-formatted instructions that guide the AI's behavior

## Available Rules

Currently, the following rules are defined:

### `simple-language.mdc`
- **Purpose**: Ensures documentation is written at a grade 10 reading level
- **Applies to**: Markdown files (*.md)
- **Auto Apply**: No
- **Key Requirements**: 
  - Write at grade 10 reading level
  - Ensure code blocks are self-contained with complete imports

### `new-features-planning.mdc`
- **Purpose**: Guides feature implementation workflow
- **Applies to**: Python files (*.py)
- **Auto Apply**: Yes
- **Key Requirements**:
  - Create new branch from main
  - Make incremental commits
  - Create todo.md for large features
  - Start pull requests using GitHub CLI (`gh`)
  - Include "This PR was written by [Cursor](https://cursor.sh)" in PRs

### `followups.mdc`
- **Purpose**: Ensures thoughtful follow-up suggestions
- **Applies to**: All files
- **Auto Apply**: Yes
- **Key Requirements**:
  - Generate actionable hotkey suggestions using:
    - [J]: First follow-up action
    - [K]: Second follow-up action
    - [L]: Third follow-up action
  - Focus on small, contextual code changes
  - Suggestions should be thoughtful and actionable

### `documentation-sync.mdc`
- **Purpose**: Maintains documentation consistency with code changes
- **Applies to**: Python and Markdown files (*.py, *.md)
- **Auto Apply**: Yes
- **Key Requirements**:
  - Update docs when code changes
  - Add new markdown files to mkdocs.yml
  - Keep API documentation current
  - Maintain documentation quality standards

## Creating New Rules

To create a new rule:

1. Create a `.mdc` file in `.cursor/rules/` using lowercase naming
2. Add YAML frontmatter with required metadata:
   ```yaml
   ---
   description: when to apply this rule
   globs: file patterns to match
   alwaysApply: true/false
   ---
   ```
3. Write clear, specific instructions in Markdown
4. Test the rule with relevant file types

## Best Practices

- Keep rules focused and specific
- Use clear, actionable language
- Test rules thoroughly before committing
- Document any special requirements or dependencies
- Update rules as project needs evolve
- Use consistent file naming (lowercase with .mdc extension)
- Ensure globs patterns are explicit and documented
