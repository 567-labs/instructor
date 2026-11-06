# Scripts Directory

This directory contains utility scripts for maintaining and improving the Instructor documentation and project structure.

## Available Scripts

### 1. `make_clean.py` - Markdown File Cleaner

**Purpose**: Cleans markdown files by removing special whitespace characters and replacing em dashes with regular dashes.

**What it does**:
- Recursively finds all `.md` files in the `docs/` directory
- Removes special Unicode whitespace characters (non-breaking spaces, zero-width spaces, etc.)
- Replaces em dashes (`—`) and en dashes (`–`) with regular dashes (`-`)
- Preserves intentional formatting while cleaning problematic characters

**Usage**:
```bash
# Clean all markdown files in docs/
python scripts/make_clean.py

# Dry run to see what would be changed
python scripts/make_clean.py --dry-run

# Clean files in a different directory
python scripts/make_clean.py --docs-dir path/to/docs
```

**Pre-commit Integration**: This script runs automatically on commits that include markdown files in the `docs/` directory.

### 2. `check_blog_excerpts.py` - Blog Post Excerpt Validator

**Purpose**: Ensures all blog posts contain the `<!-- more -->` tag for proper excerpt handling.

**What it does**:
- Scans all markdown files in `docs/blog/posts/`
- Checks for the presence of `<!-- more -->` tags
- Reports files missing the tag
- Exits with error code 1 if any files are missing the tag

**Usage**:
```bash
# Check all blog posts
python scripts/check_blog_excerpts.py

# Check posts in a different directory
python scripts/check_blog_excerpts.py --blog-posts-dir path/to/posts
```

**Pre-commit Integration**: This script runs automatically on commits that include blog post files.

### 3. `make_sitemap.py` - Enhanced Documentation Sitemap Generator

**Purpose**: Generates an enhanced sitemap (`sitemap.yaml`) with AI-powered content analysis and cross-link suggestions.

**What it does**:
- Recursively traverses the `docs/` directory
- Analyzes each markdown file using OpenAI's GPT-4o-mini
- Extracts summaries, keywords, and topics for SEO
- Identifies internal links and references
- Generates cross-link suggestions based on content similarity
- Creates a comprehensive `sitemap.yaml` file

**Features**:
- **Caching**: Reuses analysis for unchanged files (based on content hash)
- **Concurrent Processing**: Processes multiple files simultaneously
- **Cross-linking**: Suggests related documents based on content similarity
- **Retry Logic**: Handles API failures with exponential backoff

**Usage**:
```bash
# Generate sitemap with default settings
python scripts/make_sitemap.py

# Customize settings
python scripts/make_sitemap.py \
  --root-dir docs \
  --output-file sitemap.yaml \
  --max-concurrency 10 \
  --min-similarity 0.4

# Use custom API key
python scripts/make_sitemap.py --api-key your-openai-key
```

**Output**: Creates `sitemap.yaml` with structure:
```yaml
file.md:
  summary: "Brief description of the content"
  keywords: ["keyword1", "keyword2", "keyword3"]
  topics: ["topic1", "topic2", "topic3"]
  references: ["other-file.md", "another-file.md"]
  ai_references: ["ai-detected-reference.md"]
  cross_links: ["suggested-related-file.md"]
  hash: "content-hash-for-caching"
```

**Requirements**: 
- OpenAI API key (set as `OPENAI_API_KEY` environment variable or passed via `--api-key`)
- Dependencies: `openai`, `typer`, `rich`, `tenacity`, `pyyaml`

## Pre-commit Integration

These scripts are integrated into the project's pre-commit hooks to ensure code quality:

- **`make_clean.py`**: Runs on commits with markdown files in `docs/`
- **`check_blog_excerpts.py`**: Runs on commits with blog post files

The hooks are configured in `.pre-commit-config.yaml` and run automatically during the commit process.

## Running Scripts Manually

You can run any script manually for testing or one-time operations:

```bash
# Test markdown cleaning
python scripts/make_clean.py --dry-run

# Check blog excerpts
python scripts/check_blog_excerpts.py

# Generate fresh sitemap
python scripts/make_sitemap.py
```

### 4. `fix_api_calls.py` - API Call Standardization

**Purpose**: Replaces old API call patterns with simplified versions for consistency.

**What it does**:
- Finds and replaces `client.chat.completions.create` → `client.create`
- Finds and replaces `client.chat.completions.create_partial` → `client.create_partial`
- Finds and replaces `client.chat.completions.create_iterable` → `client.create_iterable`
- Finds and replaces `client.chat.completions.create_with_completion` → `client.create_with_completion`
- Processes all markdown and notebook files in the docs directory

**Usage**:
```bash
# Dry run to see what would be changed
python scripts/fix_api_calls.py --dry-run

# Apply changes to all files
python scripts/fix_api_calls.py

# Process a single file
python scripts/fix_api_calls.py --file docs/index.md

# Custom docs directory
python scripts/fix_api_calls.py --docs-dir path/to/docs
```

### 5. `fix_old_patterns.py` - Client Initialization Pattern Fixer

**Purpose**: Replaces old client initialization patterns with the modern `from_provider` API.

**What it does**:
- Replaces `instructor.from_openai(OpenAI())` → `instructor.from_provider("openai/model-name")`
- Replaces `instructor.from_anthropic(Anthropic())` → `instructor.from_provider("anthropic/model-name")`
- Replaces `instructor.patch(OpenAI())` → `instructor.from_provider("openai/model-name")`
- Handles all supported providers (OpenAI, Anthropic, Google, Cohere, Mistral, Groq, etc.)
- Attempts to extract model names from existing code

**Usage**:
```bash
# Dry run to see what would be changed
python scripts/fix_old_patterns.py --dry-run

# Apply changes to all files
python scripts/fix_old_patterns.py

# Process a single file
python scripts/fix_old_patterns.py --file docs/integrations/openai.md
```

**Note**: Model names are extracted from existing code when possible, but may need manual review for accuracy.

### 6. `audit_patterns.py` - Pattern Auditor

**Purpose**: Audits documentation files to find old patterns that need updating.

**What it does**:
- Finds old API call patterns (`client.chat.completions.*`)
- Finds old initialization patterns (`instructor.from_*`, `instructor.patch`)
- Identifies potentially unused imports
- Reports line numbers for each issue
- Provides summary statistics

**Usage**:
```bash
# Detailed report with line numbers
python scripts/audit_patterns.py

# Summary statistics only
python scripts/audit_patterns.py --summary

# Audit a single file
python scripts/audit_patterns.py --file docs/index.md

# Custom docs directory
python scripts/audit_patterns.py --docs-dir path/to/docs
```

**Output**: Reports issues by file with line numbers, or summary statistics showing total counts per pattern type.

## Adding New Scripts

When adding new scripts to this directory:

1. **Documentation**: Add a section to this README explaining the script's purpose and usage
2. **Pre-commit Integration**: If appropriate, add the script to `.pre-commit-config.yaml`
3. **Error Handling**: Ensure scripts exit with appropriate error codes
4. **Help Text**: Include `--help` functionality for command-line scripts
5. **Testing**: Test scripts manually before committing

## Dependencies

Most scripts use only Python standard library modules. The sitemap generator requires additional dependencies:

```bash
uv add openai typer rich tenacity pyyaml
```

## Troubleshooting

**Pre-commit hooks failing**:
- Check that scripts are executable: `chmod +x scripts/*.py`
- Verify script paths in `.pre-commit-config.yaml`
- Run scripts manually to identify issues

**Sitemap generation issues**:
- Ensure OpenAI API key is set correctly
- Check network connectivity for API calls
- Review error messages for specific file issues

**Markdown cleaning issues**:
- Use `--dry-run` to preview changes
- Check file permissions in the docs directory
- Verify UTF-8 encoding of markdown files 