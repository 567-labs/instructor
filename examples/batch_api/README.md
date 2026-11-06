# Batch API Examples

This directory contains examples and test scripts for Instructor's batch processing capabilities, including both traditional file-based and new in-memory processing.

## Examples

### 1. In-Memory Batch Processing (`in_memory_batch_example.py`)

Demonstrates the new in-memory batch processing feature, perfect for serverless deployments:

```bash
python in_memory_batch_example.py
```

**Key Features:**
- No disk I/O required - ideal for serverless environments
- BytesIO buffers instead of temporary files  
- Automatic cleanup - no file management needed
- Security benefits - no temporary files on disk

### 2. Unified Test Script (`run_batch_test.py`)

Tests the unified BatchProcessor with all supported providers: OpenAI, Anthropic, and Google Gemini.

The script creates a batch job to extract structured `User(name: str, age: int)` data from 10 text examples and saves the batch ID for later checking. Since batch jobs can take time to complete, the script returns immediately after creation.

## Unified Test Script (`run_batch_test.py`)

Tests the unified BatchProcessor with any supported provider/model combination.

### Usage

```bash
# Test OpenAI
export OPENAI_API_KEY="your-openai-api-key"
python run_batch_test.py create --model "openai/gpt-4o-mini"

# Test Anthropic  
export ANTHROPIC_API_KEY="your-anthropic-api-key"
python run_batch_test.py create --model "anthropic/claude-3-5-sonnet-20241022"

# Test Google (simulation mode)
python run_batch_test.py create --model "google/gemini-2.0-flash-001"
```

### Supported Models

Use the `list-models` command to see all supported models:

```bash
python run_batch_test.py list-models
```

**OpenAI Models:**
- `openai/gpt-4o-mini`
- `openai/gpt-4o`
- `openai/gpt-4-turbo`

**Anthropic Models:**
- `anthropic/claude-3-5-sonnet-20241022`
- `anthropic/claude-3-opus-20240229`
- `anthropic/claude-3-haiku-20240307`

**Google Models:**
- `google/gemini-2.0-flash-001`
- `google/gemini-pro`
- `google/gemini-pro-vision`

### What the Script Does

1. **Creates test messages**: 10 prompts containing user information
2. **Uses BatchProcessor**: Leverages the unified API with provider detection
3. **Generates batch file**: Provider-specific format with JSON schema
4. **Submits batch job**: Actual API call to create the batch
5. **Saves batch ID**: Stores ID in `{provider}_batch_id.txt`
6. **Returns immediately**: No waiting for completion

### API Keys Required

| Provider | Environment Variable | Required |
|----------|---------------------|----------|
| OpenAI | `OPENAI_API_KEY` | Yes |
| Anthropic | `ANTHROPIC_API_KEY` | Yes |
| Google | `GOOGLE_API_KEY` | No (simulation mode) |

### Output Files

Each run creates:
- `{provider}_batch_id.txt` - Contains the batch ID for status checking
- Temporary batch files (automatically cleaned up)

### Test Data

All providers use the same 10 test prompts:

1. "Hi there! My name is Alice and I'm 28 years old. I work as a software engineer."
2. "Hello, I'm Bob, 35 years old, and I love hiking and photography."
3. "This is Sarah speaking. I'm 42 and I'm a graphic designer."
4. "Hey! John here, I'm 29 years old and I teach high school math."
5. "I'm Emma, 33 years old, currently working as a marketing manager."
6. "My name is Michael and I'm 45 years old. I'm a chef at a downtown restaurant."
7. "I'm Lisa, 31 years old, working as a nurse at the local hospital."
8. "This is David, 38 years old, I'm a freelance photographer."
9. "Hello, I'm Jessica, 26 years old, and I'm a data scientist."
10. "I'm Ryan, 41 years old, working in software development for a tech startup."

### Expected Results

Each batch job should extract `User` objects:

```python
class User(BaseModel):
    name: str
    age: int
```

Expected extractions:
- Alice, 28 | Bob, 35 | Sarah, 42 | John, 29 | Emma, 33
- Michael, 45 | Lisa, 31 | David, 38 | Jessica, 26 | Ryan, 41

## Checking Batch Status

After creating batch jobs, use the CLI to check their status:

```bash
# List all batch jobs for a provider
instructor batch list --model "openai/gpt-4o-mini"
instructor batch list --model "anthropic/claude-3-5-sonnet-20241022"

# Check specific batch status
instructor batch status --batch-id "batch_123" --model "openai/gpt-4o-mini"

# Get results when completed
instructor batch results \
  --batch-id "batch_123" \
  --output-file "results.jsonl" \
  --model "openai/gpt-4o-mini"
```

## Processing Times

- **OpenAI**: Usually completes within a few hours, guaranteed within 24h
- **Anthropic**: Most batches complete in under 1 hour
- **Google**: Varies (simulation only in this test)

## Running Tests for All Providers

```bash
# Test all providers (requires API keys)
python run_batch_test.py create --model "openai/gpt-4o-mini"
python run_batch_test.py create --model "anthropic/claude-3-5-sonnet-20241022" 
python run_batch_test.py create --model "google/gemini-2.0-flash-001"

# Check what was created
ls *_batch_id.txt
```

## Troubleshooting

### Common Issues

1. **API Key Not Set**
   ```
   ❌ Error: OPENAI_API_KEY environment variable is not set
   ```
   Solution: Set the appropriate environment variable.

2. **Invalid Model Format**
   ```
   ❌ Error: Model must be in format 'provider/model-name'
   ```
   Solution: Use the format `provider/model-name`, e.g., `openai/gpt-4o-mini`.

3. **Unsupported Provider**
   ```
   ❌ Unsupported provider: xyz
   ```
   Solution: Use `openai`, `anthropic`, or `google` as the provider.

### Provider-Specific Notes

**OpenAI:**
- Requires valid API key with sufficient credits
- Supports both individual and organization accounts
- Rate limits are separate for batch vs regular API

**Anthropic:**
- Uses beta API endpoints (`client.beta.messages.batches`)
- Requires Anthropic API access
- May have different availability by region

**Google:**
- Runs in simulation mode by default
- Full implementation requires Google Cloud Storage setup
- Would need proper GCS authentication for real batch jobs

## Integration with CLI

This test validates that the unified BatchProcessor works correctly, which powers the CLI commands:

```bash
# Create batch using CLI directly
instructor batch create \
  --messages-file messages.jsonl \
  --model "openai/gpt-4o-mini" \
  --response-model "examples.User" \
  --output-file batch_requests.jsonl

# Submit the batch
instructor batch create-from-file \
  --file-path batch_requests.jsonl \
  --model "openai/gpt-4o-mini"
```

## Development

To modify the test:
1. Update `create_test_messages()` to change test data
2. Modify the `User` model if needed
3. Add new providers in the provider detection logic
4. Adjust batch creation functions for new provider-specific behavior

The test demonstrates that the same code works across all providers thanks to the unified BatchProcessor abstraction!