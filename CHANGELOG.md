# Changelog

All notable changes to instructor are documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html)

---

## [Unreleased]

## [1.17.1] - 2026-09-09

### Upgrade Notes
- GenAI tools and JSON requests now raise `IncompleteOutputException` for non-streaming output marked `MAX_TOKENS`, even if the partial payload fits schema defaults. Increase the output-token budget or handle incomplete output explicitly.
- Ordinary dictionaries containing non-string keys, including nested dictionaries, bypass caching. Pydantic models use their JSON-serialized representation for cache identity, which does not preserve every distinction between original Python key types.

### Fixed
- **Anthropic retry usage**: Accumulate documented thinking-token counts when an older SDK retains `output_tokens_details` as a dictionary. Preserve existing SDK response types, cumulative totals, and unknown/null metadata behavior.
- **CLI installation**: Declare the CLI’s `tqdm` dependency directly to fix the missing-dependency crash in OpenAI 3 installations, which no longer supply it transitively. Defer file and job client construction so top-level and subcommand help work without provider credentials.
- **Partial streaming**: Preserve model instances inside nullable list fields in sync and async streams, while retaining validation context and final validation. ([#2600](https://github.com/567-labs/instructor/pull/2600))
- **GenAI truncation**: Raise `IncompleteOutputException` for non-streaming tools and JSON responses ending with `MAX_TOKENS`, instead of accepting schema defaults for missing output. ([#2601](https://github.com/567-labs/instructor/pull/2601))
- **GenAI templating**: Preserve text-part metadata, including thought flags and signatures, without modifying caller-owned conversation history. ([#2607](https://github.com/567-labs/instructor/issues/2607), [#2608](https://github.com/567-labs/instructor/pull/2608))
- **Documentation links**: Repair context-validation, quick-start and example links, and use the correct relative API link in the provider documentation template. Consolidates [#2599](https://github.com/567-labs/instructor/pull/2599), [#2609](https://github.com/567-labs/instructor/pull/2609), [#2610](https://github.com/567-labs/instructor/pull/2610) and [#2611](https://github.com/567-labs/instructor/pull/2611).

### Security
- **Cache identity**: Bypass caching for ordinary dictionaries containing non-string keys, including nested dictionaries, to prevent key collisions before serialization. ([#2614](https://github.com/567-labs/instructor/pull/2614))

### Changed
- Consolidate duplicate helpers and utility tests while preserving public compatibility, and separate token-budget policy from retry execution without changing budget arithmetic or retry behavior. ([#2616](https://github.com/567-labs/instructor/pull/2616), [#2617](https://github.com/567-labs/instructor/pull/2617))

### Documentation
- Document provider-specific usage accounting, raw-response mutation, streaming/cache/batch limitations, and legacy CLI estimate limitations.

## [1.17.0] - 2026-09-04

Includes the fixes previously planned for 1.16.1, which was not published. The
previous published version is 1.16.0.

### Upgrade Notes
- Cached responses use new keys and isolated per-client namespaces. Existing cache entries will miss. Set the same explicit `cache_namespace` only when clients are intended to share results; keep accounts, endpoints, and tenants isolated.
- Response models with Instructor async-validator decorators are rejected before provider calls or cache lookup, including nested models. Use Pydantic validators or explicitly await application-level validation after extraction; automatic async-validator execution is deferred.
- Remote media URLs must resolve to public addresses and cannot contain credentials. Redirects and connected peers are checked, and downloads have size limits.

### Fixed
- **Cache isolation**: Include provider identity and prepared generation settings, including nested GenAI and Bedrock configuration; reuse the lookup key when storing after retries. Cache hits preserve current validation context and strictness. Clients have separate namespaces by default; use an explicit `cache_namespace` for intentional reuse across instances. Requests with unsupported opaque values bypass caching rather than using ambiguous identities. ([#2585](https://github.com/567-labs/instructor/issues/2585), [#2586](https://github.com/567-labs/instructor/pull/2586))
- **GenAI cached content**: Omit conflicting system instructions and tool declarations after resolving cached-content configuration, including plain responses, and preserve caller-owned configuration. Consolidates [#2581](https://github.com/567-labs/instructor/pull/2581), [#2582](https://github.com/567-labs/instructor/pull/2582), and [#2591](https://github.com/567-labs/instructor/pull/2591) for [#2580](https://github.com/567-labs/instructor/issues/2580).
- **GenAI request configuration**: Preserve caller-owned `generation_config` dictionaries when preparing tools and JSON requests, so repeated requests retain their sampling settings and token limits. ([#2596](https://github.com/567-labs/instructor/pull/2596))
- **OpenAI SDK compatibility**: Support OpenAI 3.x and its HTTPX2 transport when constructing sync and async clients through `from_provider`, while retaining OpenAI 2.x support on Python 3.9. ([#2553](https://github.com/567-labs/instructor/issues/2553))
- **Async validation policy**: Fail closed for `@async_field_validator` and `@async_model_validator` response models instead of silently skipping policy checks. The proposed automatic execution in [#2588](https://github.com/567-labs/instructor/pull/2588) is deferred in favor of main's security fix for [#2528](https://github.com/567-labs/instructor/issues/2528).

- **Anthropic local PDFs**: Read local PDF sources from disk instead of trying to fetch them over HTTP. ([#2577](https://github.com/567-labs/instructor/pull/2577))

### Security
- Route Anthropic PDF downloads through the bounded public-network fetcher and pin each media connection to a validated IP before sending HTTP.
- Key cached responses by the complete prepared request, provider, validation context and strictness. Clients have isolated cache namespaces by default; `cache_namespace` explicitly enables sharing and must identify the endpoint and tenant. Revalidate cache hits with the current context and strictness.
- Reject response models containing unsupported async-validator markers, including nested models, instead of silently skipping their policy checks.
- Bound JSON extraction to 1 MiB of characters and 128 nesting levels, and avoid repeatedly scanning malformed suffixes.

### Changed
- **Current model examples**: Refresh 16 provider guides, including GPT-5.6 Luna, Sonnet 5, Gemini 3.8 Flash, and current hosted model IDs. Correct Luna tool-call parameters, Sonnet adaptive thinking, and audio transcription guidance; add real model-ID smoke tests and code-block lint checks. Normalize Google test model overrides and default to Gemini 3.8 Flash.
- **Documentation examples**: Explain OpenAI-compatible JSON endpoints; correct typos, malformed client calls, missing imports, and code-block formatting across concepts and blog examples. Label displayed output separately from executable Python. ([#2578](https://github.com/567-labs/instructor/pull/2578), [#2579](https://github.com/567-labs/instructor/pull/2579), [#2584](https://github.com/567-labs/instructor/pull/2584), [#2587](https://github.com/567-labs/instructor/pull/2587), [#2593](https://github.com/567-labs/instructor/pull/2593))
- **CLI cost metadata**: Add an explicit mapping annotation to the model-cost table so static analysis preserves its nested numeric value shape. ([#2521](https://github.com/567-labs/instructor/pull/2521))
- **Provider documentation**: Correct the Mistral installation extra and xAI Python requirement, document Cerebras request-parameter forwarding with a runnable example, repair the Langfuse tracing examples, and clean up extraction typos. ([#2550](https://github.com/567-labs/instructor/pull/2550), [#2554](https://github.com/567-labs/instructor/pull/2554), [#2556](https://github.com/567-labs/instructor/pull/2556), [#2561](https://github.com/567-labs/instructor/pull/2561), [#2562](https://github.com/567-labs/instructor/pull/2562))
- **Live-provider CI signal**: Retry only failed tests once in the mixed-provider and auto-client lanes so transient API failures do not obscure deterministic regressions, while setup, collection, repeated, and other failures remain red.

### Security
- **Remote multimodal fetching**: Block non-public and credential-bearing media URLs, revalidate redirects and connected peers, disable ambient proxy credentials, cap image, audio, and PDF downloads, and avoid caching decoded media payloads in process memory.
- **Supply-chain hardening**: Require a patched `urllib3` release and pin GitHub Actions to reviewed commit SHAs, including secret-bearing scheduled workflows.
- **Release workflow hardening**: Avoid persisting checkout credentials, disable dependency-cache restoration in the PyPI publishing job, and pass release metadata to shell steps through environment variables instead of direct expression interpolation.

### Additional Fixes
- **Anthropic batch messages**: Preserve every system instruction when converting batch requests instead of keeping only the final system message. ([#2552](https://github.com/567-labs/instructor/pull/2552))
- **Anthropic batch accounting**: Include canceled and expired requests in reported batch totals. ([#2529](https://github.com/567-labs/instructor/pull/2529))
- **Bedrock request safety**: Reject unsupported numerical, string-length, and `minItems > 1` constraints locally for native structured outputs, and preserve caller-owned nested inference configuration while normalizing requests. ([#2530](https://github.com/567-labs/instructor/pull/2530), [#2532](https://github.com/567-labs/instructor/pull/2532))
- **Cache key isolation**: Include provider-hoisted system prompts in sync and async cache keys so requests with different instructions cannot share a cached response. ([#2524](https://github.com/567-labs/instructor/pull/2524))
- **Gemini safety settings**: Copy caller-provided safety mappings before applying default thresholds so request preparation cannot mutate reusable application configuration. ([#2551](https://github.com/567-labs/instructor/pull/2551))
- **Google GenAI validation retries**: Use the supported user role for function-response turns so Gemini can process correction requests. ([#2557](https://github.com/567-labs/instructor/pull/2557))
- **TypedDict response models**: Preserve `total=False`, `Required`, and `NotRequired` key semantics for single and iterable response models so valid partial outputs do not trigger retries. ([#2567](https://github.com/567-labs/instructor/issues/2567), [#2568](https://github.com/567-labs/instructor/pull/2568))
- **Provider initialization**: Reject provider strings with an empty provider or model component before client construction. ([#2522](https://github.com/567-labs/instructor/pull/2522))
- **Remote multimodal media types**: Accept valid case-insensitive HTTP `Content-Type` values with optional parameters when loading images, audio, and PDFs. ([#2525](https://github.com/567-labs/instructor/pull/2525))
- **Message history**: Preserve tool-call and other protocol fields when normalizing consecutive messages, and keep protocol messages as separate turns. ([#2527](https://github.com/567-labs/instructor/pull/2527))
- **v2 mode registry**: Keep lazily loaded modes registered while their handlers initialize, preventing concurrent first calls from failing spuriously. ([#2536](https://github.com/567-labs/instructor/pull/2536), [#2535](https://github.com/567-labs/instructor/issues/2535))

## [1.16.0] - 2026-08-09

### Added
- **Bedrock native structured outputs**: Add explicit `Mode.JSON_SCHEMA` and `Mode.TOOLS_STRICT` support through Converse `outputConfig.textFormat` and strict tool schemas, with recursive schema normalization and a boto3 `1.42.42` minimum. Model selection remains caller-controlled. ([#2515](https://github.com/567-labs/instructor/pull/2515), [#2084](https://github.com/567-labs/instructor/issues/2084), [#2086](https://github.com/567-labs/instructor/pull/2086))
- **Validation retry budgets**: Add positive cumulative `token_budget` limits for structured non-streaming retries, immutable `completion:usage` snapshots, sync/async cutoff parity, and stable cumulative usage metadata. Valid responses still win after crossing the budget. When a budget is configured, a failed attempt with unavailable usage stops before another provider call. ([#2512](https://github.com/567-labs/instructor/pull/2512), [#2391](https://github.com/567-labs/instructor/issues/2391), [#2392](https://github.com/567-labs/instructor/pull/2392))

### Changed
- **Contributor workflow**: Align contributor setup and dependency management around locked `uv sync` environments, `uv run` commands, and `uv add` only for intentional project metadata changes. ([#2516](https://github.com/567-labs/instructor/pull/2516), [#2354](https://github.com/567-labs/instructor/pull/2354))

### Fixed
- **Mistral SDK compatibility**: Support the `mistralai` 2.x client export on Python 3.10+ while retaining the compatible 1.x fallback required by Python 3.9. ([#2513](https://github.com/567-labs/instructor/pull/2513), [#2298](https://github.com/567-labs/instructor/pull/2298), [#2365](https://github.com/567-labs/instructor/issues/2365))
- **Bedrock reasoning JSON**: Parse the final complete JSON value after reasoning text or `<think>` blocks, preserve JSON escape sequences, and keep caller-owned messages unchanged during Bedrock request preparation and retries. ([#2514](https://github.com/567-labs/instructor/pull/2514), [#2076](https://github.com/567-labs/instructor/issues/2076), [#2287](https://github.com/567-labs/instructor/pull/2287))
- **Remote multimodal fetches**: Apply the existing 30-second request timeout to image, audio, and PDF downloads so an unresponsive URL cannot block a caller indefinitely. ([#2507](https://github.com/567-labs/instructor/pull/2507))
- **OpenAI streaming retries**: Keep TOOLS, JSON, JSON_SCHEMA, and MD_JSON retries on the streaming parser after the one-shot model marker is consumed, allowing corrected streamed responses to validate successfully. ([#2508](https://github.com/567-labs/instructor/pull/2508))
- **Iterable streaming unions**: Parse PEP 604 unions (`create_iterable(response_model=Weather | GoogleSearch)`, `Iterable[Weather | GoogleSearch]`) member by member instead of calling `model_validate_json` on `types.UnionType`. ([#2509](https://github.com/567-labs/instructor/pull/2509))
- **Package metadata**: Point the published distribution's repository URL at the current `567-labs/instructor` organization and validate it before release.
- **Retry usage accounting**: Accumulate nested and newly added numeric usage fields across OpenAI and Anthropic retries, including prediction, cache-write, cache-creation, and server-tool counters, without treating boolean metadata as billable usage. ([#2493](https://github.com/567-labs/instructor/issues/2493), [#2500](https://github.com/567-labs/instructor/pull/2500))
- **OpenAI Responses reask**: Add a fallback correction message when a `RESPONSES_TOOLS` response contains no tool calls (e.g. reasoning-only output), so retries carry validation feedback instead of resending the identical request. ([#2498](https://github.com/567-labs/instructor/pull/2498))
- **v2 parallel tools**: Preserve raw iterable type hints through the sync and async patch wrappers so parallel tool schemas and results retain every requested model type. ([#2501](https://github.com/567-labs/instructor/pull/2501))
- **Credential redaction**: Hide common OAuth and Google API credential aliases in nested v2 debug logging while preserving non-secret token configuration. ([#2490](https://github.com/567-labs/instructor/issues/2490), [#2491](https://github.com/567-labs/instructor/pull/2491))
- **Retry and message integrity**: Preserve cache keys and caller-owned retry messages, retain empty-content legacy function calls, return Anthropic tool results for every parallel tool call, and handle missing OpenAI/Mistral tool calls as retryable parse failures. ([#2454](https://github.com/567-labs/instructor/issues/2454), [#2455](https://github.com/567-labs/instructor/pull/2455), [#2464](https://github.com/567-labs/instructor/issues/2464), [#2484](https://github.com/567-labs/instructor/pull/2484), [#2485](https://github.com/567-labs/instructor/issues/2485), [#2486](https://github.com/567-labs/instructor/pull/2486), [#2448](https://github.com/567-labs/instructor/pull/2448), [#2453](https://github.com/567-labs/instructor/pull/2453))
- **Streaming and DSL correctness**: Isolate partial-model recursion guards, preserve partial nested models and explicit nulls, harden citation matching, derive useful Iterable union names, and continue scanning JSON streams after non-JSON or multiple balanced values. ([#2422](https://github.com/567-labs/instructor/issues/2422), [#2430](https://github.com/567-labs/instructor/pull/2430), [#2431](https://github.com/567-labs/instructor/issues/2431), [#2452](https://github.com/567-labs/instructor/pull/2452), [#2456](https://github.com/567-labs/instructor/pull/2456), [#2461](https://github.com/567-labs/instructor/issues/2461), [#2463](https://github.com/567-labs/instructor/pull/2463), [#2476](https://github.com/567-labs/instructor/pull/2476), [#2487](https://github.com/567-labs/instructor/pull/2487), [#2489](https://github.com/567-labs/instructor/pull/2489))
- **Provider request handling**: Avoid mutating Gemini generation config and cached OpenAI schemas, disable Anthropic parallel calls for forced single-tool requests, forward Bedrock default models, and label OpenAI audio as WAV or MP3 without misrepresenting unsupported formats. ([#2450](https://github.com/567-labs/instructor/issues/2450), [#2451](https://github.com/567-labs/instructor/pull/2451), [#2465](https://github.com/567-labs/instructor/issues/2465), [#2467](https://github.com/567-labs/instructor/pull/2467), [#2477](https://github.com/567-labs/instructor/issues/2477), [#2478](https://github.com/567-labs/instructor/pull/2478), [#2447](https://github.com/567-labs/instructor/pull/2447), [#2415](https://github.com/567-labs/instructor/pull/2415))
- **Batch, CLI, and citation runtime**: Accept valid empty batch objects, use typed OpenAI file attributes in the CLI, normalize `None` message content, and install `regex` as the direct dependency required by `CitationMixin`. ([#2473](https://github.com/567-labs/instructor/pull/2473), [#2441](https://github.com/567-labs/instructor/pull/2441), [#2440](https://github.com/567-labs/instructor/pull/2440), [#2443](https://github.com/567-labs/instructor/pull/2443))
- **Provider documentation**: Refresh retired Cerebras model IDs, clarify current and deprecated Google provider prefixes, and fix the Vertex Google GenAI example so its default model is passed to `from_genai()`. ([#2494](https://github.com/567-labs/instructor/pull/2494), [#2289](https://github.com/567-labs/instructor/issues/2289), [#2343](https://github.com/567-labs/instructor/pull/2343), [#2416](https://github.com/567-labs/instructor/issues/2416), [#2475](https://github.com/567-labs/instructor/pull/2475))
- **Multimodal (Audio)**: Raise explicit `ValueError` or `FileNotFoundError` from `Audio.from_url()` and `Audio.from_path()` instead of relying on bare `assert` statements that can disappear under `python -O`. ([#2361](https://github.com/567-labs/instructor/pull/2361))
- **v2 message handling**: Preserve caller-owned message lists and nested content across request preparation and retries for OpenAI-compatible, Cohere, Mistral, OpenRouter, Writer, and xAI handlers. ([#2417](https://github.com/567-labs/instructor/issues/2417), [#2428](https://github.com/567-labs/instructor/issues/2428))
- **v2 JSON extraction**: Prefer the final complete top-level JSON value in text responses and retain every JSON object when multiple objects arrive in one streaming chunk.
- **v2 schemas**: Treat fields with Pydantic `default_factory` values as optional in generated OpenAI tool schemas.
- **v2 partial streaming**: Build model instances for present `Optional[BaseModel]` fields during incomplete streams instead of exposing raw dictionaries.
- **v2 iterable unions**: Generate stable member-derived names such as `IterableAOrB` for both `Union[A, B]` and `A | B` response models.
- **Mistral/Vertex AI partial streaming**: Avoid forwarding iterable-only parser arguments into completed `Partial` responses, preventing final Pydantic validation errors for sync and async streams.
- **Batch providers**: Handle missing optional SDKs safely, validate OpenAI batch input before client setup, report exhausted output-file retries clearly, and remove unreachable fallbacks.
- **OpenAI/Writer tools**: Raise clear response-parsing errors for completions with no choices or tool calls instead of leaking attribute and index errors.
- **Fireworks streaming**: Keep non-streaming async calls non-streaming and return streaming async generators without incorrectly awaiting them.
- **GenAI uploads**: Respect `max_retries=0` without an unwanted sleep or polling request, allow recovery on the final permitted retry, and report nameless pending uploads clearly before polling.
- **Gemini/GenAI messages**: Honor an explicit system message for unstructured requests, remove the unsupported raw `system` argument, and reject invalid scalar message content clearly.
- **Templating**: Use populated `contents` when `messages` is empty, avoid mutating nested caller input, and preserve uncopyable metadata during template expansion.
- **Anthropic system messages**: Reject invalid new system-message values even when no existing system message is present.
- **Python 3.9**: Include the required type-evaluation backport in minimal installs, keep overload metadata available, and avoid runtime evaluation of unsupported union syntax in the core response path and offline tests.
- **v2 imports**: Defer OpenAI SDK imports from core v2 modules until an OpenAI-specific path actually needs them, reducing import side effects for non-OpenAI usage. ([#2390](https://github.com/567-labs/instructor/pull/2390))
- **v2 response models**: Treat `list[A | B]` PEP 604 unions of Pydantic models as iterable response models, matching `list[Union[A, B]]` schema behavior. ([#2377](https://github.com/567-labs/instructor/pull/2377))
- **OpenAI Responses API**: Align `RESPONSES_TOOLS` `text.format` with the forced tool schema and add targeted retry guidance when tool calls return empty `{}` arguments. ([#2300](https://github.com/567-labs/instructor/issues/2300), [#2304](https://github.com/567-labs/instructor/pull/2304))

### Security
- **LLM validator isolation**: Send validation rules and candidate values as structured JSON data under a fixed trusted instruction to reduce prompt-injection risk, and raise `ValueError` for rejected values instead of relying on optimization-sensitive assertions. ([#2511](https://github.com/567-labs/instructor/pull/2511), [#2056](https://github.com/567-labs/instructor/issues/2056), [#2307](https://github.com/567-labs/instructor/pull/2307))

### Tests / CI
- **Fork-safe contributor checks**: Mark auto-client network tests explicitly and exclude them from core, coverage, and release lanes so fork PRs without provider secrets do not fail with empty authorization headers.
- **Coverage and test quality**: Run the complete offline suite on Python 3.9-3.13, enforce fork-safe statement and branch coverage plus supported-version type checks in pull-request CI, add strict resource and thread warning checks, and provide a manual retry-mutation workflow. Consolidate typed response, stream, and SDK fixtures; remove duplicate tests and unreachable provider paths; and replace coverage-only stubs with meaningful edge-case and transport-backed provider checks.
- **Release safety**: Validate the declared source, lockfile, changelog, tag, and built artifacts before any publication step; require an explicit version confirmation and publish opt-in; and publish the exact tested assets instead of rebuilding from a moving branch.

---

## [1.15.4] - 2026-06-27

### Fixed
- **CLI fine-tuning**: Use the uploaded validation file ID when creating a fine-tuning job from local files, instead of passing the local validation file path through to OpenAI. ([#2397](https://github.com/567-labs/instructor/pull/2397))
- **v2 core**: Prepare list and primitive response models before provider handler dispatch, fixing `list[Model]` and scalar response-model crashes such as `AttributeError: type object 'list' has no attribute 'model_json_schema'`. ([#2374](https://github.com/567-labs/instructor/issues/2374))
- **v2 streaming**: Preserve backticks inside JSON string values during streamed JSON extraction.
- **v2 multimodal**: Accept raw bytes in `Image.autodetect()` for JPEG, PNG, GIF, and WebP, while raising clear errors for unsupported image inputs. ([#2344](https://github.com/567-labs/instructor/issues/2344))
- **Docs**: Refresh stale OpenAI and Ollama model strings in documentation examples. ([#2395](https://github.com/567-labs/instructor/issues/2395))

---

## [1.15.3] - 2026-06-15

### Fixed
- **Bedrock**: Route `top_k`/`topK` through `additionalModelRequestFields` instead of leaving it as a top-level Converse kwarg. AWS `InferenceConfiguration` only supports `maxTokens`/`stopSequences`/`temperature`/`topP`, so a leftover `top_k` reached `client.converse(top_k=...)` and boto3 raised `ParamValidationError: Unknown parameter "top_k"`.
- **Gemini/GenAI**: Fold `generation_config` (and `safety_settings`/`thinking_config`) into `config` when `response_model=None`, so plain-text calls no longer raise `generate_content() got an unexpected keyword argument 'generation_config'` ([#2366](https://github.com/567-labs/instructor/issues/2366)).
- **v2 cleanup**: Consolidate small provider/runtime fixes for Gemini JSON prompts, Cohere templating, JSON array extraction, iterable streaming, missing `jsonref` dependency guidance, retry semantics and hook metadata, and multimodal autodetection.

### Tests / CI
- **Type checking**: Upgrade to `ty` 0.0.44, enforce warning-free checks with GitHub annotations, cover V2 tests, validate supported Python versions and platforms, and strengthen installed-package public API typing tests.

---

## [1.15.2] - 2026-05-10

### Security
- **Logging**: Redact sensitive request fields from debug logs, including nested auth headers such as `Authorization` and `x-api-key`. ([#2297](https://github.com/567-labs/instructor/pull/2297))

### Fixed
- **Templating (GenAI/VertexAI)**: `process_message` no longer crashes with `TypeError: Can't compile non template nodes` when multimodal messages contain image/URI/bytes Parts alongside `validation_context`. Non-text Parts (where `part.text` is `None`) now pass through unchanged. ([#2253](https://github.com/567-labs/instructor/issues/2253))
- **Retry**: `IncompleteOutputException` now propagates directly to the caller without being wrapped in `InstructorRetryException`, making `except IncompleteOutputException` catch blocks work as documented. Applies to both sync and async paths. ([#2273](https://github.com/567-labs/instructor/issues/2273))
- **Anthropic/Bedrock**: Omit `None` fields from Anthropic tool-use retry payloads so Bedrock reasks no longer fail with HTTP 400 when `caller=None`. ([#2301](https://github.com/567-labs/instructor/pull/2301))
- **Responses streaming**: Surface reasoning-summary events in `RESPONSES_TOOLS` partial streaming and await callback return values when they are awaitable. ([#2299](https://github.com/567-labs/instructor/pull/2299))

## [1.15.1] - 2026-04-03

### Security
- **Bedrock**: Block remote HTTP(S) image URL fetching in `_openai_image_part_to_bedrock` — only `data:` URLs are now accepted, preventing SSRF via user-controlled image URLs
- **Bedrock/PDF**: Block remote URL and local file fetching in `PDF.to_bedrock` — only base64 data or `s3://` sources are now supported, preventing SSRF and local file disclosure

### Added
- **Hooks**: `completion:error` and `completion:last_attempt` handlers now receive `attempt_number`, `max_attempts`, and `is_last_attempt` as keyword arguments. Old-style handlers remain fully backward-compatible.
- **Anthropic**: `from_provider("anthropic/...")` now sets a `User-Agent: instructor/<version>` header on the Anthropic client

### Fixed
- **Anthropic usage**: Initialize usage correctly for `ANTHROPIC_REASONING_TOOLS` and `ANTHROPIC_PARALLEL_TOOLS` modes — previously fell through to OpenAI usage tracking with wrong field names
- **OpenRouter**: Use `reask_md_json` for `OPENROUTER_STRUCTURED_OUTPUTS` retries instead of `reask_default` (tool-call format), fixing malformed retry prompts
- **Templating**: Return `kwargs` unchanged instead of `None` in `handle_templating` when message list is empty or format is unrecognized; `process_message` also now returns the original message unchanged for unrecognized formats instead of `None`
- **`from_openai`**: Allow `Mode.JSON_SCHEMA` for the OpenAI provider — it was incorrectly blocked by the mode validation check
- **Bedrock**: Pass through `cachePoint` dicts in message content unchanged — previously raised `ValueError: Unsupported dict content for Bedrock`, breaking prompt caching (regression since v1.13.0)
- **Bedrock**: Allow `Mode.MD_JSON` in `from_bedrock`
- **Parallel tools**: `ParallelBase` generator now consumed into `ListResponse` in both sync and async paths, fixing `AttributeError` when setting `_raw_response` on a generator

---

## [1.15.0] - 2026-04-02

### Security
- Pin litellm to `<=1.82.6` to block compromised versions 1.82.7 and 1.82.8 ([#2219](https://github.com/567-labs/instructor/pull/2219))
- Make `diskcache` an optional dependency, removing it from all users' transitive dependency trees and mitigating CVE-2025-69872 ([#2211](https://github.com/567-labs/instructor/pull/2211))

### Fixed
- **Usage tracking**: Preserve `response.usage` subclass type (e.g. LiteLLM, Langfuse) when accumulating token counts across retries — fixes downstream `.get()` method loss ([#2217](https://github.com/567-labs/instructor/pull/2217), [#2199](https://github.com/567-labs/instructor/pull/2199))
- **Gemini**: Exclude `HARM_CATEGORY_IMAGE_*` safety categories from standard Gemini API calls — these are Vertex AI-only and caused `400 INVALID_ARGUMENT` errors ([#2174](https://github.com/567-labs/instructor/pull/2174))
- **Gemini**: Detect truncated responses (`finish_reason=MAX_TOKENS`) in `GENAI_STRUCTURED_OUTPUTS` mode and raise `IncompleteOutputException` immediately instead of retrying with malformed JSON ([#2232](https://github.com/567-labs/instructor/pull/2232))
- **`create_with_completion`**: Handle `List[Model]` response models that lack `_raw_response` attribute — previously raised `AttributeError`, now returns `None` for the completion ([#2167](https://github.com/567-labs/instructor/pull/2167))
- **Partial streaming**: Preserve default `Literal` field values (e.g. `type: Literal["Person"] = "Person"`) during streaming instead of emitting `None` before the field arrives ([#2204](https://github.com/567-labs/instructor/pull/2204))
- **Partial streaming**: Support PEP 604 union syntax (`str | int`) in `Partial` models on Python 3.10+ ([#2200](https://github.com/567-labs/instructor/pull/2200))
- **Validators**: Fix `allow_override=True` in `llm_validator` — the override branch was unreachable due to a misplaced assertion, so `fixed_value` was never returned ([#2215](https://github.com/567-labs/instructor/pull/2215))
- **Parallel tools**: `ParallelBase` responses now return `ListResponse` (consistent with `IterableBase`) instead of a raw generator with `_raw_response` set on it ([#2216](https://github.com/567-labs/instructor/pull/2216))
- **Multimodal**: Add missing `continue` in `convert_messages` after handling typed (`audio`/`image`) messages — previously fell through to `message["role"]` causing `KeyError` ([#2139](https://github.com/567-labs/instructor/pull/2139))
- **Anthropic**: Fix dead code path for `ANTHROPIC_REASONING_TOOLS` mode — the mode was shadowed by a duplicate `ANTHROPIC_TOOLS` check and never routed correctly ([#2140](https://github.com/567-labs/instructor/pull/2140))

### Added
- **Models**: Add Claude 4 (Opus, Sonnet, Haiku), OpenAI GPT-4.1 series, o3/o4 reasoning models, xAI Grok 3, and DeepSeek R1/V3 to `KnownModelName` type ([#2235](https://github.com/567-labs/instructor/pull/2235))

### Docs
- Update GitHub organization links in README from `instructor-ai` to `567-labs` ([#2149](https://github.com/567-labs/instructor/pull/2149))

### Tests / CI
- Fix `test_xai_optional_dependency` tests to use `monkeypatch` so they pass regardless of whether `xai-sdk` is installed
- Update deprecated Anthropic model names (`claude-3-5-haiku-latest` -> `claude-haiku-4-0-20250414`, `claude-3-7-sonnet-latest` -> `claude-sonnet-4-5-20250514`)
- Update deprecated OpenAI model names (`gpt-3.5-turbo` -> `gpt-4.1-mini`) across unit tests
- Update stale provider model strings in `shared_config.py`: Writer palmyra-x5, Fireworks llama-v3p3, Perplexity sonar-pro

---

## [1.14.5] - 2026-01-29

### Fixed
- **Google GenAI**: `thought_signature` is now preserved across validation retries for thinking models ([#2001](https://github.com/567-labs/instructor/pull/2001))
- **Metadata**: `pyproject.toml` author field corrected so PyPI correctly populates the `Author` field ([#2015](https://github.com/567-labs/instructor/pull/2015))
- **Deps**: Dev dependencies moved to the correct `[dependency-groups]` section in `pyproject.toml` ([#2030](https://github.com/567-labs/instructor/pull/2030))

---

## [1.14.4] - 2026-01-16

### Fixed
- **Responses API**: Validation errors during structured output parsing are now caught and retried correctly ([#2002](https://github.com/567-labs/instructor/pull/2002))
- **Google GenAI**: User-provided `GenerationConfig` labels and custom fields are no longer silently dropped when merging configs ([#2005](https://github.com/567-labs/instructor/pull/2005))
- **Google GenAI**: `SafetySettings` now applied correctly when request contains image content ([#2007](https://github.com/567-labs/instructor/pull/2007))
- **List responses**: Response wrappers no longer crash on attribute-style access ([#2011](https://github.com/567-labs/instructor/pull/2011))
- **`_raw_response`**: Attribute access on list response wrappers works correctly ([#2012](https://github.com/567-labs/instructor/pull/2012))

### Changed
- **`json_tracker`**: Sibling-heuristic algorithm simplified for improved partial-streaming reliability ([#2000](https://github.com/567-labs/instructor/pull/2000))

---

## [1.14.3] - 2026-01-13

### Added
- **Partial streaming**: Completeness-based streaming validation — fields are validated progressively rather than failing mid-stream ([#1999](https://github.com/567-labs/instructor/pull/1999))

### Fixed
- **Streaming reask**: `Stream` objects in reask handlers are now consumed correctly before retry, preventing stale-stream errors ([#1992](https://github.com/567-labs/instructor/pull/1992))

---

## [1.14.2] - 2026-01-13

### Fixed
- **Partial streaming**: Model validators now skip during partial streaming and run only once on the final complete object, preventing spurious errors ([#1994](https://github.com/567-labs/instructor/pull/1994))
- **Partial**: Infinite recursion with self-referential models (e.g. `TreeNode` with `children: List["TreeNode"]`) is now prevented ([#1997](https://github.com/567-labs/instructor/pull/1997))

### Tests / CI
- Provider tests skipped in CI when API secrets are not available ([#1990](https://github.com/567-labs/instructor/pull/1990))

---

## [1.14.1] - 2026-01-08

### Fixed
- **Google GenAI**: `cached_content` parameter now correctly forwarded to support Google context caching ([#1987](https://github.com/567-labs/instructor/pull/1987))

---

## [1.14.0] - 2026-01-04

### Added
- **Bedrock**: Document support — pass PDFs and text files directly to Bedrock models ([#1936](https://github.com/567-labs/instructor/pull/1936))

### Fixed
- **`from_provider()`**: Now respects the `base_url` keyword argument for OpenAI-compatible providers ([#1971](https://github.com/567-labs/instructor/pull/1971))
- **`from_provider()`**: Runtime `ImportError` exceptions are no longer masked, making misconfigured installs easier to diagnose ([#1975](https://github.com/567-labs/instructor/pull/1975))
- **Google GenAI**: `Union` types now allowed in structured output schemas ([#1973](https://github.com/567-labs/instructor/pull/1973))
- **Google GenAI**: `thinking_config` and additional user-provided `GenerationConfig` fields now correctly preserved ([#1972](https://github.com/567-labs/instructor/pull/1972), [#1974](https://github.com/567-labs/instructor/pull/1974))
- **Cohere**: Streaming and V2 API version detection issues resolved ([#1983](https://github.com/567-labs/instructor/pull/1983), [#1844](https://github.com/567-labs/instructor/pull/1844))
- **xAI**: Tools-mode validation fixed ([#1983](https://github.com/567-labs/instructor/pull/1983))
- **Exception handling**: Standardized across all providers ([#1897](https://github.com/567-labs/instructor/pull/1897))

### Changed
- **Type checker**: Switched from Pyright to `ty` for faster incremental type checking ([#1978](https://github.com/567-labs/instructor/pull/1978))
- **Provider factories**: `from_openai`, `from_anthropic`, etc. signatures standardized ([#1898](https://github.com/567-labs/instructor/pull/1898))

---

## [1.13.0] - 2025-11-03

### Added
- **Bedrock**: Image input support — converts OpenAI-style image parts to Bedrock's native format
- **`py.typed`**: Marker file restored for PEP 561 type-checking support ([#1868](https://github.com/567-labs/instructor/pull/1868))

### Fixed
- **`disable_pydantic_error_url()`**: Now correctly suppresses Pydantic validation error URLs via monkey-patching `ValidationError.__str__()` (environment variable approach had no effect post-import)
- **JSON mode**: JSON decode errors now trigger retry logic instead of surfacing as unhandled exceptions ([#1856](https://github.com/567-labs/instructor/pull/1856))
- **Gemini**: Streaming fixed for the Google GenAI SDK ([#1864](https://github.com/567-labs/instructor/pull/1864))
- **Gemini**: `HARM_CATEGORY_JAILBREAK` safety category and Anthropic `tool_result` content blocks now handled correctly ([#1867](https://github.com/567-labs/instructor/pull/1867))
- **Partial**: Fields with `default_factory` no longer retain the factory when made optional during streaming
- **OpenAI**: Dependency version constraint updated to support v2 ([#1858](https://github.com/567-labs/instructor/pull/1858))

---

## [1.12.0] - 2025-10-27

### Fixed
- **Python 3.13**: Compatibility issues and import path corrections in multimodal processing
- **Bedrock**: OpenAI-compatible models now correctly parse responses where reasoning appears before text content
- **Gemini**: `chunk.text ValueError` when `finish_reason=1` no longer crashes streaming
- **Gemini**: `thinking_config` no longer unintentionally passed to the tools helper
- **OpenAI**: `parse:error` hook now correctly fires for `InstructorValidationError`
- **JSON parsing**: Broken regex patterns removed from JSON extraction function
- **Cohere**: V2 API version detection improved ([#1844](https://github.com/567-labs/instructor/pull/1844))

---

## [1.11.3] - 2025-09-04

### Added
- **Hooks**: Hook combination via `__add__` / `combine()` — merge multiple hook handlers together
- **Hooks**: Per-call hooks — pass hooks directly to individual `.create()` calls without registering globally
- **Retry**: `InstructorRetryException` now tracks all failed attempts including exceptions and raw completions for better introspection
- **Docs**: `llms.txt` support via `mkdocs-llmstxt` plugin for AI/LLM consumers

### Fixed
- **`InstructorError.__str__()`**: Now correctly formats failed-attempt details
- **Retry**: Failed attempts propagated through reask handlers
- **Imports**: Backward compatibility imports restored for `function_calls` and `validators` modules

---

## [1.11.1] - 2025-08-27

### Changed
- Upgraded all dependencies to latest versions

---

## [1.11.0] - 2025-08-27

### Added
- **OpenRouter**: Provider support in `from_provider()` using `OPENROUTER_API_KEY`
- **LiteLLM**: Provider support in `from_provider()` ([#1723](https://github.com/567-labs/instructor/pull/1723))
- **xAI**: Provider utilities following standard provider structure ([#1728](https://github.com/567-labs/instructor/pull/1728))
- **Batch API**: In-memory batching support with improved error handling for OpenAI and Anthropic ([#1746](https://github.com/567-labs/instructor/pull/1746))
- **Hooks**: `completion:error` and `completion:last_attempt` hooks now fully implemented ([#1729](https://github.com/567-labs/instructor/pull/1729))

### Changed
- Codebase reorganized from flat structure to modular provider-based architecture ([#1730](https://github.com/567-labs/instructor/pull/1730))
- Provider-specific message conversion logic moved to dedicated handlers ([#1724](https://github.com/567-labs/instructor/pull/1724))

### Fixed
- Pydantic v2 deprecation warnings resolved by migrating from class `Config` to `ConfigDict` ([#1782](https://github.com/567-labs/instructor/pull/1782))

[Unreleased]: https://github.com/567-labs/instructor/compare/v1.17.1...HEAD
[1.17.1]: https://github.com/567-labs/instructor/compare/v1.17.0...v1.17.1
[1.17.0]: https://github.com/567-labs/instructor/compare/v1.16.0...v1.17.0
[1.16.0]: https://github.com/567-labs/instructor/compare/v1.15.4...v1.16.0
