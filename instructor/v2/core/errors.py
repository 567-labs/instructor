from __future__ import annotations

from textwrap import dedent
from typing import Any, NamedTuple
from jinja2 import Template


class InstructorError(Exception):
    """Base exception for all Instructor-specific errors.

    This is the root exception class for the Instructor library. All custom
    exceptions in Instructor inherit from this class, allowing you to catch
    any Instructor-related error with a single except clause.

    Attributes:
        failed_attempts: Optional list of FailedAttempt objects tracking
            retry attempts that failed before this exception was raised.
            Each attempt includes the attempt number, exception, and
            partial completion data.

    Examples:
        Catch all Instructor errors:
        ```python
        try:
            response = client.chat.completions.create(...)
        except InstructorError as e:
            logger.error(f"Instructor error: {e}")
            # Handle any Instructor-specific error
        ```

        Create error from another exception:
        ```python
        try:
            # some operation
        except ValueError as e:
            raise InstructorError.from_exception(e)
        ```

    See Also:
        - FailedAttempt: NamedTuple containing retry attempt information
        - InstructorRetryException: Raised when retries are exhausted
    """

    failed_attempts: list[FailedAttempt] | None = None

    @classmethod
    def from_exception(
        cls, exception: Exception, failed_attempts: list[FailedAttempt] | None = None
    ):
        """Create an InstructorError from another exception.

        Args:
            exception: The original exception to wrap
            failed_attempts: Optional list of failed retry attempts

        Returns:
            A new instance of this exception class with the message from
            the original exception
        """
        return cls(str(exception), failed_attempts=failed_attempts)

    def __init__(
        self,
        *args: Any,
        failed_attempts: list[FailedAttempt] | None = None,
        **kwargs: Any,
    ):
        self.failed_attempts = failed_attempts
        super().__init__(*args, **kwargs)

    def __str__(self) -> str:
        # If no failed attempts, use the standard exception string representation
        if not self.failed_attempts:
            return super().__str__()

        template = Template(
            dedent(
                """
                <failed_attempts>
                {% for attempt in failed_attempts %}
                <generation number="{{ attempt.attempt_number }}">
                <exception>
                    {{ attempt.exception }}
                </exception>
                <completion>
                    {{ attempt.completion }}
                </completion>
                </generation>
                {% endfor %}
                </failed_attempts>

                <last_exception>
                    {{ last_exception }}
                </last_exception>
                """
            ).strip()
        )
        return template.render(
            last_exception=super().__str__(), failed_attempts=self.failed_attempts
        )


class FailedAttempt(NamedTuple):
    """Represents a single failed retry attempt.

    This immutable tuple stores information about a failed attempt during
    the retry process, allowing users to inspect what went wrong across
    multiple retry attempts.

    Attributes:
        attempt_number: The sequential number of this attempt (1-indexed)
        exception: The exception that caused this attempt to fail
        completion: Optional partial completion data from the LLM before
            the failure occurred. This can be useful for debugging or
            implementing custom recovery logic.

    Examples:
        ```python
        from instructor.v2.core.errors import InstructorRetryException

        try:
            response = client.chat.completions.create(...)
        except InstructorRetryException as e:
            for attempt in e.failed_attempts:
                print(f"Attempt {attempt.attempt_number} failed:")
                print(f"  Error: {attempt.exception}")
                print(f"  Partial data: {attempt.completion}")
        ```
    """

    attempt_number: int
    exception: Exception
    completion: Any | None = None


class IncompleteOutputException(InstructorError):
    """Exception raised when LLM output is truncated due to token limits.

    This exception occurs when the LLM hits the max_tokens limit before
    completing its response. This is particularly common with:
    - Large structured outputs
    - Very detailed responses
    - Low max_tokens settings

    Attributes:
        last_completion: The partial/incomplete response from the LLM
            before truncation occurred

    Common Solutions:
        - Increase max_tokens in your request
        - Simplify your response model
        - Use streaming with Partial models to get incomplete data
        - Break down complex extractions into smaller tasks

    Examples:
        ```python
        try:
            response = client.chat.completions.create(
                response_model=DetailedReport,
                max_tokens=100,  # Too low
                ...
            )
        except IncompleteOutputException as e:
            print(f"Output truncated. Partial data: {e.last_completion}")
            # Retry with higher max_tokens
            response = client.chat.completions.create(
                response_model=DetailedReport,
                max_tokens=2000,
                ...
            )
        ```

    See Also:
        - instructor.dsl.Partial: For handling partial/incomplete responses
    """

    def __init__(
        self,
        *args: Any,
        last_completion: Any | None = None,
        message: str = "The output is incomplete due to a max_tokens length limit.",
        **kwargs: Any,
    ):
        self.last_completion = last_completion
        super().__init__(message, *args, **kwargs)


class InstructorRetryException(InstructorError):
    """Exception raised when all retry attempts have been exhausted.

    This exception is raised after the maximum number of retries has been
    reached without successfully validating the LLM response. It contains
    detailed information about all failed attempts, making it useful for
    debugging and implementing custom recovery logic.

    Attributes:
        last_completion: The final (unsuccessful) completion from the LLM
        messages: The conversation history sent to the LLM (deprecated,
            use create_kwargs instead)
        n_attempts: The total number of attempts made
        total_usage: The cumulative token usage across all attempts
        create_kwargs: The parameters used in the create() call, including
            model, messages, temperature, etc.
        failed_attempts: List of FailedAttempt objects with details about
            each failed retry

    Common Causes:
        - Response model too strict for the LLM's capabilities
        - Ambiguous or contradictory requirements
        - LLM model not powerful enough for the task
        - Insufficient context or examples in the prompt

    Examples:
        ```python
        try:
            response = client.chat.completions.create(
                response_model=StrictModel,
                max_retries=3,
                ...
            )
        except InstructorRetryException as e:
            print(f"Failed after {e.n_attempts} attempts")
            print(f"Total tokens used: {e.total_usage}")
            print(f"Model used: {e.create_kwargs.get('model')}")

            # Inspect failed attempts
            for attempt in e.failed_attempts:
                print(f"Attempt {attempt.attempt_number}: {attempt.exception}")

            # Implement fallback strategy
            response = fallback_handler(e.last_completion)
        ```

    See Also:
        - FailedAttempt: Contains details about each retry attempt
        - ValidationError: Raised when response validation fails
    """

    def __init__(
        self,
        *args: Any,
        last_completion: Any | None = None,
        messages: list[Any] | None = None,
        n_attempts: int,
        total_usage: Any,
        create_kwargs: dict[str, Any] | None = None,
        failed_attempts: list[FailedAttempt] | None = None,
        **kwargs: Any,
    ):
        self.last_completion = last_completion
        self.messages = messages
        self.n_attempts = n_attempts
        self.total_usage = total_usage
        self.create_kwargs = create_kwargs
        super().__init__(*args, failed_attempts=failed_attempts, **kwargs)


class TokenBudgetError(InstructorRetryException):
    """Base class for retry termination caused by a token budget."""

    def __init__(
        self,
        *args: Any,
        budget: int,
        last_completion: Any | None = None,
        messages: list[Any] | None = None,
        n_attempts: int,
        total_usage: Any,
        create_kwargs: dict[str, Any] | None = None,
        failed_attempts: list[FailedAttempt] | None = None,
        **kwargs: Any,
    ):
        self.budget = budget
        super().__init__(
            *args,
            last_completion=last_completion,
            messages=messages,
            n_attempts=n_attempts,
            total_usage=total_usage,
            create_kwargs=create_kwargs,
            failed_attempts=failed_attempts,
            **kwargs,
        )


class TokenBudgetExceeded(TokenBudgetError):
    """Raised before a retry that would continue at an exhausted token budget."""


class TokenUsageUnavailableError(TokenBudgetError):
    """Raised when a retry budget cannot be enforced without usage metadata."""


class ValidationError(InstructorError):
    """Exception raised when LLM response validation fails.

    This exception occurs when the LLM's response doesn't meet the
    validation requirements defined in your Pydantic model, such as:
    - Field validation failures
    - Type mismatches
    - Custom validator failures
    - Missing required fields

    Note: This is distinct from Pydantic's ValidationError and provides
    Instructor-specific context through the failed_attempts attribute.

    Examples:
        ```python
        from pydantic import BaseModel, field_validator

        class User(BaseModel):
            age: int

            @field_validator('age')
            def age_must_be_positive(cls, v):
                if v < 0:
                    raise ValueError('Age must be positive')
                return v

        try:
            response = client.chat.completions.create(
                response_model=User,
                ...
            )
        except ValidationError as e:
            print(f"Validation failed: {e}")
            # Validation errors are automatically retried
        ```

    See Also:
        - InstructorRetryException: Raised when validation fails repeatedly
    """


class ProviderError(InstructorError):
    """Exception raised for provider-specific errors.

    This exception is used to wrap errors specific to LLM providers
    (OpenAI, Anthropic, etc.) and provides context about which provider
    caused the error.

    Attributes:
        provider: The name of the provider that raised the error
            (e.g., "openai", "anthropic", "gemini")

    Common Causes:
        - API authentication failures
        - Rate limiting
        - Invalid model names
        - Provider-specific API errors
        - Network connectivity issues

    Examples:
        ```python
        try:
            client = instructor.from_openai(openai_client)
            response = client.chat.completions.create(...)
        except ProviderError as e:
            print(f"Provider {e.provider} error: {e}")
            # Implement provider-specific error handling
            if e.provider == "openai":
                # Handle OpenAI-specific errors
                pass
        ```
    """

    def __init__(self, provider: str, message: str, *args: Any, **kwargs: Any):
        self.provider = provider
        super().__init__(f"{provider}: {message}", *args, **kwargs)


class ConfigurationError(InstructorError):
    """Exception raised for configuration-related errors.

    This exception occurs when there are issues with how Instructor
    is configured or initialized, such as:
    - Missing required dependencies
    - Invalid parameters
    - Incompatible settings
    - Improper client initialization

    Common Scenarios:
        - Missing provider SDK (e.g., anthropic package not installed)
        - Invalid model string format in from_provider()
        - Incompatible parameter combinations
        - Invalid max_retries configuration

    Examples:
        ```python
        try:
            # Missing provider SDK
            client = instructor.from_provider("anthropic/claude-3")
        except ConfigurationError as e:
            print(f"Configuration issue: {e}")
            # e.g., "The anthropic package is required..."

        try:
            # Invalid model string
            client = instructor.from_provider("invalid-format")
        except ConfigurationError as e:
            print(f"Configuration issue: {e}")
            # e.g., "Model string must be in format 'provider/model-name'"
        ```
    """


class ModeError(InstructorError):
    """Exception raised when an invalid mode is used for a provider.

    Different LLM providers support different modes (e.g., TOOLS, JSON,
    FUNCTIONS). This exception is raised when you try to use a mode that
    isn't supported by the current provider.

    Attributes:
        mode: The invalid mode that was attempted
        provider: The provider name
        valid_modes: List of modes supported by this provider

    Examples:
        ```python
        try:
            client = instructor.from_openai(
                openai_client,
                mode=instructor.Mode.ANTHROPIC_TOOLS  # Wrong for OpenAI
            )
        except ModeError as e:
            print(f"Invalid mode '{e.mode}' for {e.provider}")
            print(f"Use one of: {', '.join(e.valid_modes)}")
            # Retry with valid mode
            client = instructor.from_openai(
                openai_client,
                mode=instructor.Mode.TOOLS
            )
        ```

    See Also:
        - instructor.Mode: Enum of all available modes
    """

    def __init__(
        self,
        mode: str,
        provider: str,
        valid_modes: list[str],
        *args: Any,
        **kwargs: Any,
    ):
        self.mode = mode
        self.provider = provider
        self.valid_modes = valid_modes
        message = f"Invalid mode '{mode}' for provider '{provider}'. Valid modes: {', '.join(valid_modes)}"
        super().__init__(message, *args, **kwargs)


class ClientError(InstructorError):
    """Exception raised for client initialization or usage errors.

    This exception covers errors related to improper client usage or
    initialization that don't fit other categories.

    Common Scenarios:
        - Passing invalid client object to from_* functions
        - Missing required client configuration
        - Attempting operations on improperly initialized clients

    Examples:
        ```python
        try:
            # Invalid client type
            client = instructor.from_openai("not_a_client")
        except ClientError as e:
            print(f"Client error: {e}")
        ```
    """


class AsyncValidationError(ValueError, InstructorError):
    """Exception raised during async validation.

    This exception is used specifically for errors that occur during
    asynchronous validation operations. It inherits from both ValueError
    and InstructorError to maintain compatibility with existing code.

    Attributes:
        errors: List of ValueError instances from failed validations

    Examples:
        ```python
        from instructor.v2.validation import async_field_validator

        class Model(BaseModel):
            urls: list[str]

            @async_field_validator('urls')
            async def validate_urls(cls, v):
                # Async validation logic
                ...

        try:
            response = await client.chat.completions.create(
                response_model=Model,
                ...
            )
        except AsyncValidationError as e:
            print(f"Async validation failed: {e.errors}")
        ```
    """

    def __init__(
        self,
        message: str,
        *args: Any,
        errors: list[ValueError] | None = None,
        **kwargs: Any,
    ):
        self.errors = errors if errors is not None else []
        super().__init__(message, *args, **kwargs)


class ResponseParsingError(ValueError, InstructorError):
    """Exception raised when unable to parse the LLM response.

    This exception occurs when the LLM's raw response cannot be parsed
    into the expected format. Common scenarios include:
    - Malformed JSON in JSON mode
    - Missing required fields in the response
    - Unexpected response structure
    - Invalid tool call format

    Note: This exception inherits from both ValueError and InstructorError
    to maintain backwards compatibility with code that catches ValueError.

    Attributes:
        mode: The mode being used when parsing failed
        raw_response: The raw response that failed to parse (if available)

    Examples:
        ```python
        try:
            response = client.chat.completions.create(
                response_model=User,
                mode=instructor.Mode.JSON,
                ...
            )
        except ResponseParsingError as e:
            print(f"Failed to parse response in {e.mode} mode")
            print(f"Raw response: {e.raw_response}")
            # May indicate the model doesn't support this mode well
        ```

        Backwards compatible with ValueError:
        ```python
        try:
            response = client.chat.completions.create(...)
        except ValueError as e:
            # Still catches ResponseParsingError
            print(f"Parsing error: {e}")
        ```
    """

    def __init__(
        self,
        message: str,
        *args: Any,
        mode: str | None = None,
        raw_response: Any | None = None,
        **kwargs: Any,
    ):
        self.mode = mode
        self.raw_response = raw_response
        context = f" (mode: {mode})" if mode else ""
        super().__init__(f"{message}{context}", *args, **kwargs)


class MultimodalError(ValueError, InstructorError):
    """Exception raised for multimodal content processing errors.

    This exception is raised when there are issues processing multimodal
    content (images, audio, PDFs, etc.), such as:
    - Unsupported file formats
    - File not found
    - Invalid base64 encoding
    - Provider doesn't support multimodal content

    Note: This exception inherits from both ValueError and InstructorError
    to maintain backwards compatibility with code that catches ValueError.

    Attributes:
        content_type: The type of content that failed (e.g., 'image', 'audio', 'pdf')
        file_path: The file path if applicable

    Examples:
        ```python
        from instructor import Image

        try:
            response = client.chat.completions.create(
                response_model=Analysis,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze this image"},
                        Image.from_path("/invalid/path.jpg")
                    ]
                }]
            )
        except MultimodalError as e:
            print(f"Multimodal error with {e.content_type}: {e}")
            if e.file_path:
                print(f"File path: {e.file_path}")
        ```

        Backwards compatible with ValueError:
        ```python
        try:
            img = Image.from_path("/path/to/image.jpg")
        except ValueError as e:
            # Still catches MultimodalError
            print(f"Image error: {e}")
        ```
    """

    def __init__(
        self,
        message: str,
        *args: Any,
        content_type: str | None = None,
        file_path: str | None = None,
        **kwargs: Any,
    ):
        self.content_type = content_type
        self.file_path = file_path
        context_parts = []
        if content_type:
            context_parts.append(f"content_type: {content_type}")
        if file_path:
            context_parts.append(f"file: {file_path}")
        context = f" ({', '.join(context_parts)})" if context_parts else ""
        super().__init__(f"{message}{context}", *args, **kwargs)
