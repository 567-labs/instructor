from __future__ import annotations
from enum import Enum
from collections import defaultdict
from typing import Any, Literal, TypeVar, Protocol, Union

import inspect
import traceback
import warnings

T = TypeVar("T")


class HookName(Enum):
    COMPLETION_KWARGS = "completion:kwargs"
    COMPLETION_RESPONSE = "completion:response"
    COMPLETION_ERROR = "completion:error"
    COMPLETION_LAST_ATTEMPT = "completion:last_attempt"
    COMPLETION_USAGE = "completion:usage"
    PARSE_ERROR = "parse:error"


# Handler protocol types for type safety
class CompletionKwargsHandler(Protocol):
    """Protocol for completion kwargs handlers."""

    def __call__(self, *args: Any, **kwargs: Any) -> None: ...


class CompletionResponseHandler(Protocol):
    """Protocol for completion response handlers."""

    def __call__(self, response: Any) -> None: ...


class CompletionErrorHandler(Protocol):
    """Protocol for completion error and last attempt handlers."""

    def __call__(
        self,
        error: Exception,
        *,
        attempt_number: int = ...,
        max_attempts: int | None = ...,
        is_last_attempt: bool = ...,
    ) -> None: ...


class CompletionUsageHandler(Protocol):
    """Protocol for cumulative completion-usage handlers."""

    def __call__(
        self,
        usage: Any,
        *,
        attempt_number: int = ...,
    ) -> None: ...


class ParseErrorHandler(Protocol):
    """Protocol for parse error handlers."""

    def __call__(self, error: Exception, **kwargs: Any) -> None: ...


# Type alias for hook name parameter
HookNameType = Union[
    HookName,
    Literal[
        "completion:kwargs",
        "completion:response",
        "completion:error",
        "completion:last_attempt",
        "completion:usage",
        "parse:error",
    ],
]

# Type alias for all handler types
HandlerType = Union[
    CompletionKwargsHandler,
    CompletionResponseHandler,
    CompletionErrorHandler,
    CompletionUsageHandler,
    ParseErrorHandler,
]


class Hooks:
    """
    Hooks class for handling and emitting events related to completion processes.

    This class provides a mechanism to register event handlers and emit events
    for various stages of the completion process.
    """

    def __init__(self) -> None:
        """Initialize the hooks container."""
        self._handlers: defaultdict[HookName, list[HandlerType]] = defaultdict(list)

    def on(
        self,
        hook_name: HookNameType,
        handler: HandlerType,
    ) -> None:
        """
        Register an event handler for a specific event.

        This method allows you to attach a handler function to a specific event.
        When the event is emitted, all registered handlers for that event will be called.

        Args:
            hook_name: The event to listen for. This can be either a HookName enum
                       value or a string representation of the event name.
            handler: The function to be called when the event is emitted.

        Raises:
            ValueError: If the hook_name is not a valid HookName enum or string representation.

        Example:
            >>> def on_completion_kwargs(*args: Any, **kwargs: Any) -> None:
            ...     print(f"Completion kwargs: {args}, {kwargs}")
            >>> hooks = Hooks()
            >>> hooks.on(HookName.COMPLETION_KWARGS, on_completion_kwargs)
            >>> hooks.emit_completion_arguments(model="gpt-3.5-turbo", temperature=0.7)
            Completion kwargs: (), {'model': 'gpt-3.5-turbo', 'temperature': 0.7}
        """
        hook_name = self.get_hook_name(hook_name)
        self._handlers[hook_name].append(handler)

    def get_hook_name(self, hook_name: HookNameType) -> HookName:
        """
        Convert a string hook name to its corresponding enum value.

        Args:
            hook_name: Either a HookName enum value or string representation.

        Returns:
            The corresponding HookName enum value.

        Raises:
            ValueError: If the string doesn't match any HookName enum value.
        """
        if isinstance(hook_name, str):
            try:
                return HookName(hook_name)
            except ValueError as err:
                raise ValueError(f"Invalid hook name: {hook_name}") from err
        return hook_name

    def emit(self, hook_name: HookName, *args: Any, **kwargs: Any) -> None:
        """
        Generic method to emit events for any hook type.

        Handler additions and removals take effect on the next emission.

        Args:
            hook_name: The hook to emit
            *args: Positional arguments to pass to handlers
            **kwargs: Keyword arguments to pass to handlers
        """
        for handler in self._handlers[hook_name].copy():
            try:
                if kwargs:
                    try:
                        sig = inspect.signature(handler)
                        sig.bind(*args, **kwargs)
                    except TypeError:
                        handler(*args)
                        continue
                handler(*args, **kwargs)
            except Exception:
                error_traceback = traceback.format_exc()
                warnings.warn(
                    f"Error in {hook_name.value} handler:\n{error_traceback}",
                    stacklevel=2,
                )

    def emit_completion_arguments(self, *args: Any, **kwargs: Any) -> None:
        """
        Emit a completion arguments event.

        Args:
            *args: Positional arguments to pass to handlers
            **kwargs: Keyword arguments to pass to handlers
        """
        self.emit(HookName.COMPLETION_KWARGS, *args, **kwargs)

    def emit_completion_response(self, response: Any) -> None:
        """
        Emit a completion response event.

        Args:
            response: The completion response to pass to handlers
        """
        self.emit(HookName.COMPLETION_RESPONSE, response)

    def emit_completion_error(self, error: Exception, **kwargs: Any) -> None:
        """
        Emit a completion error event.

        Args:
            error: The exception to pass to handlers
            **kwargs: Optional metadata (attempt_number, max_attempts, is_last_attempt)
        """
        self.emit(HookName.COMPLETION_ERROR, error, **kwargs)

    def emit_completion_last_attempt(self, error: Exception, **kwargs: Any) -> None:
        """
        Emit a completion last attempt event.

        Args:
            error: The exception to pass to handlers
            **kwargs: Optional metadata (attempt_number, max_attempts, is_last_attempt)
        """
        self.emit(HookName.COMPLETION_LAST_ATTEMPT, error, **kwargs)

    def emit_completion_usage(self, usage: Any, **kwargs: Any) -> None:
        """Emit an immutable snapshot of cumulative completion usage."""
        self.emit(HookName.COMPLETION_USAGE, usage, **kwargs)

    def emit_parse_error(self, error: Exception, **kwargs: Any) -> None:
        """
        Emit a parse error event.

        Args:
            error: The exception to pass to handlers
            **kwargs: Optional metadata (attempt_number, max_attempts, is_last_attempt)
        """
        self.emit(HookName.PARSE_ERROR, error, **kwargs)

    def off(
        self,
        hook_name: HookNameType,
        handler: HandlerType,
    ) -> None:
        """
        Remove a specific handler from an event.

        Args:
            hook_name: The name of the hook.
            handler: The handler to remove.
        """
        hook_name = self.get_hook_name(hook_name)
        if hook_name in self._handlers:
            if handler in self._handlers[hook_name]:
                self._handlers[hook_name].remove(handler)
                if not self._handlers[hook_name]:
                    del self._handlers[hook_name]

    def clear(
        self,
        hook_name: HookNameType | None = None,
    ) -> None:
        """
        Clear handlers for a specific event or all events.

        Args:
            hook_name: The name of the event to clear handlers for.
                      If None, all handlers are cleared.
        """
        if hook_name is not None:
            hook_name = self.get_hook_name(hook_name)
            self._handlers.pop(hook_name, None)
        else:
            self._handlers.clear()

    def __add__(self, other: Hooks) -> Hooks:
        """
        Combine two Hooks instances into a new one.

        This creates a new Hooks instance that contains all handlers from both
        the current instance and the other instance. Handlers are combined by
        appending the other's handlers after the current instance's handlers.

        Args:
            other: Another Hooks instance to combine with this one.

        Returns:
            A new Hooks instance containing all handlers from both instances.

        Example:
            >>> hooks1 = Hooks()
            >>> hooks2 = Hooks()
            >>> hooks1.on("completion:kwargs", lambda **kw: print("Hook 1"))
            >>> hooks2.on("completion:kwargs", lambda **kw: print("Hook 2"))
            >>> combined = hooks1 + hooks2
            >>> combined.emit_completion_arguments()  # Prints both "Hook 1" and "Hook 2"
        """
        if not isinstance(other, Hooks):
            return NotImplemented

        combined = Hooks()

        # Copy handlers from self
        for hook_name, handlers in self._handlers.items():
            combined._handlers[hook_name].extend(handlers.copy())

        # Add handlers from other
        for hook_name, handlers in other._handlers.items():
            combined._handlers[hook_name].extend(handlers.copy())

        return combined

    def __iadd__(self, other: Hooks) -> Hooks:
        """
        Add another Hooks instance to this one in-place.

        This modifies the current instance by adding all handlers from the other
        instance. The other instance's handlers are appended after the current
        instance's handlers for each event type.

        Args:
            other: Another Hooks instance to add to this one.

        Returns:
            This Hooks instance (for method chaining).

        Example:
            >>> hooks1 = Hooks()
            >>> hooks2 = Hooks()
            >>> hooks1.on("completion:kwargs", lambda **kw: print("Hook 1"))
            >>> hooks2.on("completion:kwargs", lambda **kw: print("Hook 2"))
            >>> hooks1 += hooks2
            >>> hooks1.emit_completion_arguments()  # Prints both "Hook 1" and "Hook 2"
        """
        if not isinstance(other, Hooks):
            return NotImplemented

        # Add handlers from other to self
        for hook_name, handlers in other._handlers.items():
            self._handlers[hook_name].extend(handlers.copy())

        return self

    @classmethod
    def combine(cls, *hooks_instances: Hooks) -> Hooks:
        """
        Combine multiple Hooks instances into a new one.

        This class method creates a new Hooks instance that contains all handlers
        from all provided instances. Handlers are combined in the order of the
        provided instances.

        Args:
            *hooks_instances: Variable number of Hooks instances to combine.

        Returns:
            A new Hooks instance containing all handlers from all instances.

        Example:
            >>> hooks1 = Hooks()
            >>> hooks2 = Hooks()
            >>> hooks3 = Hooks()
            >>> hooks1.on("completion:kwargs", lambda **kw: print("Hook 1"))
            >>> hooks2.on("completion:kwargs", lambda **kw: print("Hook 2"))
            >>> hooks3.on("completion:kwargs", lambda **kw: print("Hook 3"))
            >>> combined = Hooks.combine(hooks1, hooks2, hooks3)
            >>> combined.emit_completion_arguments()  # Prints all three hooks
        """
        combined = cls()

        for hooks_instance in hooks_instances:
            if not isinstance(hooks_instance, cls):
                raise TypeError(f"Expected Hooks instance, got {type(hooks_instance)}")
            combined += hooks_instance

        return combined

    def copy(self) -> Hooks:
        """
        Create a deep copy of this Hooks instance.

        Returns:
            A new Hooks instance with all the same handlers.

        Example:
            >>> original = Hooks()
            >>> original.on("completion:kwargs", lambda **kw: print("Hook"))
            >>> copy = original.copy()
            >>> copy.emit_completion_arguments()  # Prints "Hook"
        """
        new_hooks = Hooks()
        for hook_name, handlers in self._handlers.items():
            new_hooks._handlers[hook_name].extend(handlers.copy())
        return new_hooks
