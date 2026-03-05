"""Domain models - immutable data structures."""

from dataclasses import dataclass
from typing import Generic, TypeVar
from enum import Enum


class MessageRole(str, Enum):
    """Message roles in conversation."""
    USER = "user"
    AGENT = "agent"


@dataclass(frozen=True)
class DomainMessage:
    """Pure domain representation of an A2A message."""
    text: str
    role: MessageRole
    has_file_parts: bool = False
    has_data_parts: bool = False

    @property
    def is_text_only(self) -> bool:
        """Check if message contains only text."""
        return not (self.has_file_parts or self.has_data_parts)


@dataclass(frozen=True)
class GradientInput:
    """Input format expected by Gradient agents."""
    prompt: str
    session_id: str | None = None


@dataclass(frozen=True)
class GradientOutput:
    """Output format returned by Gradient agents."""
    text: str
    metadata: dict | None = None


@dataclass(frozen=True)
class ValidationError:
    """Domain validation error."""
    message: str
    code: str = "VALIDATION_ERROR"


@dataclass(frozen=True)
class TransformationError:
    """Domain transformation error."""
    message: str
    code: str = "TRANSFORMATION_ERROR"


# Result type for functional error handling
T = TypeVar('T')
E = TypeVar('E')


@dataclass(frozen=True)
class Ok(Generic[T]):
    """Success result."""
    value: T

    def is_ok(self) -> bool:
        return True

    def is_err(self) -> bool:
        return False


@dataclass(frozen=True)
class Err(Generic[E]):
    """Error result."""
    error: E

    def is_ok(self) -> bool:
        return False

    def is_err(self) -> bool:
        return True


Result = Ok[T] | Err[E]
