"""Exception hierarchy for Facebin.

Every error raised by Facebin derives from :class:`FacebinError`, so callers
can catch a single type at process boundaries.  Errors carry an optional
``hint`` with a concrete suggestion for fixing the problem; the hint is
appended to the message when the exception is printed.
"""


class FacebinError(Exception):
    """Base class for all Facebin errors."""

    def __init__(self, message: str, *, hint: str = None):
        self.message = message
        self.hint = hint
        super().__init__(message)

    def __str__(self):
        if self.hint:
            return "{}\nHint: {}".format(self.message, self.hint)
        return self.message


class ConfigError(FacebinError):
    """The configuration file is missing, unreadable, or invalid."""


class DatabaseError(FacebinError):
    """A SQLite operation failed."""


class RedisConnectionError(FacebinError):
    """The Redis server cannot be reached."""


class CameraError(FacebinError):
    """A camera device cannot be opened or is misconfigured."""


class ModelFileError(FacebinError):
    """A machine-learning model file is missing or unreadable."""


class DependencyError(FacebinError):
    """An optional dependency needed for the requested feature is missing."""
