"""Errors raised at the isolated parser worker boundary."""


class ParserWorkerError(RuntimeError):
    """Base error for worker startup, protocol, and parsing failures."""


class ParserWorkerUnavailable(ParserWorkerError):
    """Docker or the configured parser worker image is unavailable."""


class ParserWorkerTimeout(ParserWorkerError):
    """A PDF parse exceeded the configured wall-clock limit."""


class ParserWorkerProtocolError(ParserWorkerError):
    """The worker returned an invalid or inconsistent response."""


class ParserWorkerParseError(ParserWorkerError):
    """The worker could not complete document parsing."""
