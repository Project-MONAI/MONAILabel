class DomainError(Exception):
    """A failure that is safe to explain to a client."""

    def __init__(self, message: str, *, code: str = "invalid_request", status: int = 422):
        super().__init__(message)
        self.code = code
        self.status = status


class NotFound(DomainError):
    def __init__(self, kind: str, identifier: str):
        super().__init__(f"{kind} '{identifier}' was not found.", code="not_found", status=404)


class Conflict(DomainError):
    def __init__(self, message: str):
        super().__init__(message, code="conflict", status=409)


class Cancelled(Exception):
    """Cooperative cancellation; computed artifacts must not become visible records."""
