"""Typed provider errors preserved in benchmark records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelRequestError(RuntimeError):
    code: str
    message: str
    provider: str
    model: str
    attempts: int
    latency_seconds: float
    status_code: int | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        RuntimeError.__init__(self, self.message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_type": self.code,
            "error": self.message,
            "provider": self.provider,
            "model": self.model,
            "request_attempts": self.attempts,
            "latency_seconds": self.latency_seconds,
            "status_code": self.status_code,
            "details": self.details,
        }
