from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class NotificationMessage:
    text: str
    subject: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class NotificationConnector(ABC):
    provider_name: str = ""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    @abstractmethod
    def send(self, message: NotificationMessage) -> bool:
        pass
