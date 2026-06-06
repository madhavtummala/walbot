from __future__ import annotations

from importlib import import_module
from typing import Type

from .base import NotificationConnector


NOTIFICATION_CONNECTOR_REGISTRY: dict[str, str | Type[NotificationConnector]] = {
    "telegram": "src.notifications.providers.telegram:TelegramNotificationConnector",
}


def register_notification_connector(name: str, cls: Type[NotificationConnector]) -> None:
    normalized = str(name or "").strip().lower()
    if not normalized:
        raise ValueError("notification connector name is required")
    NOTIFICATION_CONNECTOR_REGISTRY[normalized] = cls


def _load_connector_class(path: str) -> Type[NotificationConnector]:
    module_path, _, class_name = path.partition(":")
    if not module_path or not class_name:
        raise ValueError(f"Invalid notification connector class path: {path}")
    module = import_module(module_path)
    cls = getattr(module, class_name)
    if not isinstance(cls, type) or not issubclass(cls, NotificationConnector):
        raise TypeError(f"{path} is not a NotificationConnector class")
    return cls


def get_notification_connector_class(name: str) -> Type[NotificationConnector]:
    normalized = str(name or "").strip().lower()
    entry = NOTIFICATION_CONNECTOR_REGISTRY.get(normalized)
    if entry is None:
        raise KeyError(f"Unknown notification connector: {name}")
    if isinstance(entry, str):
        cls = _load_connector_class(entry)
        NOTIFICATION_CONNECTOR_REGISTRY[normalized] = cls
        return cls
    return entry
