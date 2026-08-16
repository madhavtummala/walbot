"""One way to register a pluggable implementation.

This project has six extension points -- algorithms, brokerages, notification connectors,
market-data providers, quote providers, dividend providers -- and until now they were
registered three different ways: bare functions in a dict, eagerly-imported classes in a dict,
and lazily-imported ``"module:Class"`` strings behind a ``register()`` helper. The last of
those was the only one that let a provider be added without importing it at startup or editing
the module that dispatches to it, so it is the one that survives here.

Lazy resolution is the point, not an optimisation. A registry entry that imports its module at
definition time drags every provider's third-party dependency into every process that touches
the registry -- and makes a broken or unconfigured provider an import error for the whole
application rather than a failure of the one thing that needed it.
"""

from __future__ import annotations

from importlib import import_module
from typing import Generic, Iterator, Type, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Name -> implementation, resolved on first use.

    Entries are either a class or a ``"module.path:ClassName"`` string. A string is imported,
    type-checked against ``base`` and then cached in place, so the import cost and the
    validation are paid once.
    """

    def __init__(self, label: str, base: Type[T], entries: dict[str, str | Type[T]] | None = None):
        #: Used in error messages, so they name the thing that is missing rather than "entry".
        self.label = label
        self.base = base
        self._entries: dict[str, str | Type[T]] = dict(entries or {})
        #: Alternative spellings and retired names. Kept beside the entries rather than in the
        #: caller because a rename that is not aliased silently resolves to nothing.
        self._aliases: dict[str, str] = {}

    # -- registration ---------------------------------------------------------------------

    def alias(self, alias: str, target: str) -> None:
        self._aliases[self.normalize(alias)] = self.normalize(target)

    # -- lookup ---------------------------------------------------------------------------

    def normalize(self, name: str) -> str:
        return str(name or "").strip().lower()

    def canonical(self, name: str) -> str:
        """Resolve aliases to the registered name."""
        normalized = self.normalize(name)
        return self._aliases.get(normalized, normalized)

    def __contains__(self, name: object) -> bool:
        return self.canonical(str(name)) in self._entries

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._entries))

    def __len__(self) -> int:
        return len(self._entries)

    def names(self) -> list[str]:
        return sorted(self._entries)

    def get(self, name: str) -> Type[T]:
        """The class registered under ``name``, importing it if it is still a path."""
        canonical = self.canonical(name)
        entry = self._entries.get(canonical)
        if entry is None:
            known = ", ".join(self.names()) or "none registered"
            raise KeyError(f"Unknown {self.label}: {name!r}; known: {known}")
        if isinstance(entry, str):
            entry = self._load(entry)
            self._entries[canonical] = entry
        return entry

    def create(self, name: str, *args, **kwargs) -> T:
        """Instantiate the registered class. The common case, so it earns a method."""
        return self.get(name)(*args, **kwargs)

    def _load(self, path: str) -> Type[T]:
        module_path, _, class_name = path.partition(":")
        if not module_path or not class_name:
            raise ValueError(f"Invalid {self.label} path {path!r}; expected 'module:ClassName'")
        cls = getattr(import_module(module_path), class_name)
        if not isinstance(cls, type) or not issubclass(cls, self.base):
            raise TypeError(f"{path} is not a {self.base.__name__}")
        return cls
