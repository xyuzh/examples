"""Integration module for connecting diagnostics to MegatronActor."""

from .actor_mixin import DiagnosticsMixin
from .hooks import DiagnosticsHooks

__all__ = ["DiagnosticsMixin", "DiagnosticsHooks"]
