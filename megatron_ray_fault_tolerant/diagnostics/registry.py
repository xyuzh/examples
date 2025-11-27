"""DiagnosticRegistry - plugin pattern for registering diagnostics."""

from typing import Dict, Type, List, Optional
from loguru import logger

from .core.base import BaseDiagnostic, DiagnosticPhase
from .config import DiagnosticsConfig


class DiagnosticRegistry:
    """Registry for diagnostic components.

    Follows the same pattern as DispatchRegistry in dispatch.py.
    Diagnostics can be registered and then retrieved by name or phase.

    Example:
        # Register a diagnostic
        DiagnosticRegistry.register(LogAnalyzer)

        # Get all diagnostics for a phase
        diagnostics = DiagnosticRegistry.get_for_phase(DiagnosticPhase.ON_FAILURE)

        # Get a specific diagnostic
        log_analyzer = DiagnosticRegistry.get("log_analyzer")
    """

    _registry: Dict[str, Type[BaseDiagnostic]] = {}

    @classmethod
    def register(cls, diagnostic_class: Type[BaseDiagnostic]) -> None:
        """Register a diagnostic class.

        Args:
            diagnostic_class: A class that extends BaseDiagnostic
        """
        if not issubclass(diagnostic_class, BaseDiagnostic):
            raise TypeError(f"{diagnostic_class} must be a subclass of BaseDiagnostic")

        name = diagnostic_class.name
        if not name:
            raise ValueError(f"{diagnostic_class.__name__} must define a 'name' attribute")

        if name in cls._registry:
            logger.warning(f"Overwriting existing diagnostic: {name}")

        cls._registry[name] = diagnostic_class
        logger.debug(f"Registered diagnostic: {name}")

    @classmethod
    def unregister(cls, name: str) -> None:
        """Unregister a diagnostic by name."""
        if name in cls._registry:
            del cls._registry[name]
            logger.debug(f"Unregistered diagnostic: {name}")

    @classmethod
    def get(cls, name: str) -> Type[BaseDiagnostic]:
        """Get a diagnostic class by name.

        Args:
            name: The diagnostic name

        Returns:
            The diagnostic class

        Raises:
            KeyError: If no diagnostic with that name is registered
        """
        if name not in cls._registry:
            raise KeyError(f"Diagnostic '{name}' not registered. Available: {list(cls._registry.keys())}")
        return cls._registry[name]

    @classmethod
    def get_for_phase(
        cls,
        phase: DiagnosticPhase,
        config: Optional[DiagnosticsConfig] = None,
    ) -> List[Type[BaseDiagnostic]]:
        """Get all diagnostic classes that run in a given phase.

        Args:
            phase: The diagnostic phase
            config: Optional config to filter by enabled status

        Returns:
            List of diagnostic classes, sorted by priority
        """
        diagnostics = []
        for diagnostic_class in cls._registry.values():
            if phase in diagnostic_class.phases:
                if config is not None:
                    # Instantiate temporarily to check if enabled
                    instance = diagnostic_class()
                    if not instance.is_enabled(config):
                        continue
                diagnostics.append(diagnostic_class)

        # Sort by priority (lower = higher priority)
        diagnostics.sort(key=lambda d: d.priority)
        return diagnostics

    @classmethod
    def list_registered(cls) -> Dict[str, Type[BaseDiagnostic]]:
        """List all registered diagnostics."""
        return cls._registry.copy()

    @classmethod
    def list_names(cls) -> List[str]:
        """List all registered diagnostic names."""
        return list(cls._registry.keys())

    @classmethod
    def clear(cls) -> None:
        """Clear all registered diagnostics. Useful for testing."""
        cls._registry.clear()


def register_diagnostic(diagnostic_class: Type[BaseDiagnostic]) -> Type[BaseDiagnostic]:
    """Decorator to register a diagnostic class.

    Example:
        @register_diagnostic
        class MyDiagnostic(BaseDiagnostic):
            name = "my_diagnostic"
            phases = [DiagnosticPhase.ON_FAILURE]
            ...
    """
    DiagnosticRegistry.register(diagnostic_class)
    return diagnostic_class
