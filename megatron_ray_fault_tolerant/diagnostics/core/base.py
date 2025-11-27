"""Base class for all diagnostic components."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from .context import DiagnosticContext
    from .result import DiagnosticResult
    from ..config import DiagnosticsConfig


class DiagnosticPhase(Enum):
    """When a diagnostic should run."""
    PRE_TRAINING = "pre_training"    # Before training starts
    POST_STEP = "post_step"          # After each training step (lightweight only)
    ON_FAILURE = "on_failure"        # When a failure is detected
    ON_RECOVERY = "on_recovery"      # After recovery from failure
    ON_DEMAND = "on_demand"          # Manually triggered
    PERIODIC = "periodic"            # Run at regular intervals


class BaseDiagnostic(ABC):
    """Abstract base class for all diagnostic components.

    Each diagnostic must:
    1. Define a unique name
    2. Specify which phases it runs in
    3. Implement the check() method
    4. Implement is_enabled() to check config

    Example:
        class MyDiagnostic(BaseDiagnostic):
            name = "my_diagnostic"
            phases = [DiagnosticPhase.ON_FAILURE]

            def check(self, context: DiagnosticContext) -> DiagnosticResult:
                # ... perform check ...
                return DiagnosticResult.passed(self.name)

            def is_enabled(self, config: DiagnosticsConfig) -> bool:
                return config.my_diagnostic.enabled
    """

    # Subclasses must define these
    name: str = ""
    phases: List[DiagnosticPhase] = []

    # Priority (lower = higher priority, runs first)
    priority: int = 100

    @abstractmethod
    def check(self, context: "DiagnosticContext") -> "DiagnosticResult":
        """Perform the diagnostic check.

        Args:
            context: Runtime context with rank, step, and artifact info

        Returns:
            DiagnosticResult with status, root cause, confidence, etc.
        """
        pass

    @abstractmethod
    def is_enabled(self, config: "DiagnosticsConfig") -> bool:
        """Check if this diagnostic is enabled in the config.

        Args:
            config: The diagnostics configuration

        Returns:
            True if enabled, False otherwise
        """
        pass

    def should_run_in_phase(self, phase: DiagnosticPhase) -> bool:
        """Check if this diagnostic should run in the given phase."""
        return phase in self.phases

    def initialize(self, config: "DiagnosticsConfig") -> None:
        """Optional initialization hook called once at startup.

        Override this to perform any one-time setup.
        """
        pass

    def cleanup(self) -> None:
        """Optional cleanup hook called at shutdown.

        Override this to clean up resources.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, phases={self.phases})"
