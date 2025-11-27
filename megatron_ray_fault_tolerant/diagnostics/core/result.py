"""DiagnosticResult with confidence-based routing."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any

from ..config import RootCause


class DiagnosticStatus(Enum):
    """Status of a diagnostic check."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"
    ERROR = "error"  # Diagnostic itself failed to run


@dataclass
class DiagnosticResult:
    """Result of a diagnostic check with confidence-based routing.

    The confidence score and confirmation count determine whether
    automatic remediation can proceed or human escalation is needed.
    """
    status: DiagnosticStatus
    diagnostic_name: str = ""

    # Root cause identification
    root_cause: Optional[RootCause] = None
    root_cause_details: str = ""

    # Suspect identification
    suspects: List[int] = field(default_factory=list)  # Rank IDs or machine IDs
    suspect_details: Dict[str, Any] = field(default_factory=dict)

    # Confidence-based routing
    confidence: float = 0.0  # 0.0 to 1.0
    requires_confirmation: bool = False
    confirmation_count: int = 0

    # Recommended action
    recommended_action: Optional[str] = None

    # Additional context
    message: str = ""
    artifacts: Dict[str, str] = field(default_factory=dict)  # name -> path
    metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: float = 0.0

    @property
    def action_threshold_met(self) -> bool:
        """Check if automatic action can proceed.

        High confidence (>= 0.9) allows immediate action.
        Otherwise, requires enough confirmations from multiple checks.
        """
        if self.confidence >= 0.9:
            return True
        return self.confirmation_count >= self._required_confirmations()

    def _required_confirmations(self) -> int:
        """Number of confirmations needed based on root cause type.

        Network issues are transient and need more confirmations.
        GPU issues are typically persistent and need fewer.
        """
        if self.root_cause in [RootCause.NIC_FLAP, RootCause.SWITCH_DOWN, RootCause.NETWORK]:
            return 2
        if self.root_cause in [RootCause.NCCL_TIMEOUT]:
            return 2
        return 1

    @property
    def is_critical(self) -> bool:
        """Check if this result indicates a critical failure."""
        return (
            self.status == DiagnosticStatus.FAILED
            and self.root_cause in [
                RootCause.GPU_ECC_ERROR,
                RootCause.GPU_FALLEN_OFF_BUS,
                RootCause.OS_KERNEL_PANIC,
                RootCause.SILENT_DATA_CORRUPTION,
            ]
        )

    @property
    def needs_escalation(self) -> bool:
        """Check if this result needs human escalation."""
        return (
            self.status == DiagnosticStatus.FAILED
            and not self.action_threshold_met
            and self.confidence < 0.5
        )

    def merge_with(self, other: "DiagnosticResult") -> "DiagnosticResult":
        """Merge two results, typically for confirmation counting.

        Used when the same diagnostic runs multiple times and we want
        to accumulate confidence/confirmation.
        """
        if other.root_cause != self.root_cause:
            return self  # Can't merge different root causes

        return DiagnosticResult(
            status=self.status,
            diagnostic_name=self.diagnostic_name,
            root_cause=self.root_cause,
            root_cause_details=self.root_cause_details or other.root_cause_details,
            suspects=list(set(self.suspects + other.suspects)),
            suspect_details={**self.suspect_details, **other.suspect_details},
            confidence=max(self.confidence, other.confidence),
            requires_confirmation=self.requires_confirmation or other.requires_confirmation,
            confirmation_count=self.confirmation_count + 1,
            recommended_action=self.recommended_action or other.recommended_action,
            message=self.message or other.message,
            artifacts={**self.artifacts, **other.artifacts},
            metrics={**self.metrics, **other.metrics},
            timestamp=max(self.timestamp, other.timestamp),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "status": self.status.value,
            "diagnostic_name": self.diagnostic_name,
            "root_cause": self.root_cause.value if self.root_cause else None,
            "root_cause_details": self.root_cause_details,
            "suspects": self.suspects,
            "suspect_details": self.suspect_details,
            "confidence": self.confidence,
            "requires_confirmation": self.requires_confirmation,
            "confirmation_count": self.confirmation_count,
            "recommended_action": self.recommended_action,
            "message": self.message,
            "artifacts": self.artifacts,
            "metrics": self.metrics,
            "timestamp": self.timestamp,
            "action_threshold_met": self.action_threshold_met,
            "is_critical": self.is_critical,
            "needs_escalation": self.needs_escalation,
        }

    @classmethod
    def passed(cls, diagnostic_name: str, message: str = "") -> "DiagnosticResult":
        """Factory for a passing result."""
        return cls(
            status=DiagnosticStatus.PASSED,
            diagnostic_name=diagnostic_name,
            confidence=1.0,
            message=message,
        )

    @classmethod
    def skipped(cls, diagnostic_name: str, reason: str = "") -> "DiagnosticResult":
        """Factory for a skipped result."""
        return cls(
            status=DiagnosticStatus.SKIPPED,
            diagnostic_name=diagnostic_name,
            message=reason,
        )

    @classmethod
    def error(cls, diagnostic_name: str, error_message: str) -> "DiagnosticResult":
        """Factory for an error result (diagnostic failed to run)."""
        return cls(
            status=DiagnosticStatus.ERROR,
            diagnostic_name=diagnostic_name,
            message=error_message,
        )
