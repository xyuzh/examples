"""Log Analyzer - P0 diagnostic for parsing logs and classifying failures.

Covers ~60% of failures by pattern matching against known error signatures.
"""

import re
import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum

from ..core.base import BaseDiagnostic, DiagnosticPhase
from ..core.context import DiagnosticContext
from ..core.result import DiagnosticResult, DiagnosticStatus
from ..config import DiagnosticsConfig, RootCause, LogAnalyzerConfig
from ..registry import register_diagnostic


class FailureCategory(Enum):
    """Failure categories from ByteRobust production data."""
    # GPU/NCCL (15%)
    NCCL_TIMEOUT = "nccl_timeout"
    GPU_ERROR = "gpu_error"

    # Host resources (26%)
    CPU_OVERLOAD = "cpu_overload"      # 11%
    CPU_OOM = "cpu_oom"                # 10.1%
    INSUFFICIENT_DISK = "disk_space"   # 5%

    # Network (8%)
    NETWORK = "network"
    INFINIBAND_ERROR = "ib_error"      # 2.9%

    # Infrastructure (6%)
    FILESYSTEM_MOUNT = "fs_mount"      # 2.1%
    HDFS_ERROR = "hdfs_error"          # 2%
    CONTAINER_ERROR = "container"      # 1.4%
    OS_KERNEL_PANIC = "kernel_panic"   # 0.4%
    EXTERNAL_SERVICE = "external_svc"  # 0.2%

    # Implicit (1%)
    MFU_DECLINE = "mfu_decline"
    NAN_VALUE = "nan_value"
    HANG = "hang"


@dataclass
class LogPattern:
    """A pattern to match against log messages."""
    pattern: re.Pattern
    category: FailureCategory
    root_cause: RootCause
    confidence: float
    description: str


@dataclass
class LogEntry:
    """A captured log entry."""
    timestamp: float
    level: int
    message: str
    logger_name: str
    source_file: Optional[str] = None
    line_number: Optional[int] = None


# Compiled patterns for fast matching
LOG_PATTERNS: List[LogPattern] = [
    # NCCL patterns (high confidence - these are explicit NCCL errors)
    LogPattern(
        re.compile(r"NCCL.*timeout|ncclSystemError|ncclInternalError", re.IGNORECASE),
        FailureCategory.NCCL_TIMEOUT,
        RootCause.NCCL_TIMEOUT,
        0.95,
        "NCCL timeout or system error"
    ),
    LogPattern(
        re.compile(r"NCCL.*watchdog|Watchdog caught collective operation timeout", re.IGNORECASE),
        FailureCategory.NCCL_TIMEOUT,
        RootCause.NCCL_TIMEOUT,
        0.95,
        "NCCL watchdog timeout"
    ),
    LogPattern(
        re.compile(r"NCCL.*connection.*reset|NCCL.*peer.*closed", re.IGNORECASE),
        FailureCategory.NETWORK,
        RootCause.NETWORK,
        0.85,
        "NCCL connection reset"
    ),

    # GPU errors
    LogPattern(
        re.compile(r"CUDA.*out of memory|OOM|OutOfMemoryError.*GPU", re.IGNORECASE),
        FailureCategory.GPU_ERROR,
        RootCause.GPU_ERROR,
        0.95,
        "GPU out of memory"
    ),
    LogPattern(
        re.compile(r"CUDA.*error|cudaError|cuda.*illegal", re.IGNORECASE),
        FailureCategory.GPU_ERROR,
        RootCause.GPU_ERROR,
        0.9,
        "CUDA error"
    ),
    LogPattern(
        re.compile(r"GPU.*fallen off bus|GPU has fallen off the bus", re.IGNORECASE),
        FailureCategory.GPU_ERROR,
        RootCause.GPU_FALLEN_OFF_BUS,
        0.99,
        "GPU fallen off bus (critical hardware failure)"
    ),
    LogPattern(
        re.compile(r"XID.*Error|Xid.*\d+", re.IGNORECASE),
        FailureCategory.GPU_ERROR,
        RootCause.GPU_ERROR,
        0.85,
        "GPU XID error"
    ),

    # CPU/Memory issues
    LogPattern(
        re.compile(r"Cannot allocate memory|MemoryError|std::bad_alloc", re.IGNORECASE),
        FailureCategory.CPU_OOM,
        RootCause.CPU_OOM,
        0.9,
        "CPU memory allocation failure"
    ),
    LogPattern(
        re.compile(r"Killed.*out of memory|OOM killer|oom-killer", re.IGNORECASE),
        FailureCategory.CPU_OOM,
        RootCause.CPU_OOM,
        0.95,
        "Process killed by OOM killer"
    ),
    LogPattern(
        re.compile(r"CPU.*overload|load average.*high|system.*overloaded", re.IGNORECASE),
        FailureCategory.CPU_OVERLOAD,
        RootCause.CPU_OVERLOAD,
        0.7,
        "CPU overload detected"
    ),

    # Disk issues
    LogPattern(
        re.compile(r"No space left on device|disk.*full|ENOSPC", re.IGNORECASE),
        FailureCategory.INSUFFICIENT_DISK,
        RootCause.INSUFFICIENT_DISK,
        0.95,
        "Disk space exhausted"
    ),
    LogPattern(
        re.compile(r"disk.*error|I/O error|EIO|read-only file system", re.IGNORECASE),
        FailureCategory.INSUFFICIENT_DISK,
        RootCause.INSUFFICIENT_DISK,
        0.8,
        "Disk I/O error"
    ),

    # Network/InfiniBand
    LogPattern(
        re.compile(r"InfiniBand.*error|ibv_.*error|ib_.*failed|mlx.*error", re.IGNORECASE),
        FailureCategory.INFINIBAND_ERROR,
        RootCause.INFINIBAND_ERROR,
        0.9,
        "InfiniBand error"
    ),
    LogPattern(
        re.compile(r"Connection.*refused|Connection.*reset|ECONNREFUSED|ECONNRESET", re.IGNORECASE),
        FailureCategory.NETWORK,
        RootCause.NETWORK,
        0.7,
        "Network connection error"
    ),
    LogPattern(
        re.compile(r"socket.*timeout|read.*timed out|connection.*timed out", re.IGNORECASE),
        FailureCategory.NETWORK,
        RootCause.NETWORK,
        0.65,
        "Network timeout"
    ),

    # Filesystem
    LogPattern(
        re.compile(r"mount.*failed|cannot mount|ESTALE|Stale.*file handle", re.IGNORECASE),
        FailureCategory.FILESYSTEM_MOUNT,
        RootCause.FILESYSTEM_MOUNT,
        0.85,
        "Filesystem mount failure"
    ),
    LogPattern(
        re.compile(r"HDFS.*error|hadoop.*exception|namenode.*unavailable", re.IGNORECASE),
        FailureCategory.HDFS_ERROR,
        RootCause.HDFS_ERROR,
        0.8,
        "HDFS error"
    ),

    # Container/OS
    LogPattern(
        re.compile(r"container.*killed|docker.*error|OCI.*error", re.IGNORECASE),
        FailureCategory.CONTAINER_ERROR,
        RootCause.CONTAINER_ERROR,
        0.8,
        "Container error"
    ),
    LogPattern(
        re.compile(r"kernel.*panic|Oops|BUG:|segfault|SIGSEGV", re.IGNORECASE),
        FailureCategory.OS_KERNEL_PANIC,
        RootCause.OS_KERNEL_PANIC,
        0.95,
        "Kernel panic or segfault"
    ),

    # Training issues
    LogPattern(
        re.compile(r"NaN.*loss|loss.*nan|nan.*detected|gradient.*nan", re.IGNORECASE),
        FailureCategory.NAN_VALUE,
        RootCause.NAN_VALUE,
        0.9,
        "NaN detected in training"
    ),
    LogPattern(
        re.compile(r"gradient.*overflow|gradient.*explosion|gradient.*inf", re.IGNORECASE),
        FailureCategory.NAN_VALUE,
        RootCause.GRADIENT_EXPLOSION,
        0.85,
        "Gradient explosion detected"
    ),
]


class LogCaptureHandler(logging.Handler):
    """Custom logging handler that captures logs into a ring buffer."""

    def __init__(self, buffer: deque, max_size: int = 10000):
        super().__init__()
        self.buffer = buffer
        self.max_size = max_size
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord):
        try:
            import time
            entry = LogEntry(
                timestamp=time.time(),
                level=record.levelno,
                message=record.getMessage(),
                logger_name=record.name,
                source_file=record.pathname,
                line_number=record.lineno,
            )
            with self._lock:
                if len(self.buffer) >= self.max_size:
                    self.buffer.popleft()
                self.buffer.append(entry)
        except Exception:
            pass  # Never fail in logging handler


@register_diagnostic
class LogAnalyzer(BaseDiagnostic):
    """Analyzes captured logs to identify failure patterns.

    This is a P0 diagnostic that covers ~60% of failures through
    pattern matching against known error signatures.
    """

    name = "log_analyzer"
    phases = [DiagnosticPhase.ON_FAILURE, DiagnosticPhase.ON_DEMAND]
    priority = 10  # Run early to identify obvious failures

    def __init__(self):
        self._buffer: deque = deque(maxlen=10000)
        self._handler: Optional[LogCaptureHandler] = None
        self._patterns = LOG_PATTERNS
        self._config: Optional[LogAnalyzerConfig] = None

    def initialize(self, config: DiagnosticsConfig) -> None:
        """Set up log capture."""
        self._config = config.log_analyzer
        self._buffer = deque(maxlen=self._config.ring_buffer_size)

        if self._config.capture_python_logging:
            self._handler = LogCaptureHandler(self._buffer, self._config.ring_buffer_size)
            self._handler.setLevel(logging.WARNING)  # Only capture warnings and above
            logging.getLogger().addHandler(self._handler)

    def cleanup(self) -> None:
        """Remove log capture handler."""
        if self._handler is not None:
            logging.getLogger().removeHandler(self._handler)
            self._handler = None
        self._buffer.clear()

    def is_enabled(self, config: DiagnosticsConfig) -> bool:
        return config.log_analyzer.enabled

    def check(self, context: DiagnosticContext) -> DiagnosticResult:
        """Analyze captured logs for failure patterns."""
        if not self._buffer:
            return DiagnosticResult.passed(
                self.name,
                message="No logs captured to analyze"
            )

        # Copy buffer for thread-safe analysis
        logs = list(self._buffer)

        matches = self._find_matches(logs)

        if not matches:
            return DiagnosticResult.passed(
                self.name,
                message=f"Analyzed {len(logs)} log entries, no failure patterns found"
            )

        # Find highest confidence match
        best_match = max(matches, key=lambda m: m[0].confidence)
        pattern, entry = best_match

        # Count how many matches we have for confirmation
        same_category_matches = [m for m in matches if m[0].category == pattern.category]

        return DiagnosticResult(
            status=DiagnosticStatus.FAILED,
            diagnostic_name=self.name,
            root_cause=pattern.root_cause,
            root_cause_details=pattern.description,
            confidence=pattern.confidence,
            confirmation_count=len(same_category_matches),
            message=f"Found {len(matches)} failure patterns. Best match: {pattern.description}",
            metrics={
                "total_logs_analyzed": len(logs),
                "pattern_matches": len(matches),
                "match_confidence": pattern.confidence,
            },
            suspect_details={
                "matching_log": entry.message[:500],
                "log_source": f"{entry.source_file}:{entry.line_number}" if entry.source_file else "unknown",
                "category": pattern.category.value,
            },
        )

    def _find_matches(self, logs: List[LogEntry]) -> List[Tuple[LogPattern, LogEntry]]:
        """Find all pattern matches in the logs."""
        matches = []
        for entry in logs:
            for pattern in self._patterns:
                if pattern.pattern.search(entry.message):
                    matches.append((pattern, entry))
                    break  # One pattern per log entry
        return matches

    def add_log_entry(self, message: str, level: int = logging.ERROR) -> None:
        """Manually add a log entry (useful for testing or external log sources)."""
        import time
        entry = LogEntry(
            timestamp=time.time(),
            level=level,
            message=message,
            logger_name="external",
        )
        self._buffer.append(entry)

    def get_recent_logs(self, count: int = 100, level: int = logging.WARNING) -> List[LogEntry]:
        """Get recent log entries above a certain level."""
        return [
            entry for entry in list(self._buffer)[-count:]
            if entry.level >= level
        ]
