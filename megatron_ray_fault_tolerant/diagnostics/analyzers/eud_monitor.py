"""EUD Monitor - P0 diagnostic for NVIDIA GPU health monitoring.

Uses NVML to check GPU health metrics and detect hardware issues.
Covers ~15% of failures related to GPU hardware problems.
"""

import subprocess
from dataclasses import dataclass
from typing import List, Dict, Optional, Any

from ..core.base import BaseDiagnostic, DiagnosticPhase
from ..core.context import DiagnosticContext
from ..core.result import DiagnosticResult, DiagnosticStatus
from ..config import DiagnosticsConfig, RootCause, EUDConfig
from ..registry import register_diagnostic

# Try to import pynvml, but don't fail if not available
try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False


@dataclass
class GPUHealthSnapshot:
    """Snapshot of GPU health metrics."""
    device_id: int
    name: str
    temperature_celsius: int
    power_usage_watts: float
    memory_used_mb: int
    memory_total_mb: int
    ecc_errors_corrected: int
    ecc_errors_uncorrected: int
    retired_pages_single_bit: int
    retired_pages_double_bit: int
    retired_pages_pending: int
    pcie_replay_count: int
    compute_mode: str
    persistence_mode: bool


@dataclass
class XIDError:
    """Represents an XID error from GPU."""
    timestamp: str
    device_id: int
    xid_code: int
    message: str


@register_diagnostic
class EUDMonitor(BaseDiagnostic):
    """Monitors GPU health using NVML (NVIDIA Management Library).

    This is a P0 diagnostic that detects GPU hardware issues including:
    - Temperature and thermal throttling
    - ECC errors (corrected and uncorrected)
    - Retired pages
    - PCIe replay errors
    - XID errors from system logs

    Critical XID codes that indicate hardware failure:
    - 31: GPU memory page fault
    - 48: Double Bit ECC Error (CRITICAL)
    - 63: ECC page retirement failure
    - 64, 79: GPU fallen off bus (CRITICAL)
    - 119: GSP RPC timeout
    """

    name = "eud_monitor"
    phases = [DiagnosticPhase.ON_FAILURE, DiagnosticPhase.PRE_TRAINING, DiagnosticPhase.ON_DEMAND]
    priority = 5  # Run very early to detect hardware issues

    def __init__(self):
        self._config: Optional[EUDConfig] = None
        self._nvml_initialized = False

    def initialize(self, config: DiagnosticsConfig) -> None:
        self._config = config.eud
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self._nvml_initialized = True
            except Exception:
                self._nvml_initialized = False

    def cleanup(self) -> None:
        if self._nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml_initialized = False

    def is_enabled(self, config: DiagnosticsConfig) -> bool:
        return config.eud.enabled

    def check(self, context: DiagnosticContext) -> DiagnosticResult:
        """Check GPU health metrics."""
        if not NVML_AVAILABLE:
            return DiagnosticResult.skipped(
                self.name,
                reason="pynvml not available. Install with: pip install nvidia-ml-py"
            )

        if not self._nvml_initialized:
            try:
                pynvml.nvmlInit()
                self._nvml_initialized = True
            except Exception as e:
                return DiagnosticResult.error(self.name, f"Failed to initialize NVML: {e}")

        try:
            # Get GPU health snapshots
            snapshots = self._get_gpu_snapshots()

            # Check for XID errors
            xid_errors = self._get_xid_errors()

            # Analyze results
            return self._analyze_health(snapshots, xid_errors, context)

        except Exception as e:
            return DiagnosticResult.error(self.name, f"Error checking GPU health: {e}")

    def _get_gpu_snapshots(self) -> List[GPUHealthSnapshot]:
        """Get health snapshots for all GPUs."""
        snapshots = []
        device_count = pynvml.nvmlDeviceGetCount()

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            snapshot = self._get_device_snapshot(i, handle)
            snapshots.append(snapshot)

        return snapshots

    def _get_device_snapshot(self, device_id: int, handle) -> GPUHealthSnapshot:
        """Get health snapshot for a single GPU."""
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode('utf-8')

        # Temperature
        try:
            temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        except Exception:
            temperature = -1

        # Power
        try:
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
        except Exception:
            power = -1.0

        # Memory
        try:
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_used = mem_info.used // (1024 * 1024)
            memory_total = mem_info.total // (1024 * 1024)
        except Exception:
            memory_used = -1
            memory_total = -1

        # ECC errors
        ecc_corrected = 0
        ecc_uncorrected = 0
        try:
            ecc_corrected = pynvml.nvmlDeviceGetTotalEccErrors(
                handle, pynvml.NVML_MEMORY_ERROR_TYPE_CORRECTED,
                pynvml.NVML_VOLATILE_ECC
            )
        except Exception:
            pass
        try:
            ecc_uncorrected = pynvml.nvmlDeviceGetTotalEccErrors(
                handle, pynvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                pynvml.NVML_VOLATILE_ECC
            )
        except Exception:
            pass

        # Retired pages
        retired_single = 0
        retired_double = 0
        retired_pending = 0
        try:
            retired_single = pynvml.nvmlDeviceGetRetiredPages(
                handle, pynvml.NVML_PAGE_RETIREMENT_CAUSE_MULTIPLE_SINGLE_BIT_ECC_ERRORS
            )
            if isinstance(retired_single, (list, tuple)):
                retired_single = len(retired_single)
        except Exception:
            pass
        try:
            retired_double = pynvml.nvmlDeviceGetRetiredPages(
                handle, pynvml.NVML_PAGE_RETIREMENT_CAUSE_DOUBLE_BIT_ECC_ERROR
            )
            if isinstance(retired_double, (list, tuple)):
                retired_double = len(retired_double)
        except Exception:
            pass
        try:
            pending_info = pynvml.nvmlDeviceGetRetiredPagesPendingStatus(handle)
            retired_pending = 1 if pending_info == pynvml.NVML_FEATURE_ENABLED else 0
        except Exception:
            pass

        # PCIe replay
        pcie_replay = 0
        try:
            pcie_replay = pynvml.nvmlDeviceGetPcieReplayCounter(handle)
        except Exception:
            pass

        # Compute mode
        try:
            compute_mode = pynvml.nvmlDeviceGetComputeMode(handle)
            compute_mode_str = {
                0: "Default",
                1: "Exclusive Thread",
                2: "Prohibited",
                3: "Exclusive Process",
            }.get(compute_mode, "Unknown")
        except Exception:
            compute_mode_str = "Unknown"

        # Persistence mode
        try:
            persistence = pynvml.nvmlDeviceGetPersistenceMode(handle)
            persistence_enabled = persistence == pynvml.NVML_FEATURE_ENABLED
        except Exception:
            persistence_enabled = False

        return GPUHealthSnapshot(
            device_id=device_id,
            name=name,
            temperature_celsius=temperature,
            power_usage_watts=power,
            memory_used_mb=memory_used,
            memory_total_mb=memory_total,
            ecc_errors_corrected=ecc_corrected,
            ecc_errors_uncorrected=ecc_uncorrected,
            retired_pages_single_bit=retired_single,
            retired_pages_double_bit=retired_double,
            retired_pages_pending=retired_pending,
            pcie_replay_count=pcie_replay,
            compute_mode=compute_mode_str,
            persistence_mode=persistence_enabled,
        )

    def _get_xid_errors(self) -> List[XIDError]:
        """Get XID errors from system logs."""
        xid_errors = []

        # Try journalctl first (systemd)
        try:
            result = subprocess.run(
                ["journalctl", "-k", "--since", "1 hour ago", "--no-pager"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            xid_errors.extend(self._parse_xid_from_logs(result.stdout))
        except Exception:
            pass

        # Try dmesg as fallback
        if not xid_errors:
            try:
                result = subprocess.run(
                    ["dmesg", "--time-format=iso"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                xid_errors.extend(self._parse_xid_from_logs(result.stdout))
            except Exception:
                pass

        return xid_errors

    def _parse_xid_from_logs(self, log_output: str) -> List[XIDError]:
        """Parse XID errors from log output."""
        import re
        xid_errors = []

        # Pattern: NVRM: Xid (PCI:0000:XX:XX.X): YY, <message>
        pattern = re.compile(
            r"(\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2})?.*"
            r"NVRM:?\s*Xid\s*\(PCI:[^)]+\):\s*(\d+),?\s*(.*)",
            re.IGNORECASE
        )

        for line in log_output.split("\n"):
            match = pattern.search(line)
            if match:
                timestamp = match.group(1) or "unknown"
                xid_code = int(match.group(2))
                message = match.group(3).strip()

                # Try to extract device ID from PCI address
                pci_match = re.search(r"PCI:([0-9a-fA-F:\.]+)", line)
                device_id = 0  # Default, could map PCI to device

                xid_errors.append(XIDError(
                    timestamp=timestamp,
                    device_id=device_id,
                    xid_code=xid_code,
                    message=message,
                ))

        return xid_errors

    def _analyze_health(
        self,
        snapshots: List[GPUHealthSnapshot],
        xid_errors: List[XIDError],
        context: DiagnosticContext,
    ) -> DiagnosticResult:
        """Analyze GPU health data and return diagnostic result."""
        config = self._config
        issues = []
        suspects = []
        root_cause = None
        confidence = 0.0
        is_critical = False

        # Check XID errors first (most critical)
        critical_xids = [x for x in xid_errors if x.xid_code in config.critical_xid_codes]
        if critical_xids:
            is_critical = True
            root_cause = RootCause.GPU_FALLEN_OFF_BUS if any(
                x.xid_code in [64, 79] for x in critical_xids
            ) else RootCause.GPU_ECC_ERROR

            confidence = 0.95
            for xid in critical_xids:
                issues.append(f"Critical XID {xid.xid_code} on GPU {xid.device_id}: {xid.message}")
                if xid.device_id not in suspects:
                    suspects.append(xid.device_id)

        # Check each GPU snapshot
        for snapshot in snapshots:
            # Temperature check
            if snapshot.temperature_celsius > config.temperature_critical_celsius:
                issues.append(
                    f"GPU {snapshot.device_id}: Critical temperature {snapshot.temperature_celsius}C"
                )
                if snapshot.device_id not in suspects:
                    suspects.append(snapshot.device_id)
                if not root_cause:
                    root_cause = RootCause.GPU_ERROR
                    confidence = max(confidence, 0.8)

            # Uncorrected ECC check
            if snapshot.ecc_errors_uncorrected >= config.ecc_uncorrected_threshold:
                is_critical = True
                issues.append(
                    f"GPU {snapshot.device_id}: {snapshot.ecc_errors_uncorrected} uncorrected ECC errors"
                )
                if snapshot.device_id not in suspects:
                    suspects.append(snapshot.device_id)
                root_cause = RootCause.GPU_ECC_ERROR
                confidence = max(confidence, 0.95)

            # Retired pages check
            total_retired = (
                snapshot.retired_pages_single_bit +
                snapshot.retired_pages_double_bit
            )
            if total_retired > config.retired_pages_threshold:
                issues.append(
                    f"GPU {snapshot.device_id}: {total_retired} retired pages"
                )
                if snapshot.device_id not in suspects:
                    suspects.append(snapshot.device_id)
                if not root_cause:
                    root_cause = RootCause.GPU_ERROR
                    confidence = max(confidence, 0.7)

            # Double-bit retired pages are more serious
            if snapshot.retired_pages_double_bit > 0:
                is_critical = True
                issues.append(
                    f"GPU {snapshot.device_id}: {snapshot.retired_pages_double_bit} double-bit retired pages"
                )
                root_cause = RootCause.GPU_ECC_ERROR
                confidence = max(confidence, 0.9)

            # PCIe replay check
            if snapshot.pcie_replay_count > config.pcie_replay_threshold:
                issues.append(
                    f"GPU {snapshot.device_id}: High PCIe replay count ({snapshot.pcie_replay_count})"
                )
                if snapshot.device_id not in suspects:
                    suspects.append(snapshot.device_id)
                if not root_cause:
                    root_cause = RootCause.GPU_ERROR
                    confidence = max(confidence, 0.6)

        # Build result
        if not issues:
            return DiagnosticResult(
                status=DiagnosticStatus.PASSED,
                diagnostic_name=self.name,
                confidence=1.0,
                message=f"All {len(snapshots)} GPUs healthy",
                metrics={
                    "gpu_count": len(snapshots),
                    "xid_errors_found": len(xid_errors),
                },
            )

        return DiagnosticResult(
            status=DiagnosticStatus.FAILED,
            diagnostic_name=self.name,
            root_cause=root_cause,
            root_cause_details="; ".join(issues[:5]),  # First 5 issues
            suspects=suspects,
            confidence=confidence,
            message=f"Found {len(issues)} GPU health issues",
            metrics={
                "gpu_count": len(snapshots),
                "issues_found": len(issues),
                "xid_errors_found": len(xid_errors),
                "critical_xid_errors": len(critical_xids) if 'critical_xids' in dir() else 0,
            },
            suspect_details={
                "all_issues": issues,
                "snapshots": [
                    {
                        "device_id": s.device_id,
                        "name": s.name,
                        "temperature": s.temperature_celsius,
                        "ecc_uncorrected": s.ecc_errors_uncorrected,
                        "retired_pages": s.retired_pages_single_bit + s.retired_pages_double_bit,
                    }
                    for s in snapshots if s.device_id in suspects
                ],
            },
            recommended_action="evict_faulty_gpu" if is_critical else "monitor",
        )

    def get_gpu_snapshots(self) -> List[GPUHealthSnapshot]:
        """Public method to get current GPU snapshots."""
        if not self._nvml_initialized:
            return []
        try:
            return self._get_gpu_snapshots()
        except Exception:
            return []
