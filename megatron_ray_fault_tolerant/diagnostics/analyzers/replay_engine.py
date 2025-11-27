"""Dual-Phase Replay Engine - P2 diagnostic for SDC isolation.

Uses dimension-aware grouping to isolate faulty machines through
systematic replay of training operations.
"""

import hashlib
import time
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set, Tuple, Any
from enum import Enum

from ..core.base import BaseDiagnostic, DiagnosticPhase
from ..core.context import DiagnosticContext
from ..core.result import DiagnosticResult, DiagnosticStatus
from ..config import DiagnosticsConfig, RootCause, DualPhaseReplayConfig
from ..registry import register_diagnostic

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class OperationChecksum:
    """Checksum record for a single operation."""
    step: int
    op_name: str
    input_hash: str
    output_hash: str
    timestamp: float


@dataclass
class TensorSnapshot:
    """Snapshot of tensor state at a point in time."""
    step: int
    name: str
    path: str  # Path to saved tensor
    shape: Tuple[int, ...]
    dtype: str
    checksum: str


class GroupingStrategy(Enum):
    """Strategy for grouping machines in replay."""
    HORIZONTAL = "horizontal"  # Machines 0-3, 4-7, etc.
    VERTICAL = "vertical"      # Machines 0,4,8,...; 1,5,9,...


@register_diagnostic
class DualPhaseReplayEngine(BaseDiagnostic):
    """Isolates SDC through dimension-aware dual-phase replay.

    Key Insight: Keep TP/PP fixed, vary only DP size to preserve
    communication patterns during replay.

    Phase 1 (Normal Execution - Lightweight Logging):
    - Log operation metadata: step, op_type, input/output checksums
    - Periodic tensor snapshots: Full tensors saved every N steps
    - Ring buffer for checksums, disk/cloud for snapshots

    Phase 2 (Stop-Time Analysis):
    - Load from last checkpoint + nearest snapshot
    - Run dimension-aware grouping to isolate faulty machine(s)
    - Replay operations, compare checksums at each step

    Algorithm:
    1. Horizontal grouping: machines 0-3, 4-7, etc.
    2. Vertical grouping: machines 0,4,8,...; 1,5,9,...
    3. Solve intersection to identify faulty machine(s)
    """

    name = "dual_phase_replay"
    phases = [DiagnosticPhase.ON_DEMAND]  # Only run when explicitly triggered
    priority = 100  # Run last (expensive operation)

    def __init__(self):
        self._config: Optional[DualPhaseReplayConfig] = None
        self._checksum_buffer: deque = deque(maxlen=10000)
        self._snapshots: List[TensorSnapshot] = []
        self._current_step: int = 0

    def initialize(self, config: DiagnosticsConfig) -> None:
        self._config = config.dual_phase_replay
        self._checksum_buffer = deque(maxlen=config.dual_phase_replay.checksum_ring_buffer_size)
        self._snapshots = []

    def cleanup(self) -> None:
        self._checksum_buffer.clear()
        self._snapshots.clear()

    def is_enabled(self, config: DiagnosticsConfig) -> bool:
        return config.dual_phase_replay.enabled

    # ============= Phase 1: Logging during training =============

    def log_operation(
        self,
        step: int,
        op_name: str,
        input_tensors: List[Any],
        output_tensors: List[Any],
    ) -> None:
        """Log an operation checksum during training.

        This is called during normal training to record operation
        metadata for later replay analysis.
        """
        if not self._config.enable_checksum_logging:
            return

        input_hash = self._compute_multi_tensor_hash(input_tensors)
        output_hash = self._compute_multi_tensor_hash(output_tensors)

        checksum = OperationChecksum(
            step=step,
            op_name=op_name,
            input_hash=input_hash,
            output_hash=output_hash,
            timestamp=time.time(),
        )

        self._checksum_buffer.append(checksum)
        self._current_step = step

    def save_snapshot(
        self,
        step: int,
        tensors: Dict[str, Any],
        artifact_dir: str,
    ) -> None:
        """Save tensor snapshot to disk.

        Called periodically (every N steps) to save full tensor state
        for later replay.
        """
        if not self._config.enable_tensor_snapshots:
            return

        # Limit number of snapshots kept
        if len(self._snapshots) >= self._config.max_snapshots_kept:
            # Remove oldest
            oldest = self._snapshots.pop(0)
            if os.path.exists(oldest.path):
                os.remove(oldest.path)

        snapshot_dir = os.path.join(artifact_dir, f"snapshot_step_{step}")
        os.makedirs(snapshot_dir, exist_ok=True)

        for name, tensor in tensors.items():
            if not TORCH_AVAILABLE:
                continue

            path = os.path.join(snapshot_dir, f"{name}.pt")
            torch.save(tensor.detach().cpu(), path)

            snapshot = TensorSnapshot(
                step=step,
                name=name,
                path=path,
                shape=tuple(tensor.shape),
                dtype=str(tensor.dtype),
                checksum=self._compute_tensor_hash(tensor),
            )
            self._snapshots.append(snapshot)

    def should_snapshot(self, step: int) -> bool:
        """Check if a snapshot should be saved at this step."""
        if not self._config.enable_tensor_snapshots:
            return False
        return step > 0 and step % self._config.snapshot_every_n_steps == 0

    # ============= Phase 2: Stop-time analysis =============

    def check(self, context: DiagnosticContext) -> DiagnosticResult:
        """Run dual-phase replay analysis.

        This is typically triggered after SDC is detected to isolate
        the faulty machine(s).
        """
        if not self._checksum_buffer:
            return DiagnosticResult.skipped(
                self.name,
                reason="No operation checksums recorded"
            )

        # In a full implementation, this would:
        # 1. Load checkpoint + nearest snapshot
        # 2. Replay with horizontal grouping
        # 3. Replay with vertical grouping
        # 4. Solve intersection

        # For now, analyze the checksum divergence
        return self._analyze_checksum_divergence(context)

    def locate_faulty_machines(
        self,
        total_machines: int,
        group_size: int,
        context: DiagnosticContext,
    ) -> Set[int]:
        """Locate faulty machines using dual-phase grouping.

        Args:
            total_machines: Total number of machines
            group_size: Size of groups (recommended: k * PP_size)
            context: Diagnostic context

        Returns:
            Set of suspected faulty machine IDs
        """
        n_groups = total_machines // group_size

        # Phase 1: Horizontal grouping (machines 0-3, 4-7, ...)
        faulty_horizontal = self._replay_and_identify(
            GroupingStrategy.HORIZONTAL,
            lambda mid: mid // group_size,
            total_machines,
            n_groups,
            context,
        )

        # Phase 2: Vertical grouping (machines 0,4,8,...; 1,5,9,...)
        faulty_vertical = self._replay_and_identify(
            GroupingStrategy.VERTICAL,
            lambda mid: mid % n_groups,
            total_machines,
            n_groups,
            context,
        )

        # Solve intersection: find machine(s) in both faulty groups
        suspects = self._solve_intersection(
            faulty_horizontal,
            faulty_vertical,
            group_size,
            n_groups,
        )

        return suspects

    def _replay_and_identify(
        self,
        strategy: GroupingStrategy,
        partition_fn,
        total_machines: int,
        n_groups: int,
        context: DiagnosticContext,
    ) -> int:
        """Replay with grouping and identify faulty group.

        Returns the group ID that produced divergent output.
        """
        # This would require actually running replay on grouped machines
        # For now, return a placeholder
        # In production, this would:
        # 1. Partition machines according to strategy
        # 2. Run replay on each group
        # 3. Compare outputs across groups
        # 4. Return ID of divergent group
        return -1

    def _solve_intersection(
        self,
        faulty_horizontal: int,
        faulty_vertical: int,
        group_size: int,
        n_groups: int,
    ) -> Set[int]:
        """Solve for faulty machine(s) given group IDs.

        Given:
        - faulty_horizontal = floor(x/m) = a (horizontal group)
        - faulty_vertical = x mod n = b (vertical group)

        Solve for x where:
        - m = group_size
        - n = n_groups
        """
        if faulty_horizontal < 0 or faulty_vertical < 0:
            return set()

        suspects = set()
        for machine_id in range(group_size * n_groups):
            if (machine_id // group_size == faulty_horizontal and
                machine_id % n_groups == faulty_vertical):
                suspects.add(machine_id)

        return suspects

    def _analyze_checksum_divergence(
        self,
        context: DiagnosticContext,
    ) -> DiagnosticResult:
        """Analyze recorded checksums for divergence patterns."""
        checksums = list(self._checksum_buffer)

        if not checksums:
            return DiagnosticResult.passed(
                self.name,
                message="No checksums to analyze"
            )

        # Group checksums by step
        by_step: Dict[int, List[OperationChecksum]] = {}
        for cs in checksums:
            if cs.step not in by_step:
                by_step[cs.step] = []
            by_step[cs.step].append(cs)

        # Look for steps with divergent output hashes
        divergent_steps = []
        for step, step_checksums in by_step.items():
            output_hashes = set(cs.output_hash for cs in step_checksums)
            if len(output_hashes) > 1:
                divergent_steps.append({
                    "step": step,
                    "operations": [cs.op_name for cs in step_checksums],
                    "hash_count": len(output_hashes),
                })

        if not divergent_steps:
            return DiagnosticResult(
                status=DiagnosticStatus.PASSED,
                diagnostic_name=self.name,
                confidence=1.0,
                message=f"Analyzed {len(checksums)} operations across {len(by_step)} steps",
                metrics={
                    "total_operations": len(checksums),
                    "steps_analyzed": len(by_step),
                    "snapshots_available": len(self._snapshots),
                },
            )

        return DiagnosticResult(
            status=DiagnosticStatus.FAILED,
            diagnostic_name=self.name,
            root_cause=RootCause.SILENT_DATA_CORRUPTION,
            root_cause_details=f"Found {len(divergent_steps)} steps with divergent outputs",
            confidence=0.8,
            message=f"Output divergence detected in {len(divergent_steps)} steps",
            metrics={
                "total_operations": len(checksums),
                "divergent_steps": len(divergent_steps),
            },
            suspect_details={
                "divergent_steps": divergent_steps[:10],  # First 10
            },
            recommended_action="run_full_replay_analysis",
        )

    def _compute_tensor_hash(self, tensor) -> str:
        """Compute MD5 hash of a tensor."""
        if not TORCH_AVAILABLE:
            return "unavailable"
        data = tensor.detach().cpu().numpy().tobytes()
        return hashlib.md5(data).hexdigest()[:16]

    def _compute_multi_tensor_hash(self, tensors: List[Any]) -> str:
        """Compute combined hash of multiple tensors."""
        if not TORCH_AVAILABLE or not tensors:
            return "empty"

        combined = hashlib.md5()
        for t in tensors:
            if hasattr(t, 'detach'):
                data = t.detach().cpu().numpy().tobytes()
                combined.update(data)

        return combined.hexdigest()[:16]

    @property
    def checksum_count(self) -> int:
        """Get number of recorded checksums."""
        return len(self._checksum_buffer)

    @property
    def snapshot_count(self) -> int:
        """Get number of saved snapshots."""
        return len(self._snapshots)

    def get_nearest_snapshot(self, step: int) -> Optional[TensorSnapshot]:
        """Get the snapshot nearest to a given step."""
        if not self._snapshots:
            return None

        return min(self._snapshots, key=lambda s: abs(s.step - step))
