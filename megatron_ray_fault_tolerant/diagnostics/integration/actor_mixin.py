"""DiagnosticsMixin - adds diagnostic capabilities to MegatronActor.

This mixin provides a non-intrusive way to add diagnostics to the
existing MegatronActor class without modifying its core logic.
"""

import time
from typing import Optional, List, Dict, Any, TYPE_CHECKING

from ..core.base import DiagnosticPhase
from ..core.context import DiagnosticContext
from ..core.result import DiagnosticResult
from ..core.collector import DiagnosticCollector
from ..config import DiagnosticsConfig
from ..storage.artifact_store import ArtifactStore, DiagnosticReport

if TYPE_CHECKING:
    from dispatch import MeshRank


class DiagnosticsMixin:
    """Mixin that adds diagnostic capabilities to MegatronActor.

    This mixin provides methods for:
    - Running diagnostics at various phases
    - Recording training metrics for implicit failure detection
    - Triggering on-demand diagnostics
    - Collecting and storing diagnostic reports

    Usage:
        class MegatronActor(DiagnosticsMixin):
            def init_worker_process_group(self):
                # ... existing code ...
                self.run_diagnostics(DiagnosticPhase.PRE_TRAINING)

            def ppo_train(self, train_data):
                # ... existing training ...
                self.record_training_metrics(step, loss, grad_norm)
                self.run_diagnostics(DiagnosticPhase.POST_STEP)
    """

    # These attributes should be set by the host class
    _rank: int
    _world_size: int
    mesh_rank: Optional["MeshRank"]
    cfg: Any

    # Diagnostic components (initialized by init_diagnostics)
    _diag_config: Optional[DiagnosticsConfig] = None
    _diag_collector: Optional[DiagnosticCollector] = None
    _diag_store: Optional[ArtifactStore] = None
    _diag_initialized: bool = False
    _current_step: int = 0

    def init_diagnostics(
        self,
        config: Optional[DiagnosticsConfig] = None,
    ) -> None:
        """Initialize the diagnostics system.

        Args:
            config: Diagnostics configuration. If None, uses defaults.
        """
        if self._diag_initialized:
            return

        self._diag_config = config or DiagnosticsConfig()

        if not self._diag_config.enabled:
            self._diag_initialized = True
            return

        # Initialize collector and store
        self._diag_collector = DiagnosticCollector(self._diag_config)
        self._diag_collector.initialize()

        self._diag_store = ArtifactStore(
            local_dir=self._diag_config.artifact_dir,
            cloud_dir=self._diag_config.cloud_artifact_dir,
        )

        self._diag_initialized = True

    def cleanup_diagnostics(self) -> None:
        """Cleanup diagnostics resources."""
        if self._diag_collector:
            self._diag_collector.cleanup()
            self._diag_collector = None

        self._diag_initialized = False

    def run_diagnostics(
        self,
        phase: DiagnosticPhase,
        extra_context: Optional[Dict[str, Any]] = None,
        stop_on_failure: bool = False,
    ) -> List[DiagnosticResult]:
        """Run diagnostics for a given phase.

        Args:
            phase: The diagnostic phase to run
            extra_context: Additional context to pass to diagnostics
            stop_on_failure: If True, stop after first failure

        Returns:
            List of diagnostic results
        """
        if not self._diag_initialized or not self._diag_config.enabled:
            return []

        context = self._create_context(extra_context)

        results = self._diag_collector.run_phase(
            phase, context, stop_on_failure
        )

        # Save report if we have failures
        if any(r.status.value == "failed" for r in results):
            self._save_diagnostic_report(phase, context, results)

        return results

    def run_diagnostic_by_name(
        self,
        name: str,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> DiagnosticResult:
        """Run a specific diagnostic by name.

        Args:
            name: Name of the diagnostic to run
            extra_context: Additional context

        Returns:
            The diagnostic result
        """
        if not self._diag_initialized or not self._diag_config.enabled:
            return DiagnosticResult.skipped(name, "Diagnostics not enabled")

        context = self._create_context(extra_context)
        return self._diag_collector.run_specific(name, context)

    def record_training_metrics(
        self,
        step: int,
        loss: Optional[float] = None,
        gradient_norm: Optional[float] = None,
        throughput: Optional[float] = None,
        mfu: Optional[float] = None,
    ) -> None:
        """Record training metrics for implicit failure detection.

        This should be called after each training step to track
        progress and enable anomaly detection.

        Args:
            step: Current training step
            loss: Training loss
            gradient_norm: Gradient norm
            throughput: Training throughput (samples/sec or tokens/sec)
            mfu: Model FLOPS Utilization
        """
        if not self._diag_initialized or not self._diag_config.enabled:
            return

        self._current_step = step

        # Find the implicit detector and record metrics
        from ..analyzers.implicit_detector import ImplicitFailureDetector

        for name, instance in self._diag_collector._instances.items():
            if isinstance(instance, ImplicitFailureDetector):
                instance.record_step_metrics(
                    step=step,
                    loss=loss,
                    gradient_norm=gradient_norm,
                    throughput=throughput,
                    mfu=mfu,
                )
                break

    def check_for_nan(self, model) -> Optional[DiagnosticResult]:
        """Check model for NaN values.

        Args:
            model: PyTorch model to check

        Returns:
            DiagnosticResult if NaN found, None otherwise
        """
        if not self._diag_initialized or not self._diag_config.enabled:
            return None

        context = self._create_context({"model": model})
        result = self._diag_collector.run_specific("nan_monitor", context)

        if result.status.value == "failed":
            # NaN detected - optionally trigger SDC test
            if self._diag_config.nan_monitor.trigger_sdc_test_on_nan:
                sdc_result = self._diag_collector.run_specific("sdc_detector", context)
                if sdc_result.status.value == "failed":
                    # SDC confirmed
                    result = result.merge_with(sdc_result)

        return result if result.status.value == "failed" else None

    def get_diagnostic_summary(self) -> Dict[str, Any]:
        """Get a summary of enabled diagnostics and their status.

        Returns:
            Dictionary with diagnostic status information
        """
        if not self._diag_initialized:
            return {"enabled": False}

        return {
            "enabled": self._diag_config.enabled,
            "enabled_diagnostics": self._diag_collector.enabled_diagnostics,
            "current_step": self._current_step,
            "config": {
                "artifact_dir": self._diag_config.artifact_dir,
                "cloud_artifact_dir": self._diag_config.cloud_artifact_dir,
            },
        }

    def get_stack_trace(self) -> str:
        """Get current stack trace from this process.

        Used by stack aggregation analyzer to collect distributed stacks.
        """
        import traceback
        return traceback.format_stack()

    def _create_context(
        self,
        extra: Optional[Dict[str, Any]] = None,
    ) -> DiagnosticContext:
        """Create a DiagnosticContext for running diagnostics."""
        return DiagnosticContext(
            rank=getattr(self, "_rank", 0),
            world_size=getattr(self, "_world_size", 1),
            mesh_rank=getattr(self, "mesh_rank", None),
            step=self._current_step,
            artifact_dir=self._diag_config.artifact_dir if self._diag_config else "/tmp/diagnostics",
            cloud_artifact_dir=self._diag_config.cloud_artifact_dir if self._diag_config else None,
            extra=extra or {},
            actor_ref=self,
        )

    def _save_diagnostic_report(
        self,
        phase: DiagnosticPhase,
        context: DiagnosticContext,
        results: List[DiagnosticResult],
    ) -> None:
        """Save a diagnostic report with results."""
        if not self._diag_store:
            return

        summary = self._diag_collector.get_summary(results)

        report = DiagnosticReport(
            trace_id=context.trace_id,
            timestamp=time.time(),
            rank=context.rank,
            world_size=context.world_size,
            step=context.step,
            phase=phase.value,
            results=[r.to_dict() for r in results],
            summary=summary,
        )

        self._diag_store.save_report(report)
