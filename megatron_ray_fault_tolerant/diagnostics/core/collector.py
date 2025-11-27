"""DiagnosticCollector - orchestrates running diagnostics."""

import time
from typing import List, Optional, Dict, Any
from loguru import logger

from .base import BaseDiagnostic, DiagnosticPhase
from .context import DiagnosticContext
from .result import DiagnosticResult, DiagnosticStatus
from ..config import DiagnosticsConfig
from ..registry import DiagnosticRegistry


class DiagnosticCollector:
    """Orchestrates running diagnostic checks.

    The collector manages the lifecycle of diagnostics:
    1. Initializes enabled diagnostics at startup
    2. Runs diagnostics for a given phase
    3. Aggregates and reports results
    4. Cleans up at shutdown

    Example:
        collector = DiagnosticCollector(config)
        collector.initialize()

        # Run diagnostics for a phase
        results = collector.run_phase(DiagnosticPhase.ON_FAILURE, context)

        # Check results
        for result in results:
            if result.status == DiagnosticStatus.FAILED:
                print(f"Failure detected: {result.root_cause}")
    """

    def __init__(self, config: DiagnosticsConfig):
        self.config = config
        self._instances: Dict[str, BaseDiagnostic] = {}
        self._initialized = False

    def initialize(self) -> None:
        """Initialize all enabled diagnostics."""
        if self._initialized:
            return

        if not self.config.enabled:
            logger.info("Diagnostics disabled in config")
            return

        for name, diagnostic_class in DiagnosticRegistry.list_registered().items():
            instance = diagnostic_class()
            if instance.is_enabled(self.config):
                try:
                    instance.initialize(self.config)
                    self._instances[name] = instance
                    logger.debug(f"Initialized diagnostic: {name}")
                except Exception as e:
                    logger.error(f"Failed to initialize diagnostic {name}: {e}")

        self._initialized = True
        logger.info(f"Initialized {len(self._instances)} diagnostics: {list(self._instances.keys())}")

    def cleanup(self) -> None:
        """Cleanup all diagnostics."""
        for name, instance in self._instances.items():
            try:
                instance.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up diagnostic {name}: {e}")

        self._instances.clear()
        self._initialized = False

    def run_phase(
        self,
        phase: DiagnosticPhase,
        context: DiagnosticContext,
        stop_on_failure: bool = False,
    ) -> List[DiagnosticResult]:
        """Run all diagnostics for a given phase.

        Args:
            phase: The diagnostic phase to run
            context: Runtime context
            stop_on_failure: If True, stop after first failure

        Returns:
            List of diagnostic results
        """
        if not self.config.enabled:
            return []

        if not self._initialized:
            self.initialize()

        results = []

        # Get diagnostics for this phase, sorted by priority
        phase_diagnostics = [
            (name, instance)
            for name, instance in self._instances.items()
            if instance.should_run_in_phase(phase)
        ]
        phase_diagnostics.sort(key=lambda x: x[1].priority)

        logger.info(f"Running {len(phase_diagnostics)} diagnostics for phase {phase.value}")

        for name, instance in phase_diagnostics:
            try:
                start_time = time.time()
                result = instance.check(context)
                result.diagnostic_name = name
                result.timestamp = time.time()
                elapsed = time.time() - start_time

                logger.debug(
                    f"Diagnostic {name}: {result.status.value} "
                    f"(confidence={result.confidence:.2f}, elapsed={elapsed:.3f}s)"
                )

                results.append(result)

                if stop_on_failure and result.status == DiagnosticStatus.FAILED:
                    logger.info(f"Stopping diagnostics early due to failure in {name}")
                    break

            except Exception as e:
                logger.error(f"Exception in diagnostic {name}: {e}", exc_info=True)
                results.append(DiagnosticResult.error(name, str(e)))

        return results

    def run_specific(
        self,
        diagnostic_name: str,
        context: DiagnosticContext,
    ) -> DiagnosticResult:
        """Run a specific diagnostic by name.

        Args:
            diagnostic_name: Name of the diagnostic to run
            context: Runtime context

        Returns:
            The diagnostic result
        """
        if diagnostic_name not in self._instances:
            return DiagnosticResult.error(
                diagnostic_name,
                f"Diagnostic '{diagnostic_name}' not found or not enabled"
            )

        instance = self._instances[diagnostic_name]
        try:
            result = instance.check(context)
            result.diagnostic_name = diagnostic_name
            result.timestamp = time.time()
            return result
        except Exception as e:
            logger.error(f"Exception in diagnostic {diagnostic_name}: {e}", exc_info=True)
            return DiagnosticResult.error(diagnostic_name, str(e))

    def get_summary(self, results: List[DiagnosticResult]) -> Dict[str, Any]:
        """Get a summary of diagnostic results.

        Args:
            results: List of diagnostic results

        Returns:
            Summary dictionary with counts and key findings
        """
        summary = {
            "total": len(results),
            "passed": 0,
            "failed": 0,
            "warnings": 0,
            "skipped": 0,
            "errors": 0,
            "critical_failures": [],
            "suspects": set(),
            "root_causes": [],
            "needs_escalation": [],
        }

        for result in results:
            if result.status == DiagnosticStatus.PASSED:
                summary["passed"] += 1
            elif result.status == DiagnosticStatus.FAILED:
                summary["failed"] += 1
                if result.is_critical:
                    summary["critical_failures"].append(result.diagnostic_name)
                if result.root_cause:
                    summary["root_causes"].append(result.root_cause.value)
                summary["suspects"].update(result.suspects)
                if result.needs_escalation:
                    summary["needs_escalation"].append(result.diagnostic_name)
            elif result.status == DiagnosticStatus.WARNING:
                summary["warnings"] += 1
            elif result.status == DiagnosticStatus.SKIPPED:
                summary["skipped"] += 1
            elif result.status == DiagnosticStatus.ERROR:
                summary["errors"] += 1

        summary["suspects"] = list(summary["suspects"])
        return summary

    @property
    def enabled_diagnostics(self) -> List[str]:
        """Get list of enabled diagnostic names."""
        return list(self._instances.keys())
