"""Implicit Failure Detector - P0 diagnostic for silent failures.

Detects failures that don't produce explicit error messages:
- MFU (Model FLOPS Utilization) decline
- Job hang (no progress)
- Loss/gradient spikes

These are the hardest to diagnose (~10% of failures).
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Deque
from statistics import mean, stdev

from ..core.base import BaseDiagnostic, DiagnosticPhase
from ..core.context import DiagnosticContext
from ..core.result import DiagnosticResult, DiagnosticStatus
from ..config import DiagnosticsConfig, RootCause, ImplicitFailureConfig
from ..registry import register_diagnostic


@dataclass
class TrainingMetrics:
    """Metrics from a single training step."""
    step: int
    timestamp: float
    loss: Optional[float] = None
    gradient_norm: Optional[float] = None
    throughput: Optional[float] = None  # samples/sec or tokens/sec
    mfu: Optional[float] = None


@dataclass
class MetricsHistory:
    """Rolling history of training metrics for anomaly detection."""
    max_size: int = 100
    losses: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    gradient_norms: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    throughputs: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    mfus: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    timestamps: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    last_step: int = 0
    last_update: float = 0.0

    def add(self, metrics: TrainingMetrics) -> None:
        """Add metrics from a training step."""
        if metrics.loss is not None:
            self.losses.append(metrics.loss)
        if metrics.gradient_norm is not None:
            self.gradient_norms.append(metrics.gradient_norm)
        if metrics.throughput is not None:
            self.throughputs.append(metrics.throughput)
        if metrics.mfu is not None:
            self.mfus.append(metrics.mfu)
        self.timestamps.append(metrics.timestamp)
        self.last_step = metrics.step
        self.last_update = metrics.timestamp


@register_diagnostic
class ImplicitFailureDetector(BaseDiagnostic):
    """Detects implicit/silent failures in training.

    This is a P0 diagnostic that runs lightweight checks after each step
    to detect failures that don't produce explicit errors:

    1. **Job Hang**: No progress for extended period
       - RDMA traffic stalled
       - No step progress

    2. **MFU Decline**: Performance degradation
       - Gradual decline in throughput
       - Often indicates hardware issue developing

    3. **Training Anomalies**:
       - Loss spikes (5x normal)
       - Gradient norm spikes (5x normal)

    When an anomaly is detected, triggers Stack Aggregation for root cause.
    """

    name = "implicit_detector"
    phases = [DiagnosticPhase.POST_STEP, DiagnosticPhase.ON_DEMAND]
    priority = 20  # Run after basic checks

    def __init__(self):
        self._config: Optional[ImplicitFailureConfig] = None
        self._history = MetricsHistory()
        self._baseline_mfu: Optional[float] = None
        self._baseline_throughput: Optional[float] = None

    def initialize(self, config: DiagnosticsConfig) -> None:
        self._config = config.implicit_failure
        window_size = self._config.history_window_size
        self._history = MetricsHistory(max_size=window_size)
        self._history.losses = deque(maxlen=window_size)
        self._history.gradient_norms = deque(maxlen=window_size)
        self._history.throughputs = deque(maxlen=window_size)
        self._history.mfus = deque(maxlen=window_size)
        self._history.timestamps = deque(maxlen=window_size)

    def cleanup(self) -> None:
        self._history = MetricsHistory()
        self._baseline_mfu = None
        self._baseline_throughput = None

    def is_enabled(self, config: DiagnosticsConfig) -> bool:
        return config.implicit_failure.enabled

    def record_step_metrics(
        self,
        step: int,
        loss: Optional[float] = None,
        gradient_norm: Optional[float] = None,
        throughput: Optional[float] = None,
        mfu: Optional[float] = None,
    ) -> None:
        """Record metrics from a training step.

        This should be called after each training step to track progress
        and detect anomalies.
        """
        metrics = TrainingMetrics(
            step=step,
            timestamp=time.time(),
            loss=loss,
            gradient_norm=gradient_norm,
            throughput=throughput,
            mfu=mfu,
        )
        self._history.add(metrics)

        # Establish baseline from first N steps
        if len(self._history.mfus) == self._config.mfu_window_steps:
            if self._baseline_mfu is None and self._history.mfus:
                self._baseline_mfu = mean(self._history.mfus)
            if self._baseline_throughput is None and self._history.throughputs:
                self._baseline_throughput = mean(self._history.throughputs)

    def check(self, context: DiagnosticContext) -> DiagnosticResult:
        """Check for implicit failures."""
        config = self._config

        issues = []
        root_cause = None
        confidence = 0.0
        recommended_action = None

        # Check 1: Job Hang - no progress
        hang_result = self._check_job_hang(config)
        if hang_result:
            issues.append(hang_result["message"])
            root_cause = RootCause.HANG
            confidence = max(confidence, hang_result["confidence"])
            recommended_action = "trigger_stack_aggregation"

        # Check 2: MFU Decline
        mfu_result = self._check_mfu_decline(config)
        if mfu_result:
            issues.append(mfu_result["message"])
            if not root_cause:
                root_cause = RootCause.MFU_DECLINE
            confidence = max(confidence, mfu_result["confidence"])
            recommended_action = recommended_action or "investigate_performance"

        # Check 3: Loss Spike
        loss_result = self._check_loss_spike(config)
        if loss_result:
            issues.append(loss_result["message"])
            if not root_cause:
                root_cause = RootCause.NAN_VALUE
            confidence = max(confidence, loss_result["confidence"])
            recommended_action = recommended_action or "check_training_config"

        # Check 4: Gradient Spike
        grad_result = self._check_gradient_spike(config)
        if grad_result:
            issues.append(grad_result["message"])
            if not root_cause:
                root_cause = RootCause.GRADIENT_EXPLOSION
            confidence = max(confidence, grad_result["confidence"])
            recommended_action = recommended_action or "reduce_learning_rate"

        if not issues:
            return DiagnosticResult(
                status=DiagnosticStatus.PASSED,
                diagnostic_name=self.name,
                confidence=1.0,
                message=f"No implicit failures detected. History: {len(self._history.losses)} steps",
                metrics={
                    "history_size": len(self._history.losses),
                    "last_step": self._history.last_step,
                    "baseline_mfu": self._baseline_mfu or 0.0,
                },
            )

        return DiagnosticResult(
            status=DiagnosticStatus.FAILED,
            diagnostic_name=self.name,
            root_cause=root_cause,
            root_cause_details="; ".join(issues),
            confidence=confidence,
            message=f"Detected {len(issues)} implicit failure(s)",
            recommended_action=recommended_action,
            metrics={
                "history_size": len(self._history.losses),
                "last_step": self._history.last_step,
                "baseline_mfu": self._baseline_mfu or 0.0,
                "current_mfu": self._history.mfus[-1] if self._history.mfus else 0.0,
            },
            suspect_details={
                "issues": issues,
                "recent_losses": list(self._history.losses)[-5:] if self._history.losses else [],
                "recent_gradient_norms": list(self._history.gradient_norms)[-5:] if self._history.gradient_norms else [],
            },
        )

    def _check_job_hang(self, config: ImplicitFailureConfig) -> Optional[Dict]:
        """Check if the job is hanging (no progress)."""
        if not self._history.timestamps:
            return None

        time_since_update = time.time() - self._history.last_update
        threshold = config.rdma_traffic_zero_threshold_seconds

        if time_since_update > threshold:
            return {
                "message": f"Job appears hung: no progress for {time_since_update:.0f}s (threshold: {threshold}s)",
                "confidence": min(0.95, 0.5 + (time_since_update / threshold) * 0.45),
            }
        return None

    def _check_mfu_decline(self, config: ImplicitFailureConfig) -> Optional[Dict]:
        """Check for MFU/throughput decline."""
        if self._baseline_mfu is None or len(self._history.mfus) < 10:
            return None

        # Check recent MFU against baseline
        recent_mfus = list(self._history.mfus)[-10:]
        recent_avg = mean(recent_mfus)

        decline_pct = (self._baseline_mfu - recent_avg) / self._baseline_mfu

        if decline_pct > config.mfu_decline_threshold:
            return {
                "message": f"MFU declined {decline_pct*100:.1f}% from baseline {self._baseline_mfu:.3f} to {recent_avg:.3f}",
                "confidence": min(0.9, 0.5 + decline_pct),
            }
        return None

    def _check_loss_spike(self, config: ImplicitFailureConfig) -> Optional[Dict]:
        """Check for sudden loss spikes."""
        if len(self._history.losses) < 10:
            return None

        losses = list(self._history.losses)
        recent_losses = losses[-5:]
        historical_losses = losses[:-5]

        if not historical_losses:
            return None

        historical_mean = mean(historical_losses)
        historical_std = stdev(historical_losses) if len(historical_losses) > 1 else historical_mean * 0.1

        for loss in recent_losses:
            if loss > historical_mean + (config.loss_spike_multiplier * historical_std):
                return {
                    "message": f"Loss spike detected: {loss:.4f} vs historical mean {historical_mean:.4f}",
                    "confidence": 0.8,
                }

        return None

    def _check_gradient_spike(self, config: ImplicitFailureConfig) -> Optional[Dict]:
        """Check for gradient norm spikes."""
        if len(self._history.gradient_norms) < 10:
            return None

        norms = list(self._history.gradient_norms)
        recent_norms = norms[-5:]
        historical_norms = norms[:-5]

        if not historical_norms:
            return None

        historical_mean = mean(historical_norms)
        historical_std = stdev(historical_norms) if len(historical_norms) > 1 else historical_mean * 0.1

        for norm in recent_norms:
            if norm > historical_mean + (config.gradient_norm_spike_multiplier * historical_std):
                return {
                    "message": f"Gradient norm spike: {norm:.4f} vs historical mean {historical_mean:.4f}",
                    "confidence": 0.75,
                }

        return None

    @property
    def history(self) -> MetricsHistory:
        """Get the metrics history."""
        return self._history

    def get_health_summary(self) -> Dict:
        """Get a summary of training health."""
        return {
            "steps_recorded": len(self._history.losses),
            "last_step": self._history.last_step,
            "last_update": self._history.last_update,
            "baseline_mfu": self._baseline_mfu,
            "recent_loss_avg": mean(list(self._history.losses)[-10:]) if len(self._history.losses) >= 10 else None,
            "recent_mfu_avg": mean(list(self._history.mfus)[-10:]) if len(self._history.mfus) >= 10 else None,
        }
