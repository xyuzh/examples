"""NaN Monitor - Detects and diagnoses numerical instabilities.

Triggered when training encounters NaN/Inf values, not continuous monitoring.
Helps identify whether the cause is software (training config) or hardware (SDC).
"""

import time
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass

from ..core.base import BaseDiagnostic, DiagnosticPhase
from ..core.context import DiagnosticContext
from ..core.result import DiagnosticResult, DiagnosticStatus
from ..config import DiagnosticsConfig, RootCause, NaNMonitorConfig
from ..registry import register_diagnostic

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class TensorStats:
    """Statistics for a tensor."""
    name: str
    shape: Tuple[int, ...]
    dtype: str
    has_nan: bool
    has_inf: bool
    nan_count: int
    inf_count: int
    min_val: float
    max_val: float
    mean_val: float
    std_val: float


@dataclass
class NaNReport:
    """Report of NaN/Inf detection."""
    parameter_stats: List[TensorStats]
    gradient_stats: List[TensorStats]
    total_nan_params: int
    total_nan_grads: int
    first_nan_layer: Optional[str]
    suspected_cause: str
    confidence: float


@register_diagnostic
class NaNMonitor(BaseDiagnostic):
    """Monitors for NaN/Inf values in model parameters and gradients.

    This diagnostic is triggered when NaN is detected (not continuous).
    It inspects:
    - All model parameters for NaN/Inf
    - Gradient buffers
    - Tensor statistics (min, max, mean, std)

    Source Identification Heuristics:
    - Attention layer NaN → softmax overflow (very negative scores)
    - Loss NaN → log(0) or division by zero
    - Gradient Inf → learning rate too high, exploding gradients
    - Random parameter NaN → possible SDC (hardware issue)

    Can trigger SDC test if hardware issue is suspected.
    """

    name = "nan_monitor"
    phases = [DiagnosticPhase.ON_DEMAND, DiagnosticPhase.ON_FAILURE]
    priority = 25

    def __init__(self):
        self._config: Optional[NaNMonitorConfig] = None
        self._last_check_step: int = -1
        self._model_ref: Optional[Any] = None

    def initialize(self, config: DiagnosticsConfig) -> None:
        self._config = config.nan_monitor

    def is_enabled(self, config: DiagnosticsConfig) -> bool:
        return config.nan_monitor.enabled

    def set_model(self, model) -> None:
        """Set the model reference for parameter inspection."""
        self._model_ref = model

    def check(self, context: DiagnosticContext) -> DiagnosticResult:
        """Check for NaN/Inf in model parameters and gradients."""
        if not TORCH_AVAILABLE:
            return DiagnosticResult.skipped(self.name, "PyTorch not available")

        model = context.extra.get("model") or self._model_ref
        if model is None:
            return DiagnosticResult.skipped(self.name, "No model reference available")

        report = self._analyze_model(model)

        if report.total_nan_params == 0 and report.total_nan_grads == 0:
            return DiagnosticResult(
                status=DiagnosticStatus.PASSED,
                diagnostic_name=self.name,
                confidence=1.0,
                message="No NaN/Inf values detected",
                metrics={
                    "params_checked": len(report.parameter_stats),
                    "grads_checked": len(report.gradient_stats),
                },
            )

        # Determine root cause based on where NaN was found
        root_cause = self._determine_root_cause(report)

        return DiagnosticResult(
            status=DiagnosticStatus.FAILED,
            diagnostic_name=self.name,
            root_cause=root_cause,
            root_cause_details=report.suspected_cause,
            confidence=report.confidence,
            message=f"NaN detected: {report.total_nan_params} params, {report.total_nan_grads} grads",
            metrics={
                "nan_params": report.total_nan_params,
                "nan_grads": report.total_nan_grads,
                "first_nan_layer": report.first_nan_layer or "unknown",
            },
            suspect_details={
                "nan_parameters": [
                    {"name": s.name, "shape": s.shape, "nan_count": s.nan_count}
                    for s in report.parameter_stats if s.has_nan or s.has_inf
                ][:10],  # First 10
                "nan_gradients": [
                    {"name": s.name, "shape": s.shape, "nan_count": s.nan_count}
                    for s in report.gradient_stats if s.has_nan or s.has_inf
                ][:10],
            },
            recommended_action="trigger_sdc_test" if root_cause == RootCause.SILENT_DATA_CORRUPTION else "check_training_config",
        )

    def _analyze_model(self, model) -> NaNReport:
        """Analyze model for NaN/Inf values."""
        param_stats = []
        grad_stats = []
        total_nan_params = 0
        total_nan_grads = 0
        first_nan_layer = None

        for name, param in model.named_parameters():
            # Check parameter
            stats = self._get_tensor_stats(name, param.data)
            param_stats.append(stats)

            if stats.has_nan or stats.has_inf:
                total_nan_params += 1
                if first_nan_layer is None:
                    first_nan_layer = name

            # Check gradient if exists
            if param.grad is not None:
                grad_name = f"{name}.grad"
                grad_stats_item = self._get_tensor_stats(grad_name, param.grad)
                grad_stats.append(grad_stats_item)

                if grad_stats_item.has_nan or grad_stats_item.has_inf:
                    total_nan_grads += 1
                    if first_nan_layer is None:
                        first_nan_layer = grad_name

        # Determine suspected cause
        suspected_cause, confidence = self._diagnose_cause(
            param_stats, grad_stats, first_nan_layer
        )

        return NaNReport(
            parameter_stats=param_stats,
            gradient_stats=grad_stats,
            total_nan_params=total_nan_params,
            total_nan_grads=total_nan_grads,
            first_nan_layer=first_nan_layer,
            suspected_cause=suspected_cause,
            confidence=confidence,
        )

    def _get_tensor_stats(self, name: str, tensor) -> TensorStats:
        """Get statistics for a tensor."""
        has_nan = bool(torch.isnan(tensor).any().item())
        has_inf = bool(torch.isinf(tensor).any().item())
        nan_count = int(torch.isnan(tensor).sum().item())
        inf_count = int(torch.isinf(tensor).sum().item())

        # Get finite values for stats
        finite_mask = torch.isfinite(tensor)
        finite_vals = tensor[finite_mask]

        if finite_vals.numel() > 0:
            min_val = float(finite_vals.min().item())
            max_val = float(finite_vals.max().item())
            mean_val = float(finite_vals.mean().item())
            std_val = float(finite_vals.std().item()) if finite_vals.numel() > 1 else 0.0
        else:
            min_val = max_val = mean_val = std_val = float('nan')

        return TensorStats(
            name=name,
            shape=tuple(tensor.shape),
            dtype=str(tensor.dtype),
            has_nan=has_nan,
            has_inf=has_inf,
            nan_count=nan_count,
            inf_count=inf_count,
            min_val=min_val,
            max_val=max_val,
            mean_val=mean_val,
            std_val=std_val,
        )

    def _diagnose_cause(
        self,
        param_stats: List[TensorStats],
        grad_stats: List[TensorStats],
        first_nan_layer: Optional[str],
    ) -> Tuple[str, float]:
        """Diagnose the likely cause of NaN values."""
        if first_nan_layer is None:
            return "Unknown cause", 0.5

        name_lower = first_nan_layer.lower()

        # Check for attention layer NaN
        if "attention" in name_lower or "attn" in name_lower:
            if "softmax" in name_lower or "score" in name_lower:
                return "Softmax overflow in attention (very negative scores)", 0.85
            return "Attention computation overflow", 0.7

        # Check for loss-related NaN
        if "loss" in name_lower or "output" in name_lower:
            return "Loss computation error (log(0) or division by zero)", 0.8

        # Check for gradient explosion
        if ".grad" in first_nan_layer:
            # Check if we have very large gradients preceding the NaN
            for stats in grad_stats:
                if not stats.has_nan and abs(stats.max_val) > 1e6:
                    return "Gradient explosion (learning rate too high)", 0.75

        # Check for embedding/input layer NaN
        if "embed" in name_lower or "input" in name_lower:
            return "Input or embedding corruption", 0.7

        # Random NaN in the middle of the network
        nan_param_pct = sum(1 for s in param_stats if s.has_nan) / len(param_stats) if param_stats else 0

        if nan_param_pct < 0.1:
            # Only a few parameters have NaN - could be SDC
            return "Isolated NaN (possible Silent Data Corruption)", 0.6

        return "Numerical instability (check hyperparameters)", 0.5

    def _determine_root_cause(self, report: NaNReport) -> RootCause:
        """Determine the root cause enum from the report."""
        cause = report.suspected_cause.lower()

        if "silent data corruption" in cause or "sdc" in cause:
            return RootCause.SILENT_DATA_CORRUPTION
        elif "gradient explosion" in cause:
            return RootCause.GRADIENT_EXPLOSION
        else:
            return RootCause.NAN_VALUE

    def check_tensor(self, name: str, tensor) -> Optional[TensorStats]:
        """Check a single tensor for NaN/Inf. Returns stats if issues found."""
        if not TORCH_AVAILABLE:
            return None

        stats = self._get_tensor_stats(name, tensor)
        if stats.has_nan or stats.has_inf:
            return stats
        return None
