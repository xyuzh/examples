"""Stop-Time Diagnosis System for Megatron + Ray Fault-Tolerant Training.

This package provides a post-mortem diagnosis system that runs when training
stops unexpectedly. It identifies root causes without impacting training throughput.

Key Principles:
1. Zero overhead during normal training - diagnostics triggered at stop-time
2. Simple approaches first - address 80%+ of incidents with straightforward checks
3. Confidence-based routing - high-confidence issues auto-remediate, low-confidence escalate
4. Modular MVP - start simple, extend via plugin architecture

Components (Priority Order):
- P0: LogAnalyzer (60%), EUDMonitor (15%), ImplicitFailureDetector (10%)
- P1: StackAggregationAnalyzer (10%), NCCLConnectivityTest (5%)
- P2: SDCDetector (1%), DualPhaseReplayEngine (1%)

Usage:
    from diagnostics import DiagnosticsConfig, DiagnosticsMixin, DiagnosticsHooks
    from diagnostics.core import DiagnosticPhase, DiagnosticResult

    # Configure diagnostics
    config = DiagnosticsConfig(
        enabled=True,
        artifact_dir="/tmp/diagnostics",
        cloud_artifact_dir="s3://bucket/diagnostics",
    )

    # In MegatronActor (using mixin)
    class MegatronActor(DiagnosticsMixin):
        def init_worker_process_group(self):
            self.init_diagnostics(config)
            # ... existing code ...
            self.run_diagnostics(DiagnosticPhase.PRE_TRAINING)

    # In MegatronActorGroup (using hooks)
    hooks = DiagnosticsHooks(config)
    hooks.run_pre_training_diagnostics(actor_infos)

For more details, see the plan at:
    /Users/xinyu/.claude/plans/crispy-wibbling-hoare.md
"""

# Core components
from .config import (
    DiagnosticsConfig,
    RootCause,
    DetectionTimeTargets,
    OverEvictionConfig,
    ImplicitFailureConfig,
    LogAnalyzerConfig,
    EUDConfig,
    NCCLConnectivityConfig,
    NaNMonitorConfig,
    BitwiseAlignmentConfig,
    DualPhaseReplayConfig,
    StackAggregatorConfig,
    AlertConfig,
)

from .core import (
    BaseDiagnostic,
    DiagnosticPhase,
    DiagnosticResult,
    DiagnosticStatus,
    DiagnosticContext,
    DiagnosticCollector,
)

from .registry import DiagnosticRegistry, register_diagnostic

# Integration components
from .integration import DiagnosticsMixin, DiagnosticsHooks

# Storage
from .storage import ArtifactStore

# Analyzers (auto-registered via @register_diagnostic decorator)
from .analyzers import (
    LogAnalyzer,
    EUDMonitor,
    ImplicitFailureDetector,
    StackAggregationAnalyzer,
    NCCLConnectivityTest,
    NaNMonitor,
    SDCDetector,
    DualPhaseReplayEngine,
)

__all__ = [
    # Config
    "DiagnosticsConfig",
    "RootCause",
    "DetectionTimeTargets",
    "OverEvictionConfig",
    "ImplicitFailureConfig",
    "LogAnalyzerConfig",
    "EUDConfig",
    "NCCLConnectivityConfig",
    "NaNMonitorConfig",
    "BitwiseAlignmentConfig",
    "DualPhaseReplayConfig",
    "StackAggregatorConfig",
    "AlertConfig",
    # Core
    "BaseDiagnostic",
    "DiagnosticPhase",
    "DiagnosticResult",
    "DiagnosticStatus",
    "DiagnosticContext",
    "DiagnosticCollector",
    # Registry
    "DiagnosticRegistry",
    "register_diagnostic",
    # Integration
    "DiagnosticsMixin",
    "DiagnosticsHooks",
    # Storage
    "ArtifactStore",
    # Analyzers
    "LogAnalyzer",
    "EUDMonitor",
    "ImplicitFailureDetector",
    "StackAggregationAnalyzer",
    "NCCLConnectivityTest",
    "NaNMonitor",
    "SDCDetector",
    "DualPhaseReplayEngine",
]

__version__ = "0.1.0"
