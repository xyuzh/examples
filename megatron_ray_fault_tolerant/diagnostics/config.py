"""Configuration dataclasses for the diagnostics system.

All diagnostic component configurations are defined here for centralized management.
"""

from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum


class RootCause(Enum):
    """Root cause categories for failure diagnosis."""
    # GPU/NCCL (15%)
    NCCL_TIMEOUT = "nccl_timeout"
    GPU_ERROR = "gpu_error"
    GPU_MEMORY_FAULT = "gpu_memory_fault"
    GPU_ECC_ERROR = "gpu_ecc_error"
    GPU_FALLEN_OFF_BUS = "gpu_fallen_off_bus"

    # Host resources (26%)
    CPU_OVERLOAD = "cpu_overload"
    CPU_OOM = "cpu_oom"
    INSUFFICIENT_DISK = "disk_space"

    # Network (8%)
    NETWORK = "network"
    INFINIBAND_ERROR = "ib_error"
    NIC_FLAP = "nic_flap"
    SWITCH_DOWN = "switch_down"

    # Infrastructure (6%)
    FILESYSTEM_MOUNT = "fs_mount"
    HDFS_ERROR = "hdfs_error"
    CONTAINER_ERROR = "container"
    OS_KERNEL_PANIC = "kernel_panic"
    EXTERNAL_SERVICE = "external_svc"

    # Implicit (1%)
    MFU_DECLINE = "mfu_decline"
    NAN_VALUE = "nan_value"
    HANG = "hang"
    GRADIENT_EXPLOSION = "gradient_explosion"

    # Hardware SDC
    SILENT_DATA_CORRUPTION = "sdc"

    # Unknown
    UNKNOWN = "unknown"


@dataclass
class DetectionTimeTargets:
    """Targets from ByteRobust production experience."""
    nic_crash_seconds: int = 30
    port_flapping_seconds: int = 30
    switch_down_seconds: int = 60  # 30s * 2 confirmations
    driver_hang_seconds: int = 10
    gpu_lost_seconds: int = 10
    os_kernel_fault_seconds: int = 2
    job_hang_detection_seconds: int = 600
    mfu_decline_detection_steps: int = 100


@dataclass
class OverEvictionConfig:
    """Settings for parallel-group over-eviction."""
    enabled: bool = True
    prefer_parallel_group: bool = True  # Evict entire PP group vs surgical
    mfu_aggregation_rounds: int = 5
    mfu_aggregation_interval_seconds: int = 10


@dataclass
class ImplicitFailureConfig:
    """Configuration for implicit failure detection."""
    enabled: bool = True

    # Job hang detection
    rdma_traffic_zero_threshold_seconds: int = 600  # 10 min
    tensorcore_utilization_min: float = 0.1

    # MFU decline detection
    mfu_decline_threshold: float = 0.2  # 20% drop
    mfu_window_steps: int = 100

    # Training trajectory anomaly
    loss_spike_multiplier: float = 5.0
    gradient_norm_spike_multiplier: float = 5.0

    # Rolling window for anomaly detection
    history_window_size: int = 50


@dataclass
class LogAnalyzerConfig:
    """Configuration for the log analyzer."""
    enabled: bool = True
    ring_buffer_size: int = 10000
    capture_python_logging: bool = True
    capture_stderr: bool = True

    # Confidence thresholds
    high_confidence_threshold: float = 0.9
    medium_confidence_threshold: float = 0.6


@dataclass
class EUDConfig:
    """Configuration for NVIDIA EUD (GPU health) monitoring."""
    enabled: bool = True

    # Polling settings (only active during diagnostics, not training)
    polling_interval_seconds: float = 30.0

    # Thresholds
    temperature_critical_celsius: int = 85
    ecc_uncorrected_threshold: int = 1  # Any uncorrected ECC is critical
    retired_pages_threshold: int = 10
    pcie_replay_threshold: int = 100

    # Critical XID codes that indicate hardware failure
    critical_xid_codes: List[int] = field(default_factory=lambda: [
        31,   # GPU memory page fault
        48,   # Double Bit ECC Error (CRITICAL)
        63,   # ECC page retirement failure
        64,   # GPU fallen off bus (CRITICAL)
        79,   # GPU fallen off bus (CRITICAL)
        119,  # GSP RPC timeout
    ])


@dataclass
class NCCLConnectivityConfig:
    """Configuration for NCCL connectivity tests."""
    enabled: bool = True

    # Quick test settings
    quick_test_size_bytes: int = 1024  # 1KB
    quick_test_timeout_ms: int = 5000

    # Full test settings
    full_test_size_bytes: int = 1024 * 1024  # 1MB
    full_test_timeout_ms: int = 30000

    # Latency thresholds (microseconds)
    latency_warning_us: int = 1000
    latency_critical_us: int = 5000


@dataclass
class NaNMonitorConfig:
    """Configuration for NaN/Inf monitoring."""
    enabled: bool = True

    # Sampling rate (0.0 = never, 1.0 = every step)
    sampling_rate: float = 0.01  # 1% of steps

    # Which tensors to check
    check_parameters: bool = True
    check_gradients: bool = True
    check_activations: bool = False  # Can be expensive

    # Alert settings
    alert_on_first_nan: bool = True
    trigger_sdc_test_on_nan: bool = True


@dataclass
class BitwiseAlignmentConfig:
    """Configuration for bitwise alignment (SDC) tests."""
    enabled: bool = True

    # Test matrix size
    matrix_size: int = 4096

    # Number of test iterations
    num_iterations: int = 3

    # Which operations to test
    test_gemm: bool = True
    test_attention: bool = True
    test_reduction: bool = True

    # Schedule: run at these points
    run_on_job_start: bool = True
    run_on_recovery: bool = True
    run_periodic_hours: float = 4.0  # Every 4 hours
    run_on_nan_detected: bool = True


@dataclass
class DualPhaseReplayConfig:
    """Configuration for dual-phase replay engine."""
    enabled: bool = True

    # Operation logging (Phase 1)
    enable_checksum_logging: bool = True
    checksum_ring_buffer_size: int = 10000

    # Tensor snapshot settings
    enable_tensor_snapshots: bool = True
    snapshot_every_n_steps: int = 100
    max_snapshots_kept: int = 5

    # Replay settings (Phase 2)
    group_size_multiplier: int = 4  # group_size = multiplier * PP_size


@dataclass
class StackAggregatorConfig:
    """Configuration for stack trace aggregation analysis."""
    enabled: bool = True

    # Stack collection timeout
    collection_timeout_seconds: float = 10.0

    # Clustering parameters
    similarity_threshold: float = 0.8
    min_cluster_size: int = 2


@dataclass
class AlertConfig:
    """Configuration for alerting."""
    webhook_url: Optional[str] = None

    # Alert levels
    alert_on_critical: bool = True
    alert_on_warning: bool = False

    # Cooldown to avoid alert storms
    cooldown_seconds: int = 60


@dataclass
class DiagnosticsConfig:
    """Main configuration for the diagnostics system."""
    enabled: bool = True

    # Artifact storage
    artifact_dir: str = "/tmp/diagnostics"
    cloud_artifact_dir: Optional[str] = None

    # Per-component configs
    log_analyzer: LogAnalyzerConfig = field(default_factory=LogAnalyzerConfig)
    eud: EUDConfig = field(default_factory=EUDConfig)
    nccl_connectivity: NCCLConnectivityConfig = field(default_factory=NCCLConnectivityConfig)
    nan_monitor: NaNMonitorConfig = field(default_factory=NaNMonitorConfig)
    bitwise_alignment: BitwiseAlignmentConfig = field(default_factory=BitwiseAlignmentConfig)
    dual_phase_replay: DualPhaseReplayConfig = field(default_factory=DualPhaseReplayConfig)
    stack_aggregator: StackAggregatorConfig = field(default_factory=StackAggregatorConfig)
    implicit_failure: ImplicitFailureConfig = field(default_factory=ImplicitFailureConfig)

    # Detection timing targets
    detection_targets: DetectionTimeTargets = field(default_factory=DetectionTimeTargets)

    # Over-eviction settings
    over_eviction: OverEvictionConfig = field(default_factory=OverEvictionConfig)

    # Alert config
    alert: AlertConfig = field(default_factory=AlertConfig)
