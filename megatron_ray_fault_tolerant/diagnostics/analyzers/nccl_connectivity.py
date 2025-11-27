"""NCCL Connectivity Test - P1 diagnostic for network verification.

Verifies all-to-all NCCL connectivity before training and after recovery.
Covers ~5% of failures related to network connectivity issues.
"""

import time
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass

from ..core.base import BaseDiagnostic, DiagnosticPhase
from ..core.context import DiagnosticContext
from ..core.result import DiagnosticResult, DiagnosticStatus
from ..config import DiagnosticsConfig, RootCause, NCCLConnectivityConfig
from ..registry import register_diagnostic

# Try to import torch distributed
try:
    import torch
    import torch.distributed as dist
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class ConnectivityResult:
    """Result of a connectivity test between two ranks."""
    source_rank: int
    dest_rank: int
    success: bool
    latency_us: float
    error_message: Optional[str] = None


@register_diagnostic
class NCCLConnectivityTest(BaseDiagnostic):
    """Tests NCCL connectivity between all pairs of ranks.

    This diagnostic runs:
    1. **Quick Test** (~100ms): Small allreduce to verify basic connectivity
    2. **Full Test** (~5s): Pairwise send/recv with latency measurements

    The test identifies:
    - Completely broken connections (no communication possible)
    - High latency links (possible switch/NIC issues)
    - Asymmetric failures (one direction works, other doesn't)

    Output includes a connectivity matrix and list of failing pairs.
    """

    name = "nccl_connectivity"
    phases = [DiagnosticPhase.PRE_TRAINING, DiagnosticPhase.ON_RECOVERY, DiagnosticPhase.ON_DEMAND]
    priority = 15  # Run early in pre-training

    def __init__(self):
        self._config: Optional[NCCLConnectivityConfig] = None

    def initialize(self, config: DiagnosticsConfig) -> None:
        self._config = config.nccl_connectivity

    def is_enabled(self, config: DiagnosticsConfig) -> bool:
        return config.nccl_connectivity.enabled

    def check(self, context: DiagnosticContext) -> DiagnosticResult:
        """Run connectivity tests."""
        if not TORCH_AVAILABLE:
            return DiagnosticResult.skipped(
                self.name,
                reason="PyTorch not available"
            )

        if not dist.is_initialized():
            return DiagnosticResult.skipped(
                self.name,
                reason="Distributed not initialized"
            )

        config = self._config

        # Run quick test first
        quick_result = self._run_quick_test(config, context)

        if quick_result.status == DiagnosticStatus.FAILED:
            # Quick test failed - don't bother with full test
            return quick_result

        # If requested (e.g., after recovery), run full test
        if context.extra.get("run_full_test", False):
            return self._run_full_test(config, context)

        return quick_result

    def _run_quick_test(
        self,
        config: NCCLConnectivityConfig,
        context: DiagnosticContext,
    ) -> DiagnosticResult:
        """Run a quick allreduce test to verify basic connectivity."""
        try:
            # Create a small tensor
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            data = torch.ones(config.quick_test_size_bytes // 4, device=device) * context.rank

            # Time the allreduce
            start = time.perf_counter()

            # Use barrier first to sync all processes
            dist.barrier()

            # Run allreduce
            dist.all_reduce(data, op=dist.ReduceOp.SUM)

            elapsed_us = (time.perf_counter() - start) * 1e6

            # Verify result: sum of all ranks should be 0 + 1 + 2 + ... + (n-1) = n*(n-1)/2
            expected_sum = context.world_size * (context.world_size - 1) / 2
            actual_sum = data[0].item()

            if abs(actual_sum - expected_sum) > 0.01:
                return DiagnosticResult(
                    status=DiagnosticStatus.FAILED,
                    diagnostic_name=self.name,
                    root_cause=RootCause.NCCL_TIMEOUT,
                    root_cause_details=f"Allreduce result mismatch: expected {expected_sum}, got {actual_sum}",
                    confidence=0.9,
                    message="Quick connectivity test failed: result mismatch",
                    metrics={
                        "expected_sum": expected_sum,
                        "actual_sum": actual_sum,
                        "latency_us": elapsed_us,
                    },
                )

            # Check latency
            status = DiagnosticStatus.PASSED
            message = f"Quick connectivity test passed in {elapsed_us:.0f}us"

            if elapsed_us > config.latency_critical_us * 1000:  # Convert to ms for quick test
                status = DiagnosticStatus.WARNING
                message = f"Quick test passed but high latency: {elapsed_us:.0f}us"

            return DiagnosticResult(
                status=status,
                diagnostic_name=self.name,
                confidence=1.0 if status == DiagnosticStatus.PASSED else 0.7,
                message=message,
                metrics={
                    "latency_us": elapsed_us,
                    "world_size": context.world_size,
                },
            )

        except Exception as e:
            return DiagnosticResult(
                status=DiagnosticStatus.FAILED,
                diagnostic_name=self.name,
                root_cause=RootCause.NCCL_TIMEOUT,
                root_cause_details=str(e),
                confidence=0.95,
                message=f"Quick connectivity test failed with exception: {e}",
            )

    def _run_full_test(
        self,
        config: NCCLConnectivityConfig,
        context: DiagnosticContext,
    ) -> DiagnosticResult:
        """Run full pairwise connectivity test."""
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            rank = context.rank
            world_size = context.world_size

            results: List[ConnectivityResult] = []
            failures: List[ConnectivityResult] = []
            high_latency: List[ConnectivityResult] = []

            # Test send/recv with each other rank
            for other_rank in range(world_size):
                if other_rank == rank:
                    continue

                # Create test data
                send_data = torch.ones(config.full_test_size_bytes // 4, device=device) * rank
                recv_data = torch.zeros(config.full_test_size_bytes // 4, device=device)

                try:
                    start = time.perf_counter()

                    # Odd ranks send first, even ranks receive first
                    if rank < other_rank:
                        dist.send(send_data, dst=other_rank)
                        dist.recv(recv_data, src=other_rank)
                    else:
                        dist.recv(recv_data, src=other_rank)
                        dist.send(send_data, dst=other_rank)

                    elapsed_us = (time.perf_counter() - start) * 1e6

                    # Verify data
                    if not torch.allclose(recv_data, torch.ones_like(recv_data) * other_rank):
                        result = ConnectivityResult(
                            source_rank=other_rank,
                            dest_rank=rank,
                            success=False,
                            latency_us=elapsed_us,
                            error_message="Data verification failed",
                        )
                        failures.append(result)
                    else:
                        result = ConnectivityResult(
                            source_rank=other_rank,
                            dest_rank=rank,
                            success=True,
                            latency_us=elapsed_us,
                        )
                        if elapsed_us > config.latency_critical_us:
                            high_latency.append(result)

                    results.append(result)

                except Exception as e:
                    result = ConnectivityResult(
                        source_rank=other_rank,
                        dest_rank=rank,
                        success=False,
                        latency_us=-1,
                        error_message=str(e),
                    )
                    failures.append(result)
                    results.append(result)

            # Aggregate results
            if failures:
                failing_pairs = [(f.source_rank, f.dest_rank) for f in failures]
                # Identify suspect ranks (those appearing most in failures)
                rank_failure_count: Dict[int, int] = {}
                for f in failures:
                    rank_failure_count[f.source_rank] = rank_failure_count.get(f.source_rank, 0) + 1
                    rank_failure_count[f.dest_rank] = rank_failure_count.get(f.dest_rank, 0) + 1

                # Ranks with most failures are suspects
                max_failures = max(rank_failure_count.values())
                suspects = [r for r, c in rank_failure_count.items() if c == max_failures]

                return DiagnosticResult(
                    status=DiagnosticStatus.FAILED,
                    diagnostic_name=self.name,
                    root_cause=RootCause.NETWORK,
                    root_cause_details=f"{len(failures)} connectivity failures detected",
                    suspects=suspects,
                    confidence=0.85,
                    message=f"Full connectivity test failed: {len(failures)} failing pairs",
                    metrics={
                        "total_tests": len(results),
                        "failures": len(failures),
                        "high_latency_pairs": len(high_latency),
                    },
                    suspect_details={
                        "failing_pairs": failing_pairs,
                        "rank_failure_counts": rank_failure_count,
                    },
                    recommended_action="investigate_network",
                )

            # All passed - check for warnings
            status = DiagnosticStatus.PASSED
            message = f"Full connectivity test passed: {len(results)} pairs tested"

            if high_latency:
                status = DiagnosticStatus.WARNING
                message = f"Connectivity OK but {len(high_latency)} high latency pairs"

            return DiagnosticResult(
                status=status,
                diagnostic_name=self.name,
                confidence=1.0 if status == DiagnosticStatus.PASSED else 0.8,
                message=message,
                metrics={
                    "total_tests": len(results),
                    "high_latency_pairs": len(high_latency),
                    "avg_latency_us": sum(r.latency_us for r in results if r.success) / len([r for r in results if r.success]) if results else 0,
                },
                suspect_details={
                    "high_latency_pairs": [(r.source_rank, r.dest_rank, r.latency_us) for r in high_latency],
                } if high_latency else {},
            )

        except Exception as e:
            return DiagnosticResult.error(self.name, f"Full connectivity test error: {e}")

    def run_quick_check(self, context: DiagnosticContext) -> DiagnosticResult:
        """Public method to run just the quick test."""
        return self._run_quick_test(self._config, context)

    def run_full_check(self, context: DiagnosticContext) -> DiagnosticResult:
        """Public method to run the full pairwise test."""
        return self._run_full_test(self._config, context)
