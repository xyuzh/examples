"""SDC Detector - P2 diagnostic for Silent Data Corruption detection.

Uses bitwise alignment tests to detect hardware issues that cause
computation errors without explicit error signals.
"""

import hashlib
import time
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass

from ..core.base import BaseDiagnostic, DiagnosticPhase
from ..core.context import DiagnosticContext
from ..core.result import DiagnosticResult, DiagnosticStatus
from ..config import DiagnosticsConfig, RootCause, BitwiseAlignmentConfig
from ..registry import register_diagnostic

try:
    import torch
    import torch.distributed as dist
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class SDCTestResult:
    """Result of a single SDC test."""
    test_name: str
    rank: int
    hash_value: str
    duration_ms: float


@register_diagnostic
class SDCDetector(BaseDiagnostic):
    """Detects Silent Data Corruption through bitwise alignment tests.

    SDC occurs when GPU hardware produces incorrect computation results
    without generating any error signals. This is particularly dangerous
    because training can continue with corrupted data.

    The test works by:
    1. Broadcasting a seed to all GPUs
    2. Each GPU performs deterministic computation with same seed
    3. Computing MD5 hash of results
    4. Comparing hashes across all GPUs
    5. Mismatches indicate potential SDC

    Tests performed:
    - GEMM (tests tensor cores)
    - Attention (tests mixed precision)
    - Reduction (tests accumulation)

    Schedule (user-specified 4 trigger points):
    1. Job start - before training begins
    2. After eviction/recovery - verify replacement GPUs
    3. Every 4-8 hours - catch thermal/aging issues
    4. NaN detected - investigate if hardware caused NaN
    """

    name = "sdc_detector"
    phases = [DiagnosticPhase.PRE_TRAINING, DiagnosticPhase.ON_RECOVERY, DiagnosticPhase.ON_DEMAND]
    priority = 50  # Run after basic checks

    def __init__(self):
        self._config: Optional[BitwiseAlignmentConfig] = None
        self._last_periodic_check: float = 0.0

    def initialize(self, config: DiagnosticsConfig) -> None:
        self._config = config.bitwise_alignment

    def is_enabled(self, config: DiagnosticsConfig) -> bool:
        return config.bitwise_alignment.enabled

    def check(self, context: DiagnosticContext) -> DiagnosticResult:
        """Run SDC detection tests."""
        if not TORCH_AVAILABLE:
            return DiagnosticResult.skipped(self.name, "PyTorch not available")

        if not dist.is_initialized():
            return DiagnosticResult.skipped(self.name, "Distributed not initialized")

        config = self._config
        all_results: List[SDCTestResult] = []
        divergent_tests: List[Tuple[str, List[int]]] = []

        # Run each enabled test
        if config.test_gemm:
            gemm_divergent = self._run_gemm_test(context, all_results)
            if gemm_divergent:
                divergent_tests.append(("gemm", gemm_divergent))

        if config.test_attention:
            attn_divergent = self._run_attention_test(context, all_results)
            if attn_divergent:
                divergent_tests.append(("attention", attn_divergent))

        if config.test_reduction:
            reduce_divergent = self._run_reduction_test(context, all_results)
            if reduce_divergent:
                divergent_tests.append(("reduction", reduce_divergent))

        if not divergent_tests:
            return DiagnosticResult(
                status=DiagnosticStatus.PASSED,
                diagnostic_name=self.name,
                confidence=1.0,
                message=f"All SDC tests passed ({len(all_results)} tests)",
                metrics={
                    "tests_run": len(all_results),
                    "matrix_size": config.matrix_size,
                },
            )

        # Identify suspect GPUs (those with divergent hashes)
        suspect_ranks = set()
        for test_name, ranks in divergent_tests:
            suspect_ranks.update(ranks)

        return DiagnosticResult(
            status=DiagnosticStatus.FAILED,
            diagnostic_name=self.name,
            root_cause=RootCause.SILENT_DATA_CORRUPTION,
            root_cause_details=f"SDC detected in {len(divergent_tests)} test(s)",
            suspects=list(suspect_ranks),
            confidence=0.95,
            message=f"Silent Data Corruption detected on ranks {list(suspect_ranks)}",
            metrics={
                "tests_run": len(all_results),
                "divergent_tests": len(divergent_tests),
            },
            suspect_details={
                "divergent_tests": [
                    {"test": name, "divergent_ranks": ranks}
                    for name, ranks in divergent_tests
                ],
            },
            recommended_action="evict_faulty_gpu",
        )

    def _run_gemm_test(
        self,
        context: DiagnosticContext,
        results: List[SDCTestResult],
    ) -> Optional[List[int]]:
        """Run GEMM (matrix multiplication) test."""
        config = self._config
        size = config.matrix_size
        rank = context.rank

        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Use deterministic seed
            seed = 42
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

            # Create matrices with fixed seed
            A = torch.randn(size, size, device=device, dtype=torch.float32)
            B = torch.randn(size, size, device=device, dtype=torch.float32)

            start = time.perf_counter()

            # Perform matmul
            C = torch.matmul(A, B)

            # Compute hash
            hash_val = self._compute_tensor_hash(C)

            duration_ms = (time.perf_counter() - start) * 1000

            result = SDCTestResult(
                test_name="gemm",
                rank=rank,
                hash_value=hash_val,
                duration_ms=duration_ms,
            )
            results.append(result)

            # Gather all hashes
            return self._compare_hashes("gemm", hash_val, context)

        except Exception as e:
            results.append(SDCTestResult(
                test_name="gemm",
                rank=rank,
                hash_value=f"ERROR: {e}",
                duration_ms=0,
            ))
            return [rank]

    def _run_attention_test(
        self,
        context: DiagnosticContext,
        results: List[SDCTestResult],
    ) -> Optional[List[int]]:
        """Run attention computation test (mixed precision)."""
        config = self._config
        size = config.matrix_size // 4  # Smaller for attention
        rank = context.rank

        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            seed = 42
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

            # Simulate attention: Q, K, V
            batch_size = 4
            seq_len = size
            head_dim = 64

            Q = torch.randn(batch_size, seq_len, head_dim, device=device, dtype=torch.float16)
            K = torch.randn(batch_size, seq_len, head_dim, device=device, dtype=torch.float16)
            V = torch.randn(batch_size, seq_len, head_dim, device=device, dtype=torch.float16)

            start = time.perf_counter()

            # Compute attention
            scores = torch.matmul(Q, K.transpose(-2, -1)) / (head_dim ** 0.5)
            attn_weights = torch.softmax(scores, dim=-1)
            output = torch.matmul(attn_weights, V)

            # Hash the output
            hash_val = self._compute_tensor_hash(output.float())

            duration_ms = (time.perf_counter() - start) * 1000

            result = SDCTestResult(
                test_name="attention",
                rank=rank,
                hash_value=hash_val,
                duration_ms=duration_ms,
            )
            results.append(result)

            return self._compare_hashes("attention", hash_val, context)

        except Exception as e:
            results.append(SDCTestResult(
                test_name="attention",
                rank=rank,
                hash_value=f"ERROR: {e}",
                duration_ms=0,
            ))
            return [rank]

    def _run_reduction_test(
        self,
        context: DiagnosticContext,
        results: List[SDCTestResult],
    ) -> Optional[List[int]]:
        """Run reduction (accumulation) test."""
        config = self._config
        size = config.matrix_size
        rank = context.rank

        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            seed = 42
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

            # Create large tensor
            data = torch.randn(size * size, device=device, dtype=torch.float32)

            start = time.perf_counter()

            # Various reductions
            sum_val = data.sum()
            mean_val = data.mean()
            std_val = data.std()

            # Combine into hash
            result_tensor = torch.tensor([sum_val, mean_val, std_val], device=device)
            hash_val = self._compute_tensor_hash(result_tensor)

            duration_ms = (time.perf_counter() - start) * 1000

            result = SDCTestResult(
                test_name="reduction",
                rank=rank,
                hash_value=hash_val,
                duration_ms=duration_ms,
            )
            results.append(result)

            return self._compare_hashes("reduction", hash_val, context)

        except Exception as e:
            results.append(SDCTestResult(
                test_name="reduction",
                rank=rank,
                hash_value=f"ERROR: {e}",
                duration_ms=0,
            ))
            return [rank]

    def _compute_tensor_hash(self, tensor) -> str:
        """Compute MD5 hash of tensor bytes."""
        # Move to CPU and get bytes
        data = tensor.detach().cpu().numpy().tobytes()
        return hashlib.md5(data).hexdigest()

    def _compare_hashes(
        self,
        test_name: str,
        local_hash: str,
        context: DiagnosticContext,
    ) -> Optional[List[int]]:
        """Gather and compare hashes across all ranks.

        Returns list of divergent ranks, or None if all match.
        """
        world_size = context.world_size
        rank = context.rank

        # Create tensor to hold hash (as integer codes)
        hash_bytes = local_hash.encode('utf-8')
        hash_tensor = torch.tensor(
            [ord(c) for c in local_hash],
            dtype=torch.int32,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        # Pad to fixed length (32 chars for MD5)
        if len(hash_tensor) < 32:
            hash_tensor = torch.nn.functional.pad(hash_tensor, (0, 32 - len(hash_tensor)))

        # Gather all hashes
        all_hashes = [torch.zeros_like(hash_tensor) for _ in range(world_size)]
        dist.all_gather(all_hashes, hash_tensor)

        # Convert back to strings and compare
        hash_strings = []
        for h in all_hashes:
            chars = [chr(c.item()) for c in h if c.item() > 0]
            hash_strings.append(''.join(chars))

        # Find majority hash
        hash_counts: Dict[str, List[int]] = {}
        for i, h in enumerate(hash_strings):
            if h not in hash_counts:
                hash_counts[h] = []
            hash_counts[h].append(i)

        if len(hash_counts) == 1:
            # All hashes match
            return None

        # Find divergent ranks (not in majority)
        majority_hash = max(hash_counts.keys(), key=lambda k: len(hash_counts[k]))
        divergent_ranks = []
        for h, ranks in hash_counts.items():
            if h != majority_hash:
                divergent_ranks.extend(ranks)

        return divergent_ranks

    def should_run_periodic(self) -> bool:
        """Check if periodic SDC test should run."""
        if not self._config.enabled:
            return False

        hours_since_last = (time.time() - self._last_periodic_check) / 3600

        if hours_since_last >= self._config.run_periodic_hours:
            self._last_periodic_check = time.time()
            return True

        return False
