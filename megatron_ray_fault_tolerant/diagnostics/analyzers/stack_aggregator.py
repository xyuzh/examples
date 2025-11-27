"""Stack Aggregation Analyzer - P1 diagnostic for stack-based failure detection.

Uses stack trace clustering to identify outlier ranks - the key insight is that
healthy machines typically have identical stacks, while faulty ones are outliers.
"""

import sys
import traceback
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from collections import defaultdict
import hashlib

from ..core.base import BaseDiagnostic, DiagnosticPhase
from ..core.context import DiagnosticContext
from ..core.result import DiagnosticResult, DiagnosticStatus
from ..config import DiagnosticsConfig, RootCause, StackAggregatorConfig
from ..registry import register_diagnostic


@dataclass
class StackTrace:
    """Represents a stack trace from a process."""
    rank: int
    thread_id: int
    frames: List[str]
    frame_hash: str = ""

    def __post_init__(self):
        # Compute hash for clustering
        # Normalize frames to remove line numbers for fuzzy matching
        normalized = []
        for frame in self.frames:
            # Remove line numbers and memory addresses
            import re
            normalized_frame = re.sub(r'line \d+', 'line N', frame)
            normalized_frame = re.sub(r'0x[0-9a-fA-F]+', '0xNNN', normalized_frame)
            normalized.append(normalized_frame)
        self.frame_hash = hashlib.md5("".join(normalized).encode()).hexdigest()[:16]


@dataclass
class StackCluster:
    """A cluster of similar stack traces."""
    hash_key: str
    ranks: Set[int] = field(default_factory=set)
    representative_stack: Optional[StackTrace] = None

    def add(self, stack: StackTrace) -> None:
        self.ranks.add(stack.rank)
        if self.representative_stack is None:
            self.representative_stack = stack


@register_diagnostic
class StackAggregationAnalyzer(BaseDiagnostic):
    """Analyzes stack traces across all ranks to find outliers.

    Key Insight: In distributed training, most healthy machines will have
    identical or very similar stack traces. Outliers (machines with different
    stacks) are likely the faulty ones.

    This is particularly useful for:
    - Detecting which rank is stuck in a different code path
    - Finding the process that's blocked/hanging
    - Identifying the machine that caused a distributed failure

    The analyzer clusters stacks by similarity and flags outliers.
    """

    name = "stack_aggregation"
    phases = [DiagnosticPhase.ON_FAILURE, DiagnosticPhase.ON_DEMAND]
    priority = 30  # Run after initial diagnostics

    def __init__(self):
        self._config: Optional[StackAggregatorConfig] = None

    def initialize(self, config: DiagnosticsConfig) -> None:
        self._config = config.stack_aggregator

    def is_enabled(self, config: DiagnosticsConfig) -> bool:
        return config.stack_aggregator.enabled

    def check(self, context: DiagnosticContext) -> DiagnosticResult:
        """Analyze stack traces and find outliers.

        In a distributed setting, this should be called with collected
        stacks from all ranks. For local testing, analyzes current process.
        """
        # Get stacks - in distributed mode, these would be collected from all ranks
        # For now, collect local stacks
        local_stacks = self._collect_local_stacks(context.rank)

        # If we have pre-collected stacks in context, use those
        all_stacks = context.extra.get("collected_stacks", local_stacks)

        if not all_stacks:
            return DiagnosticResult.skipped(
                self.name,
                reason="No stack traces available for analysis"
            )

        # Cluster the stacks
        clusters = self._cluster_stacks(all_stacks)

        # Find outliers
        result = self._identify_outliers(clusters, context)

        return result

    def _collect_local_stacks(self, rank: int) -> List[StackTrace]:
        """Collect stack traces from all threads in the current process."""
        stacks = []

        for thread_id, frame in sys._current_frames().items():
            frames = []
            while frame:
                frames.append(
                    f"File {frame.f_code.co_filename}, line {frame.f_lineno}, in {frame.f_code.co_name}"
                )
                frame = frame.f_back

            stacks.append(StackTrace(
                rank=rank,
                thread_id=thread_id,
                frames=frames,
            ))

        return stacks

    def _cluster_stacks(self, stacks: List[StackTrace]) -> List[StackCluster]:
        """Cluster stacks by similarity using frame hashes."""
        clusters: Dict[str, StackCluster] = {}

        for stack in stacks:
            hash_key = stack.frame_hash

            if hash_key not in clusters:
                clusters[hash_key] = StackCluster(hash_key=hash_key)

            clusters[hash_key].add(stack)

        return list(clusters.values())

    def _identify_outliers(
        self,
        clusters: List[StackCluster],
        context: DiagnosticContext,
    ) -> DiagnosticResult:
        """Identify outlier ranks based on stack clustering."""
        if not clusters:
            return DiagnosticResult.passed(self.name, "No stacks to analyze")

        # Find the majority cluster (healthy baseline)
        sorted_clusters = sorted(clusters, key=lambda c: len(c.ranks), reverse=True)
        majority_cluster = sorted_clusters[0]

        # Calculate what percentage of ranks are in the majority
        all_ranks = set()
        for cluster in clusters:
            all_ranks.update(cluster.ranks)

        majority_pct = len(majority_cluster.ranks) / len(all_ranks) if all_ranks else 0

        # Find outlier ranks (not in majority cluster)
        outlier_ranks = []
        outlier_clusters = []
        for cluster in clusters:
            if cluster != majority_cluster:
                outlier_ranks.extend(cluster.ranks)
                outlier_clusters.append(cluster)

        if not outlier_ranks:
            return DiagnosticResult(
                status=DiagnosticStatus.PASSED,
                diagnostic_name=self.name,
                confidence=1.0,
                message=f"All {len(all_ranks)} ranks have consistent stacks",
                metrics={
                    "total_ranks": len(all_ranks),
                    "cluster_count": len(clusters),
                    "majority_percentage": majority_pct * 100,
                },
            )

        # Outliers found - compute confidence based on cluster distribution
        # Higher confidence when majority is very clear
        confidence = min(0.9, 0.5 + (majority_pct * 0.4))

        # Find the parallel group for over-eviction if applicable
        suspects = sorted(outlier_ranks)
        recommended_action = "investigate_outlier_ranks"

        # If we have mesh rank info, recommend over-eviction of parallel group
        if context.mesh_rank is not None:
            recommended_action = "over_evict_parallel_group"

        # Build detailed info about outlier stacks
        outlier_details = []
        for cluster in outlier_clusters:
            if cluster.representative_stack:
                outlier_details.append({
                    "ranks": list(cluster.ranks),
                    "stack_preview": cluster.representative_stack.frames[:5],
                })

        return DiagnosticResult(
            status=DiagnosticStatus.FAILED,
            diagnostic_name=self.name,
            root_cause=RootCause.HANG,  # Most common cause of stack divergence
            root_cause_details=f"Found {len(outlier_ranks)} outlier rank(s) with divergent stacks",
            suspects=suspects,
            confidence=confidence,
            recommended_action=recommended_action,
            message=f"{len(outlier_ranks)} of {len(all_ranks)} ranks have divergent stacks",
            metrics={
                "total_ranks": len(all_ranks),
                "cluster_count": len(clusters),
                "majority_size": len(majority_cluster.ranks),
                "majority_percentage": majority_pct * 100,
                "outlier_count": len(outlier_ranks),
            },
            suspect_details={
                "outlier_ranks": outlier_ranks,
                "majority_ranks": list(majority_cluster.ranks),
                "outlier_stacks": outlier_details[:3],  # First 3 for brevity
            },
        )

    def _find_shared_parallel_group(
        self,
        outlier_ranks: List[int],
        mesh_rank,  # MeshRank type
    ) -> Set[int]:
        """Find the parallel group that contains the outlier ranks.

        For over-eviction, we want to evict the entire PP group containing
        the faulty rank, not just the single rank.
        """
        # This would need access to the full mesh topology
        # For now, return just the outlier ranks
        # In production, this would map ranks to their PP/TP groups
        return set(outlier_ranks)

    @staticmethod
    def collect_distributed_stacks(actor_handles, timeout: float = 10.0) -> List[StackTrace]:
        """Collect stacks from all actors in the distributed system.

        This is a helper method to be called from the MegatronActorGroup
        to gather stacks from all workers.
        """
        # This would be implemented using Ray's actor interface
        # to collect stacks from all workers in parallel
        # For now, return empty list - actual implementation in hooks.py
        return []
