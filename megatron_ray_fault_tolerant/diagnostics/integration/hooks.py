"""Diagnostic hooks for MegatronActorGroup.

Provides integration points for running diagnostics at the group level,
collecting distributed diagnostic data, and managing recovery diagnostics.
"""

import time
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from loguru import logger

from ..core.base import DiagnosticPhase
from ..core.context import DiagnosticContext
from ..core.result import DiagnosticResult, DiagnosticStatus
from ..core.collector import DiagnosticCollector
from ..config import DiagnosticsConfig
from ..storage.artifact_store import ArtifactStore, DiagnosticReport
from ..analyzers.stack_aggregator import StackTrace

if TYPE_CHECKING:
    from dispatch import ActorInfo


class DiagnosticsHooks:
    """Hooks for integrating diagnostics with MegatronActorGroup.

    This class provides methods that can be called from MegatronActorGroup
    to run diagnostics at various points:

    - pre_training: Before training starts (SDC test, connectivity)
    - post_step: After each training step (lightweight checks)
    - on_failure: When a failure is detected (full analysis)
    - on_recovery: After recovery (verify new GPUs)

    Example integration:
        class MegatronActorGroup:
            def __init__(self, ...):
                self.diag_hooks = DiagnosticsHooks(config)

            def initiate_worker_process_group(self):
                # ... existing code ...
                self.diag_hooks.run_pre_training_diagnostics(self.actor_infos)

            def recover_from_failure(self, backup_group):
                # Collect diagnostics before recovery
                failure_report = self.diag_hooks.collect_failure_diagnostics(self.actor_infos)

                # ... recovery logic ...

                # Verify recovery
                self.diag_hooks.run_recovery_diagnostics(self.actor_infos)
    """

    def __init__(
        self,
        config: Optional[DiagnosticsConfig] = None,
    ):
        self.config = config or DiagnosticsConfig()
        self.store = ArtifactStore(
            local_dir=self.config.artifact_dir,
            cloud_dir=self.config.cloud_artifact_dir,
        ) if self.config.enabled else None

    def run_pre_training_diagnostics(
        self,
        actor_infos: List["ActorInfo"],
    ) -> Dict[str, Any]:
        """Run pre-training diagnostics on all actors.

        This should be called after process group initialization but
        before training starts. Runs:
        - SDC test (if enabled)
        - NCCL connectivity test
        - GPU health check

        Args:
            actor_infos: List of actor information

        Returns:
            Summary of diagnostic results
        """
        if not self.config.enabled:
            return {"enabled": False}

        import ray

        logger.info("Running pre-training diagnostics...")

        # Run diagnostics on all actors in parallel
        results_refs = []
        for actor_info in actor_infos:
            ref = actor_info.handle.run_diagnostics.remote(
                DiagnosticPhase.PRE_TRAINING
            )
            results_refs.append((actor_info.rank, ref))

        # Collect results
        all_results = {}
        failures = []

        for rank, ref in results_refs:
            try:
                results = ray.get(ref, timeout=60)
                all_results[str(rank)] = [r.to_dict() for r in results]

                # Check for failures
                for r in results:
                    if r.status == DiagnosticStatus.FAILED:
                        failures.append({
                            "rank": rank,
                            "diagnostic": r.diagnostic_name,
                            "root_cause": r.root_cause.value if r.root_cause else None,
                            "confidence": r.confidence,
                        })
            except Exception as e:
                logger.error(f"Failed to get diagnostics from rank {rank}: {e}")
                failures.append({
                    "rank": rank,
                    "diagnostic": "collection_error",
                    "error": str(e),
                })

        summary = {
            "phase": "pre_training",
            "total_actors": len(actor_infos),
            "failures": failures,
            "passed": len(failures) == 0,
            "timestamp": time.time(),
        }

        logger.info(f"Pre-training diagnostics complete: {'PASSED' if summary['passed'] else 'FAILED'}")

        return summary

    def run_recovery_diagnostics(
        self,
        actor_infos: List["ActorInfo"],
    ) -> Dict[str, Any]:
        """Run diagnostics after recovery.

        This should be called after recovery is complete to verify
        that the new configuration is healthy. Runs:
        - SDC test on new GPUs
        - NCCL connectivity test
        - Verify model state

        Args:
            actor_infos: List of actor information (including new actors)

        Returns:
            Summary of diagnostic results
        """
        if not self.config.enabled:
            return {"enabled": False}

        import ray

        logger.info("Running recovery diagnostics...")

        # Run diagnostics with full connectivity test
        results_refs = []
        for actor_info in actor_infos:
            ref = actor_info.handle.run_diagnostics.remote(
                DiagnosticPhase.ON_RECOVERY,
                extra_context={"run_full_test": True},
            )
            results_refs.append((actor_info.rank, ref))

        # Collect results
        failures = []
        for rank, ref in results_refs:
            try:
                results = ray.get(ref, timeout=120)
                for r in results:
                    if r.status == DiagnosticStatus.FAILED:
                        failures.append({
                            "rank": rank,
                            "diagnostic": r.diagnostic_name,
                            "root_cause": r.root_cause.value if r.root_cause else None,
                        })
            except Exception as e:
                logger.error(f"Failed to get recovery diagnostics from rank {rank}: {e}")
                failures.append({
                    "rank": rank,
                    "diagnostic": "collection_error",
                    "error": str(e),
                })

        summary = {
            "phase": "on_recovery",
            "total_actors": len(actor_infos),
            "failures": failures,
            "passed": len(failures) == 0,
            "timestamp": time.time(),
        }

        logger.info(f"Recovery diagnostics complete: {'PASSED' if summary['passed'] else 'FAILED'}")

        return summary

    def collect_failure_diagnostics(
        self,
        actor_infos: List["ActorInfo"],
        dead_actor_ids: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Collect diagnostic information when a failure is detected.

        This should be called when a failure is detected (e.g., actor death,
        NCCL timeout) to gather information for root cause analysis.

        Collects:
        - Logs from surviving actors
        - GPU health snapshots
        - Stack traces for stack aggregation
        - Training metrics history

        Args:
            actor_infos: List of actor information
            dead_actor_ids: IDs of known dead actors

        Returns:
            Diagnostic report with root cause analysis
        """
        if not self.config.enabled:
            return {"enabled": False}

        import ray

        logger.info("Collecting failure diagnostics...")
        dead_actor_ids = dead_actor_ids or []

        # Collect from surviving actors
        alive_actors = [
            info for i, info in enumerate(actor_infos)
            if i not in dead_actor_ids
        ]

        # Run on_failure diagnostics
        results_refs = []
        for actor_info in alive_actors:
            try:
                ref = actor_info.handle.run_diagnostics.remote(
                    DiagnosticPhase.ON_FAILURE
                )
                results_refs.append((actor_info.rank, ref))
            except Exception:
                pass  # Actor might be dead

        # Collect stack traces for aggregation
        stack_refs = []
        for actor_info in alive_actors:
            try:
                ref = actor_info.handle.get_stack_trace.remote()
                stack_refs.append((actor_info.rank, ref))
            except Exception:
                pass

        # Gather results with timeout
        all_results = {}
        stacks = []
        root_causes = []
        suspects = set()

        for rank, ref in results_refs:
            try:
                results = ray.get(ref, timeout=30)
                all_results[str(rank)] = [r.to_dict() for r in results]

                for r in results:
                    if r.status == DiagnosticStatus.FAILED and r.root_cause:
                        root_causes.append(r.root_cause.value)
                        suspects.update(r.suspects)
            except Exception as e:
                logger.debug(f"Could not collect diagnostics from rank {rank}: {e}")

        for rank, ref in stack_refs:
            try:
                stack_str = ray.get(ref, timeout=10)
                stacks.append({
                    "rank": rank,
                    "stack": stack_str,
                })
            except Exception:
                pass

        # Add dead actors as suspects
        suspects.update(dead_actor_ids)

        # Determine primary root cause (most common)
        primary_root_cause = None
        if root_causes:
            from collections import Counter
            primary_root_cause = Counter(root_causes).most_common(1)[0][0]

        report = {
            "phase": "on_failure",
            "timestamp": time.time(),
            "total_actors": len(actor_infos),
            "alive_actors": len(alive_actors),
            "dead_actors": dead_actor_ids,
            "primary_root_cause": primary_root_cause,
            "all_root_causes": list(set(root_causes)),
            "suspects": list(suspects),
            "stack_count": len(stacks),
            "results_collected": len(all_results),
        }

        # Save to artifact store
        if self.store:
            trace_id = f"failure_{int(time.time())}"
            self.store.save_json_artifact(trace_id, "failure_report.json", report)
            if stacks:
                self.store.save_json_artifact(trace_id, "stack_traces.json", stacks)
            logger.info(f"Saved failure diagnostics to trace_id: {trace_id}")
            report["trace_id"] = trace_id

        return report

    def aggregate_training_health(
        self,
        actor_infos: List["ActorInfo"],
    ) -> Dict[str, Any]:
        """Aggregate training health from all actors.

        Gets training metrics history from all actors to check for
        implicit failures across the cluster.

        Args:
            actor_infos: List of actor information

        Returns:
            Aggregated health summary
        """
        if not self.config.enabled:
            return {"enabled": False}

        import ray

        # Get health summary from each actor
        health_refs = []
        for actor_info in actor_infos:
            try:
                ref = actor_info.handle.get_diagnostic_summary.remote()
                health_refs.append((actor_info.rank, ref))
            except Exception:
                pass

        health_data = {}
        for rank, ref in health_refs:
            try:
                summary = ray.get(ref, timeout=10)
                health_data[str(rank)] = summary
            except Exception:
                health_data[str(rank)] = {"error": "timeout"}

        return {
            "timestamp": time.time(),
            "actor_count": len(actor_infos),
            "health_data": health_data,
        }

    def should_run_periodic_sdc(self) -> bool:
        """Check if periodic SDC test should run.

        Based on config, checks if enough time has passed since
        the last SDC test.
        """
        if not self.config.enabled or not self.config.bitwise_alignment.enabled:
            return False

        # This would track last run time - simplified for now
        return False

    def get_over_eviction_candidates(
        self,
        suspects: List[int],
        actor_infos: List["ActorInfo"],
    ) -> List[int]:
        """Get the full list of actors to evict based on over-eviction config.

        When over-eviction is enabled, this expands the suspect list to
        include the entire parallel group (PP group typically).

        Args:
            suspects: Initial list of suspect actor IDs
            actor_infos: All actor information

        Returns:
            Expanded list of actors to evict
        """
        if not self.config.over_eviction.enabled:
            return suspects

        if not self.config.over_eviction.prefer_parallel_group:
            return suspects

        # Find the PP group for each suspect
        eviction_set = set(suspects)

        for suspect_id in suspects:
            if suspect_id >= len(actor_infos):
                continue

            suspect_rank = actor_infos[suspect_id].rank

            # Add all actors with same DP rank (same PP group)
            for i, actor_info in enumerate(actor_infos):
                if actor_info.rank.dp == suspect_rank.dp:
                    eviction_set.add(i)

        return list(eviction_set)
