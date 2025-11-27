"""DiagnosticContext - runtime context for diagnostic checks."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, TYPE_CHECKING
import time
import uuid

if TYPE_CHECKING:
    from dispatch import MeshRank


@dataclass
class DiagnosticContext:
    """Runtime context passed to diagnostic checks.

    Contains all information a diagnostic might need about the current
    training state, rank information, and where to store artifacts.
    """
    # Correlation ID for this diagnostic run
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    # Rank information
    rank: int = 0
    world_size: int = 1
    mesh_rank: Optional["MeshRank"] = None

    # Training state
    step: Optional[int] = None
    epoch: Optional[int] = None

    # Artifact storage
    artifact_dir: str = "/tmp/diagnostics"
    cloud_artifact_dir: Optional[str] = None

    # Timing
    timestamp: float = field(default_factory=time.time)

    # Additional context (for extensibility)
    extra: Dict[str, Any] = field(default_factory=dict)

    # Reference to the actor (for collecting distributed info)
    actor_ref: Optional[Any] = None

    @property
    def is_rank_zero(self) -> bool:
        """Check if this is the global rank 0."""
        return self.rank == 0

    @property
    def is_dp_rank_zero(self) -> bool:
        """Check if this is DP rank 0 (for data-parallel operations)."""
        if self.mesh_rank is None:
            return self.rank == 0
        return self.mesh_rank.dp == 0

    @property
    def artifact_path(self) -> str:
        """Get the artifact path including trace_id for this run."""
        import os
        return os.path.join(self.artifact_dir, f"diag_{self.trace_id}")

    def with_step(self, step: int) -> "DiagnosticContext":
        """Return a new context with updated step."""
        return DiagnosticContext(
            trace_id=self.trace_id,
            rank=self.rank,
            world_size=self.world_size,
            mesh_rank=self.mesh_rank,
            step=step,
            epoch=self.epoch,
            artifact_dir=self.artifact_dir,
            cloud_artifact_dir=self.cloud_artifact_dir,
            timestamp=time.time(),
            extra=self.extra,
            actor_ref=self.actor_ref,
        )

    def with_extra(self, **kwargs) -> "DiagnosticContext":
        """Return a new context with additional extra fields."""
        new_extra = {**self.extra, **kwargs}
        return DiagnosticContext(
            trace_id=self.trace_id,
            rank=self.rank,
            world_size=self.world_size,
            mesh_rank=self.mesh_rank,
            step=self.step,
            epoch=self.epoch,
            artifact_dir=self.artifact_dir,
            cloud_artifact_dir=self.cloud_artifact_dir,
            timestamp=time.time(),
            extra=new_extra,
            actor_ref=self.actor_ref,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        mesh_rank_dict = None
        if self.mesh_rank is not None:
            mesh_rank_dict = {
                "dp": self.mesh_rank.dp,
                "sp": self.mesh_rank.sp,
                "tp": self.mesh_rank.tp,
                "pp": self.mesh_rank.pp,
                "world_size": self.mesh_rank.world_size,
                "dp_size": self.mesh_rank.dp_size,
                "pp_size": self.mesh_rank.pp_size,
            }
        return {
            "trace_id": self.trace_id,
            "rank": self.rank,
            "world_size": self.world_size,
            "mesh_rank": mesh_rank_dict,
            "step": self.step,
            "epoch": self.epoch,
            "artifact_dir": self.artifact_dir,
            "cloud_artifact_dir": self.cloud_artifact_dir,
            "timestamp": self.timestamp,
            "extra": self.extra,
        }
