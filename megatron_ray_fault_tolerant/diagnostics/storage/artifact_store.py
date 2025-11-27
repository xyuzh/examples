"""Artifact Store - manages diagnostic artifact storage.

Reuses the file_io module for cloud storage (S3/GCS) support.
"""

import os
import json
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from loguru import logger

# Import from parent package's file_io module
import file_io as io
from ..core.result import DiagnosticResult


@dataclass
class DiagnosticReport:
    """A complete diagnostic report to be stored."""
    trace_id: str
    timestamp: float
    rank: int
    world_size: int
    step: Optional[int]
    phase: str
    results: List[Dict[str, Any]]
    summary: Dict[str, Any]


class ArtifactStore:
    """Manages storage of diagnostic artifacts.

    Supports both local filesystem and cloud storage (S3/GCS) through
    the file_io module. Stores:
    - Diagnostic reports (JSON)
    - Tensor snapshots
    - Log captures
    - Stack traces

    Directory structure:
        artifact_dir/
        ├── reports/
        │   └── {trace_id}/
        │       ├── report.json
        │       ├── summary.json
        │       └── artifacts/
        │           ├── log_capture.txt
        │           └── stack_traces.json
        └── snapshots/
            └── step_{N}/
                └── {tensor_name}.pt
    """

    def __init__(
        self,
        local_dir: str = "/tmp/diagnostics",
        cloud_dir: Optional[str] = None,
    ):
        self.local_dir = local_dir
        self.cloud_dir = cloud_dir
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        """Ensure local directories exist."""
        os.makedirs(os.path.join(self.local_dir, "reports"), exist_ok=True)
        os.makedirs(os.path.join(self.local_dir, "snapshots"), exist_ok=True)

    def save_report(self, report: DiagnosticReport) -> str:
        """Save a diagnostic report.

        Args:
            report: The diagnostic report to save

        Returns:
            Path to the saved report
        """
        report_dir = os.path.join(
            self.local_dir, "reports", report.trace_id
        )
        os.makedirs(report_dir, exist_ok=True)

        # Save full report
        report_path = os.path.join(report_dir, "report.json")
        with open(report_path, "w") as f:
            json.dump(asdict(report), f, indent=2, default=str)

        # Save summary separately for quick access
        summary_path = os.path.join(report_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(report.summary, f, indent=2, default=str)

        # Upload to cloud if configured
        if self.cloud_dir:
            try:
                cloud_report_dir = f"{self.cloud_dir}/reports/{report.trace_id}"
                io.upload_directory(report_dir, cloud_report_dir)
                logger.info(f"Uploaded report to {cloud_report_dir}")
            except Exception as e:
                logger.error(f"Failed to upload report to cloud: {e}")

        return report_path

    def save_artifact(
        self,
        trace_id: str,
        name: str,
        content: str,
    ) -> str:
        """Save a text artifact (logs, traces, etc).

        Args:
            trace_id: The diagnostic trace ID
            name: Artifact name (e.g., "log_capture.txt")
            content: Text content to save

        Returns:
            Path to saved artifact
        """
        artifact_dir = os.path.join(
            self.local_dir, "reports", trace_id, "artifacts"
        )
        os.makedirs(artifact_dir, exist_ok=True)

        artifact_path = os.path.join(artifact_dir, name)
        with open(artifact_path, "w") as f:
            f.write(content)

        return artifact_path

    def save_json_artifact(
        self,
        trace_id: str,
        name: str,
        data: Any,
    ) -> str:
        """Save a JSON artifact.

        Args:
            trace_id: The diagnostic trace ID
            name: Artifact name (e.g., "stack_traces.json")
            data: Data to serialize as JSON

        Returns:
            Path to saved artifact
        """
        artifact_dir = os.path.join(
            self.local_dir, "reports", trace_id, "artifacts"
        )
        os.makedirs(artifact_dir, exist_ok=True)

        artifact_path = os.path.join(artifact_dir, name)
        with open(artifact_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        return artifact_path

    def save_tensor_snapshot(
        self,
        step: int,
        name: str,
        tensor,
    ) -> str:
        """Save a tensor snapshot.

        Args:
            step: Training step number
            name: Tensor name
            tensor: PyTorch tensor to save

        Returns:
            Path to saved tensor
        """
        try:
            import torch
        except ImportError:
            logger.warning("PyTorch not available, cannot save tensor")
            return ""

        snapshot_dir = os.path.join(
            self.local_dir, "snapshots", f"step_{step}"
        )
        os.makedirs(snapshot_dir, exist_ok=True)

        tensor_path = os.path.join(snapshot_dir, f"{name}.pt")
        torch.save(tensor.detach().cpu(), tensor_path)

        return tensor_path

    def load_report(self, trace_id: str) -> Optional[DiagnosticReport]:
        """Load a diagnostic report.

        Args:
            trace_id: The diagnostic trace ID

        Returns:
            The loaded report, or None if not found
        """
        report_path = os.path.join(
            self.local_dir, "reports", trace_id, "report.json"
        )

        if not os.path.exists(report_path):
            # Try cloud
            if self.cloud_dir:
                cloud_report_dir = f"{self.cloud_dir}/reports/{trace_id}"
                try:
                    io.download_directory(
                        cloud_report_dir,
                        os.path.join(self.local_dir, "reports", trace_id)
                    )
                except Exception as e:
                    logger.error(f"Failed to download report from cloud: {e}")
                    return None

        if not os.path.exists(report_path):
            return None

        with open(report_path, "r") as f:
            data = json.load(f)

        return DiagnosticReport(**data)

    def list_reports(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List recent diagnostic reports.

        Args:
            limit: Maximum number of reports to return

        Returns:
            List of report summaries
        """
        reports_dir = os.path.join(self.local_dir, "reports")
        if not os.path.exists(reports_dir):
            return []

        reports = []
        for trace_id in os.listdir(reports_dir):
            summary_path = os.path.join(reports_dir, trace_id, "summary.json")
            if os.path.exists(summary_path):
                with open(summary_path, "r") as f:
                    summary = json.load(f)
                    summary["trace_id"] = trace_id
                    reports.append(summary)

        # Sort by timestamp (newest first)
        reports.sort(key=lambda r: r.get("timestamp", 0), reverse=True)

        return reports[:limit]

    def cleanup_old_reports(self, max_age_hours: float = 24.0) -> int:
        """Remove reports older than max_age_hours.

        Args:
            max_age_hours: Maximum age in hours

        Returns:
            Number of reports removed
        """
        import shutil

        reports_dir = os.path.join(self.local_dir, "reports")
        if not os.path.exists(reports_dir):
            return 0

        cutoff_time = time.time() - (max_age_hours * 3600)
        removed = 0

        for trace_id in os.listdir(reports_dir):
            report_dir = os.path.join(reports_dir, trace_id)
            report_path = os.path.join(report_dir, "report.json")

            if os.path.exists(report_path):
                with open(report_path, "r") as f:
                    try:
                        data = json.load(f)
                        if data.get("timestamp", 0) < cutoff_time:
                            shutil.rmtree(report_dir)
                            removed += 1
                    except Exception:
                        pass

        return removed

    def get_artifact_path(self, trace_id: str, name: str) -> str:
        """Get the path to an artifact file.

        Args:
            trace_id: The diagnostic trace ID
            name: Artifact name

        Returns:
            Full path to the artifact
        """
        return os.path.join(
            self.local_dir, "reports", trace_id, "artifacts", name
        )

    def upload_to_cloud(self, trace_id: str) -> bool:
        """Upload a specific report to cloud storage.

        Args:
            trace_id: The diagnostic trace ID

        Returns:
            True if successful, False otherwise
        """
        if not self.cloud_dir:
            return False

        local_report_dir = os.path.join(self.local_dir, "reports", trace_id)
        if not os.path.exists(local_report_dir):
            return False

        try:
            cloud_report_dir = f"{self.cloud_dir}/reports/{trace_id}"
            io.upload_directory(local_report_dir, cloud_report_dir)
            logger.info(f"Uploaded report {trace_id} to {cloud_report_dir}")
            return True
        except Exception as e:
            logger.error(f"Failed to upload report to cloud: {e}")
            return False
