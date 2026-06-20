import atexit
import csv
import os
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

from vidur.config import SimulationConfig


class _QueueDiagnosticsWriter:
    HEADER = [
        "event_level",
        "time",
        "event_index",
        "scheduler",
        "model",
        "replicas",
        "trace",
        "scale",
        "run_id",
        "replica_id",
        "queue_depth_before",
        "queue_depth_after",
        "num_assigned_this_call",
        "waiting_depth_before",
        "token_budget_before",
        "head_num_new_tokens",
        "head_is_feasible",
        "break_reason",
        "feasible_count_top2",
        "feasible_count_top4",
        "feasible_count_top8",
        "best_feasible_num_new_tokens_top2",
        "best_feasible_num_new_tokens_top4",
        "best_feasible_num_new_tokens_top8",
    ]

    def __init__(self, output_dir: str, metadata: Dict[str, Any]):
        self._lock = Lock()
        self._metadata = metadata
        self._counter = 0

        metrics_dir = Path(output_dir) / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        self._path = metrics_dir / "queue_diag_events.csv"

        self._file = self._path.open("w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.HEADER)
        self._writer.writeheader()
        self._file.flush()

    @property
    def path(self) -> str:
        return str(self._path)

    def log(self, event_level: str, **kwargs: Any) -> None:
        with self._lock:
            self._counter += 1
            row = {k: "" for k in self.HEADER}
            row.update(self._metadata)
            row.update(kwargs)
            row["event_level"] = event_level
            row["event_index"] = self._counter
            self._writer.writerow(row)
            self._file.flush()

    def close(self) -> None:
        with self._lock:
            try:
                self._file.flush()
                self._file.close()
            except Exception:
                pass


_QUEUE_DIAG_WRITER: Optional[_QueueDiagnosticsWriter] = None


def _normalize_model(model_name: str) -> str:
    m = model_name.lower()
    if "70b" in m:
        return "70b"
    if "7b" in m:
        return "7b"
    return model_name


def _normalize_trace(trace_file: str) -> str:
    name = Path(trace_file).stem.lower()
    if "tool" in name:
        return "Tool"
    if "conv" in name or "conversation" in name:
        return "Conv"
    if "synth" in name or "synthetic" in name:
        return "Synth"
    return Path(trace_file).stem


def _normalize_scheduler(global_scheduler_config: Any) -> str:
    if hasattr(global_scheduler_config, "name"):
        return str(global_scheduler_config.name).upper()
    if hasattr(global_scheduler_config, "get_type"):
        try:
            t = global_scheduler_config.get_type()
            if hasattr(t, "name"):
                return str(t.name).upper()
            return str(t).upper()
        except Exception:
            pass
    return global_scheduler_config.__class__.__name__.replace("GlobalSchedulerConfig", "").upper()


def configure_queue_diagnostics(config: SimulationConfig) -> None:
    global _QUEUE_DIAG_WRITER

    enabled = os.getenv("ENABLE_QUEUE_DIAGNOSTICS", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not enabled:
        return

    if _QUEUE_DIAG_WRITER is not None:
        return

    metadata = {
        "scheduler": _normalize_scheduler(config.cluster_config.global_scheduler_config),
        "model": _normalize_model(config.cluster_config.replica_config.model_name),
        "replicas": f"{config.cluster_config.num_replicas}R",
        "trace": _normalize_trace(config.request_generator_config.trace_file)
        if hasattr(config.request_generator_config, "trace_file")
        else "synthetic",
        "scale": getattr(config.request_generator_config, "time_scale_factor", ""),
        "run_id": Path(config.metrics_config.output_dir).name,
    }
    _QUEUE_DIAG_WRITER = _QueueDiagnosticsWriter(config.metrics_config.output_dir, metadata)
    atexit.register(close_queue_diagnostics)


def get_queue_diagnostics_writer() -> Optional[_QueueDiagnosticsWriter]:
    return _QUEUE_DIAG_WRITER


def close_queue_diagnostics() -> None:
    global _QUEUE_DIAG_WRITER
    if _QUEUE_DIAG_WRITER is None:
        return
    _QUEUE_DIAG_WRITER.close()
    _QUEUE_DIAG_WRITER = None
