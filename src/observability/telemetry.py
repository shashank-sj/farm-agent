"""
Per-interaction telemetry for the agent.

Always: one structured JSON log line per interaction (stdout → visible in the
HF Spaces log viewer, and trivially shippable to Loki/ELK later).

Optional: if MLFLOW_TRACKING_URI is set, the same numbers are logged to one
long-lived MLflow run per process, as step-indexed metrics. (One MLflow run
per chat message — what the old mlflow/tracking.py did — creates thousands of
runs and makes the UI useless; a run is meant to be an experiment, not a request.)

Privacy: the raw question is never logged — only its length and a short hash,
which is enough to spot repeated queries without storing what farmers typed.
"""

import hashlib
import json
import logging
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("farm-telemetry")


@dataclass
class InteractionRecord:
    query_chars: int
    query_hash: str
    has_image: bool
    tools_used: list[str] = field(default_factory=list)
    steps: int = 0
    latency_s: float = 0.0
    guardrail: str = "ok"
    fallback_used: bool = False
    error_type: Optional[str] = None
    timestamp: str = ""

    @classmethod
    def new(cls, query: str, has_image: bool) -> "InteractionRecord":
        return cls(
            query_chars=len(query or ""),
            query_hash=hashlib.sha256((query or "").encode("utf-8")).hexdigest()[:12],
            has_image=has_image,
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )


class _MlflowSink:
    def __init__(self, tracking_uri: str, experiment: str):
        import mlflow

        self.mlflow = mlflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
        self.run = mlflow.start_run(run_name=f"serving-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}")
        self.step = 0
        self.lock = threading.Lock()

    def log(self, rec: InteractionRecord) -> None:
        with self.lock:
            metrics = {
                "latency_s": rec.latency_s,
                "steps": rec.steps,
                "fallback_used": float(rec.fallback_used),
                "error": float(rec.error_type is not None),
                "guardrail_blocked": float(rec.guardrail in ("crisis", "poisoning")),
            }
            for tool in ("farm_rag", "farm_vision", "farm_web_search", "yield_prediction"):
                metrics[f"tool_{tool}"] = float(tool in rec.tools_used)
            self.mlflow.log_metrics(metrics, step=self.step)
            self.step += 1


_sink: Optional[_MlflowSink] = None
_sink_init_done = False


def _get_sink() -> Optional[_MlflowSink]:
    global _sink, _sink_init_done
    if not _sink_init_done:
        _sink_init_done = True
        uri = os.getenv("MLFLOW_TRACKING_URI", "")
        if uri:
            try:
                _sink = _MlflowSink(uri, os.getenv("MLFLOW_EXPERIMENT", "farm-agent-serving"))
            except Exception as e:
                logger.warning(f"MLflow disabled (could not initialise: {e})")
    return _sink


def log_interaction(rec: InteractionRecord) -> None:
    """Never raises — telemetry must not break a farmer's answer."""
    try:
        logger.info(json.dumps({"event": "interaction", **asdict(rec)}))
        sink = _get_sink()
        if sink:
            sink.log(rec)
    except Exception as e:
        logger.warning(f"Telemetry failed (non-critical): {e}")
