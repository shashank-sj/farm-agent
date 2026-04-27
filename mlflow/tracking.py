"""
MLflow Tracking for Farm Agent
Tracks: queries, tool usage, response times, model versions
Run MLflow UI: mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db
"""

import time
import json
import logging
import functools
from pathlib import Path
from datetime import datetime

import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger("farm-mlflow")

TRACKING_URI = "sqlite:///mlflow/mlflow.db"
EXPERIMENT  = "farm-agent-production"


def setup():
    Path("mlflow").mkdir(exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)


def log_agent_call(query: str, response: str, tool_used: str,
                   latency_s: float, has_image: bool = False):
    """Log a single agent interaction to MLflow."""
    setup()
    with mlflow.start_run(run_name=f"query_{int(time.time())}"):
        mlflow.log_params({
            "query_length": len(query),
            "has_image": has_image,
            "tool_used": tool_used or "none",
            "timestamp": datetime.utcnow().isoformat(),
        })
        mlflow.log_metrics({
            "latency_seconds": latency_s,
            "response_length": len(response),
        })


def track(func):
    """Decorator to auto-track agent.chat() calls in MLflow."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = func(*args, **kwargs)
            latency = time.time() - start
            try:
                query = args[1] if len(args) > 1 else kwargs.get("message", "")
                has_image = bool(kwargs.get("image_path") or (len(args) > 2 and args[2]))
                log_agent_call(
                    query=str(query),
                    response=str(result),
                    tool_used="auto",
                    latency_s=round(latency, 3),
                    has_image=has_image,
                )
            except Exception as e:
                logger.warning(f"MLflow logging failed (non-critical): {e}")
            return result
        except Exception:
            latency = time.time() - start
            logger.error(f"Agent call failed after {latency:.2f}s")
            raise
    return wrapper
