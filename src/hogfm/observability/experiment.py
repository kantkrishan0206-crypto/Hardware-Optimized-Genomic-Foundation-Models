from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


class ExperimentTracker:
    """Dependency-light experiment tracker with optional TensorBoard and W&B hooks."""

    def __init__(self, run_dir: str | Path, run_name: str = "hogfm-run") -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.events_path = self.run_dir / "events.jsonl"
        self.tensorboard_writer = self._build_tensorboard_writer()
        self.wandb_run = self._build_wandb_run()
        self.mlflow = self._build_mlflow()

    def _build_tensorboard_writer(self) -> Any | None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            return None
        return SummaryWriter(log_dir=str(self.run_dir / "tensorboard"))

    def _build_wandb_run(self) -> Any | None:
        if os.getenv("HOGFM_ENABLE_WANDB") != "1":
            return None
        try:
            import wandb
        except ImportError:
            return None
        return wandb.init(project="hogfm", name=self.run_name, dir=str(self.run_dir), reinit=True)

    def _build_mlflow(self) -> Any | None:
        if os.getenv("HOGFM_ENABLE_MLFLOW") != "1":
            return None
        try:
            import mlflow
        except ImportError:
            return None
        mlflow.set_experiment("hogfm")
        mlflow.start_run(run_name=self.run_name)
        return mlflow

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        event = {
            "time": time.time(),
            "run_name": self.run_name,
            "step": step,
            "metrics": metrics,
        }
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
        if self.tensorboard_writer is not None:
            for key, value in metrics.items():
                self.tensorboard_writer.add_scalar(key, value, step)
        if self.wandb_run is not None:
            self.wandb_run.log(metrics, step=step)
        if self.mlflow is not None:
            self.mlflow.log_metrics(metrics, step=step)

    def close(self) -> None:
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.close()
        if self.wandb_run is not None:
            self.wandb_run.finish()
        if self.mlflow is not None:
            self.mlflow.end_run()
