"""Experiment plan validation and a batch runner that never dies on a single failure.

Each run is wrapped in try/except; a failure is logged with its parameters
and the batch continues, rather than one bad configuration killing the job.
Resume mode skips outputs that already exist on disk.
"""
from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .deepdream import build_dream_model, run_deep_dream_with_octaves


@dataclass
class Experiment:
    run_tag: str
    layer_names: list[str]
    octaves: list[int] = field(default_factory=lambda: list(range(-2, 3)))
    octave_scale: float = 1.3
    steps_per_octave: int = 80
    step_size: float = 0.01
    tile_size: int = 512

    def validate(self, available_layers: set[str]):
        missing = [n for n in self.layer_names if n not in available_layers]
        if missing:
            raise ValueError(f"Invalid layer name(s): {missing}")
        if self.steps_per_octave <= 0:
            raise ValueError("steps_per_octave must be > 0")
        if self.step_size <= 0:
            raise ValueError("step_size must be > 0")
        if self.octave_scale <= 1.0:
            raise ValueError("octave_scale should be > 1.0")
        if not self.octaves:
            raise ValueError("octaves cannot be empty")


DEFAULT_EXPERIMENTS = [
    Experiment(run_tag="baseline_mid", layer_names=["mixed3", "mixed5"]),
    Experiment(run_tag="early_layers", layer_names=["mixed1", "mixed2"]),
    Experiment(run_tag="deep_layers", layer_names=["mixed7", "mixed8"]),
    Experiment(run_tag="more_octaves_stronger", layer_names=["mixed3", "mixed5"],
               octaves=list(range(-3, 4)), octave_scale=1.35, steps_per_octave=70, step_size=0.012),
]


def run_batch(images: dict[str, np.ndarray], experiments: list[Experiment], base_model,
              out_dir: str, resume: bool = True) -> list[dict]:
    """Run every (image, experiment) pair. Returns a list of run-log records; never raises."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    available_layers = {l.name for l in base_model.layers}

    records = []
    for img_key, img_arr in images.items():
        for exp in experiments:
            record = {"image": img_key, "run_tag": exp.run_tag, "params": exp.__dict__.copy(),
                      "status": None, "error": None, "output_path": None, "seconds": None}
            out_file = out_path / f"{img_key}__{exp.run_tag}.png"

            if resume and out_file.exists():
                record["status"] = "SKIPPED_EXISTS"
                record["output_path"] = str(out_file)
                records.append(record)
                continue

            t0 = time.time()
            try:
                exp.validate(available_layers)
                dream_model = build_dream_model(base_model, exp.layer_names)
                result = run_deep_dream_with_octaves(
                    img_arr, dream_model, steps_per_octave=exp.steps_per_octave,
                    step_size=exp.step_size, octaves=exp.octaves,
                    octave_scale=exp.octave_scale, tile_size=exp.tile_size,
                )
                Image.fromarray(result).save(out_file)
                record["status"] = "OK"
                record["output_path"] = str(out_file)
            except Exception as e:
                record["status"] = "FAIL"
                record["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=5)}"
            record["seconds"] = round(time.time() - t0, 2)
            records.append(record)

    log_path = out_path / "run_log.json"
    log_path.write_text(json.dumps(records, indent=2, default=str))
    return records
