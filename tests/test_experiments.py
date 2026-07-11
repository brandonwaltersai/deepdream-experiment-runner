import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from src.experiments import DEFAULT_EXPERIMENTS, Experiment

AVAILABLE_LAYERS = {f"mixed{i}" for i in range(11)}


def test_default_experiments_all_validate():
    for exp in DEFAULT_EXPERIMENTS:
        exp.validate(AVAILABLE_LAYERS)  # should not raise


def test_experiment_rejects_unknown_layer():
    exp = Experiment(run_tag="bad", layer_names=["not_a_real_layer"])
    with pytest.raises(ValueError, match="Invalid layer name"):
        exp.validate(AVAILABLE_LAYERS)


def test_experiment_rejects_zero_steps():
    exp = Experiment(run_tag="bad", layer_names=["mixed3"], steps_per_octave=0)
    with pytest.raises(ValueError, match="steps_per_octave"):
        exp.validate(AVAILABLE_LAYERS)


def test_experiment_rejects_octave_scale_at_or_below_one():
    exp = Experiment(run_tag="bad", layer_names=["mixed3"], octave_scale=1.0)
    with pytest.raises(ValueError, match="octave_scale"):
        exp.validate(AVAILABLE_LAYERS)


def test_experiment_rejects_empty_octaves():
    exp = Experiment(run_tag="bad", layer_names=["mixed3"], octaves=[])
    with pytest.raises(ValueError, match="octaves cannot be empty"):
        exp.validate(AVAILABLE_LAYERS)


@pytest.mark.slow
def test_run_batch_end_to_end_real_model(tmp_path):
    """Loads real InceptionV3 weights and runs one small real DeepDream experiment."""
    import numpy as np
    from src.deepdream import load_base_model
    from src.experiments import run_batch

    base_model = load_base_model()
    img = (np.random.default_rng(1).integers(0, 256, (64, 64, 3))).astype("uint8")
    exp = Experiment(run_tag="smoke", layer_names=["mixed3"], octaves=[0], steps_per_octave=3, tile_size=64)

    records = run_batch({"test": img}, [exp], base_model, out_dir=str(tmp_path), resume=False)

    assert records[0]["status"] == "OK"
    assert Path(records[0]["output_path"]).exists()


@pytest.mark.slow
def test_run_batch_continues_after_one_failure(tmp_path):
    """One bad experiment shouldn't stop the batch -- the core 'always finish' guarantee."""
    import numpy as np
    from src.deepdream import load_base_model
    from src.experiments import run_batch

    base_model = load_base_model()
    img = (np.random.default_rng(2).integers(0, 256, (64, 64, 3))).astype("uint8")
    bad = Experiment(run_tag="bad", layer_names=["mixed3"], octaves=[0], steps_per_octave=0, tile_size=64)
    good = Experiment(run_tag="good", layer_names=["mixed3"], octaves=[0], steps_per_octave=3, tile_size=64)

    records = run_batch({"test": img}, [bad, good], base_model, out_dir=str(tmp_path), resume=False)

    statuses = {r["run_tag"]: r["status"] for r in records}
    assert statuses["bad"] == "FAIL"
    assert statuses["good"] == "OK"
