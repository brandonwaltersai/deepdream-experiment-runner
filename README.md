# DeepDream Experiment Runner

![tests](https://github.com/brandonwaltersai/deepdream-experiment-runner/actions/workflows/tests.yml/badge.svg)

A resilient batch DeepDream pipeline built on pretrained InceptionV3: config-driven experiments, multi-octave optimization, tiled gradients for bounded memory use, failure isolation, resume behavior, and reproducible tests.

## Why this project exists

DeepDream is gradient ascent on a trained CNN's internal activations. Instead of optimizing model weights, it optimizes the input image to increase selected layer responses. That makes it useful as a compact experiment in model introspection, gradient-based image generation, and controlled parameter testing.

The repository grew out of a 12-run graduate experiment across three image types and four controlled parameter conditions. The original experiment measured output changes with Mean Absolute Difference and Sobel-based edge density in addition to visual inspection. See [`docs/original-experiment-summary.md`](docs/original-experiment-summary.md).

## Engineering features

- **Config-driven experiments** — each run is represented as a validated `Experiment` configuration rather than an edited notebook cell.
- **Tiled gradients** — larger images are processed in tiles to bound memory use, with randomized shifts to reduce visible seams.
- **Failure isolation** — one bad configuration or runtime failure is logged without terminating the remaining batch.
- **Resume mode** — existing outputs are skipped so interrupted batches can continue without repeating completed runs.
- **Real-model tests** — the test suite includes checks that load genuine InceptionV3 weights and execute real DeepDream passes.
- **Documented provenance** — the controlled academic experiment is separated from the refactored reusable runner so results are not overstated.

## What the original experiment found

Across the controlled runs:

- early-layer targets generally amplified edges and textures;
- deeper layers produced more object-like composite patterns;
- the higher-intensity multi-octave condition produced the largest pixel transformation across all three source-image categories;
- the same parameter set behaved differently on object-centric, architectural, and natural-texture inputs.

Full metrics and experiment settings are documented in [`docs/original-experiment-summary.md`](docs/original-experiment-summary.md).

## Verified execution

The refactored runner has also been executed independently with real InceptionV3 weights and a real multi-octave DeepDream pass. See [`docs/results.md`](docs/results.md) for the run log and output evidence.

## Running it

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```

```python
from src.deepdream import load_base_model
from src.experiments import DEFAULT_EXPERIMENTS, run_batch
import numpy as np
from PIL import Image

base_model = load_base_model()
img = np.array(Image.open("your_image.png").convert("RGB"))
records = run_batch(
    {"my_image": img},
    DEFAULT_EXPERIMENTS,
    base_model,
    out_dir="./outputs",
)
```

## Project structure

```text
src/
  deepdream.py          DeepDream engine: simple and multi-octave/tiled variants
  experiments.py        experiment config, validation, batch runner, resume logic
tests/
  test_experiments.py   validation tests plus real-model execution tests
docs/
  results.md                    verified refactored-run evidence
  original-experiment-summary.md 12-run controlled experiment and metrics
data/
  demo_input.png         synthetic input used for verification
```

## Stack

Python · TensorFlow/Keras · InceptionV3 · NumPy · Pillow

## Scope

This project is not presented as state-of-the-art generative vision research. Its purpose is to demonstrate controlled experimentation, CNN feature behavior, gradient-based optimization, reproducibility, and reliable execution patterns.

## Author

Brandon Walters — [LinkedIn](https://www.linkedin.com/in/bw172b29208/)
