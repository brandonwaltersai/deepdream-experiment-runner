# DeepDream Experiment Runner

![tests](https://github.com/brandonwaltersai/deepdream-experiment-runner/actions/workflows/tests.yml/badge.svg)

A batch DeepDream pipeline on a pretrained InceptionV3 (ImageNet) —
multi-octave, tiled gradients for memory safety, and a runner that never
lets one failed configuration kill the whole batch.

## What DeepDream actually is (beyond "trippy art")

Gradient ascent on a trained CNN's *internal* activations instead of its
output layer — literally "what pattern would make this layer fire
harder." It's a real, if unusual, model-interpretability tool: which
patterns different layers respond to. Early layers amplify edges and
textures; deeper layers amplify object-like motifs (the model was trained
on ImageNet, so you'll see dog/eye-like shapes emerge from deep layers on
almost any input — that's a real signal about what the network learned,
not an artistic choice).

## What makes this "production-grade" rather than a tutorial notebook

- **Config-driven experiments**, not "edit random cells" — each run is an
  `Experiment` dataclass (layers, octaves, step size, tile size), validated
  before it runs
- **Tiled gradients**: large images are processed in tiles so memory usage
  stays bounded regardless of input size, with random jitter between tiles
  to avoid visible seams
- **Batch runner that always finishes**: each (image, experiment) pair is
  wrapped in try/except — one out-of-memory or bad-config failure is
  logged and the batch continues, rather than dying partway through
- **Resume mode**: skips outputs that already exist on disk, so a killed
  batch run can restart without redoing completed work

## Verified, not just written

This was actually executed in this environment: real InceptionV3 weights
downloaded and loaded, a real multi-octave DeepDream pass run, and a real
output image produced showing the characteristic InceptionV3 activation
patterns. Full run log and the actual output image:
[`docs/results.md`](docs/results.md).

## Running it

```bash
pip install -r requirements.txt
python -m pytest tests/ -v              # fast tests + 2 real-model tests (~10s total)
```

```python
from src.deepdream import load_base_model
from src.experiments import DEFAULT_EXPERIMENTS, run_batch
import numpy as np
from PIL import Image

base_model = load_base_model()
img = np.array(Image.open("your_image.png").convert("RGB"))
records = run_batch({"my_image": img}, DEFAULT_EXPERIMENTS, base_model, out_dir="./outputs")
```

## Project structure

```
src/
  deepdream.py     the DeepDream engine: simple + multi-octave/tiled variants
  experiments.py   Experiment config, validation, and the always-finish batch runner
tests/
  test_experiments.py   7 tests: 5 fast validation tests + 2 that load real
                         InceptionV3 weights and run genuine DeepDream passes
docs/
  results.md            the actual verified run log + output image
data/
  demo_input.png         synthetic test input used for the verified run
```

## Stack

Python · TensorFlow/Keras · InceptionV3 (ImageNet pretrained)

## Author

Brandon Walters — [LinkedIn](https://linkedin.com/in/brandon-walters-172b29208)
