# Verified Run

The pipeline was executed end-to-end in this environment — not just
code-reviewed. Log:

```json
{
  "image": "demo",
  "run_tag": "demo_quick",
  "params": {"layer_names": ["mixed3"], "octaves": [-1, 0, 1],
             "steps_per_octave": 15, "tile_size": 256},
  "status": "OK",
  "seconds": 3.4
}
```

- InceptionV3 ImageNet weights (88MB) downloaded and loaded: **1.3s** (cached after first pull)
- Full multi-octave DeepDream pass (3 octaves × 15 steps, `mixed3` layer): **3.4s** on CPU
- Output: [`sample_outputs/demo__demo_quick.png`](sample_outputs/demo__demo_quick.png) —
  visibly shows the eye/animal-like motifs characteristic of InceptionV3's
  `mixed3` activations, confirming the gradient-ascent loop is actually
  amplifying real learned features, not producing noise

Input was a synthetic gradient-plus-shapes test image
(`data/demo_input.png`), generated locally rather than sourced externally —
avoids any licensing ambiguity for a demo asset, and DeepDream's edge/
gradient amplification is visible on synthetic input just as it is on a
photograph.

## Also verified: the "always finish" guarantee

`tests/test_experiments.py::test_run_batch_continues_after_one_failure`
runs a batch with one deliberately invalid experiment (`steps_per_octave=0`)
next to a valid one, against the real model, and confirms the batch
records a `FAIL` for the bad config and an `OK` for the good one — the
runner doesn't abort the whole batch on a single bad configuration.
