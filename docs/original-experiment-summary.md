# Original Controlled Experiment Summary

This repository grew out of a controlled DeepDream experiment completed as part of graduate AI coursework. The original experiment used TensorFlow/Keras with pretrained InceptionV3 and evaluated how layer choice, octave configuration, and optimization intensity affected generated outputs across three different image types.

## Experimental design

Three input-image categories were used:

- object-centric image (cat)
- urban/architectural scene
- forest/natural texture scene

Four conditions were applied to each image for 12 total experiment runs:

| Condition | Target layers | Octaves | Steps/octave | Step size | Tile size |
|---|---|---|---:|---:|---:|
| baseline_mid | `mixed3`, `mixed5` | `[-2,-1,0,1,2]` | 100 | 0.01 | 512 |
| deep_layers | `mixed7`, `mixed8` | `[-2,-1,0,1,2]` | 100 | 0.01 | 512 |
| early_layers | `mixed1`, `mixed2` | `[-2,-1,0,1,2]` | 100 | 0.01 | 512 |
| octaves_plus_intensity | `mixed3`, `mixed5` | `[-1,0,1]` | 150 | 0.02 | 512 |

The run environment recorded TensorFlow/Python metadata and GPU availability. The original experiment executed on CPU.

## Evaluation

Two lightweight quantitative measures complemented visual inspection:

- **Mean Absolute Difference (MAD):** average per-pixel absolute difference between the generated output and original input after resizing to a common shape.
- **Edge density:** proportion of pixels above a normalized Sobel-gradient threshold, used as a rough measure of high-frequency texture/structure.

## Results

| Image | Condition | MAD | Edge density (output) | Edge density (original) | Edge delta |
|---|---|---:|---:|---:|---:|
| cat | baseline_mid | 0.1840 | 0.1194 | 0.0257 | 0.0938 |
| cat | deep_layers | 0.1525 | 0.1152 | 0.0257 | 0.0896 |
| cat | early_layers | 0.1900 | 0.1364 | 0.0257 | 0.1107 |
| cat | octaves_plus_intensity | 0.2465 | 0.1993 | 0.0257 | 0.1736 |
| city | baseline_mid | 0.3053 | 0.1330 | 0.1379 | -0.0049 |
| city | deep_layers | 0.2977 | 0.2081 | 0.1379 | 0.0703 |
| city | early_layers | 0.3073 | 0.1603 | 0.1379 | 0.0224 |
| city | octaves_plus_intensity | 0.3176 | 0.2242 | 0.1379 | 0.0863 |
| forest | baseline_mid | 0.1232 | 0.2212 | 0.1969 | 0.0243 |
| forest | deep_layers | 0.0851 | 0.1951 | 0.1969 | -0.0018 |
| forest | early_layers | 0.1270 | 0.2393 | 0.1969 | 0.0424 |
| forest | octaves_plus_intensity | 0.1754 | 0.2882 | 0.1969 | 0.0913 |

## Findings

- Early-layer conditions generally amplified edge and texture structure, consistent with low-level CNN feature behavior.
- Deeper layers produced more object-like and composite motifs, though edge-density changes varied by input.
- The higher-intensity multi-octave condition produced the largest MAD across all three images, indicating the strongest overall transformation.
- The effect depended strongly on source imagery: structured city scenes, natural textures, and object-centric imagery responded differently to the same activation objectives.

## Why this matters for the repository

The current codebase turns that experimental work into a reusable, config-driven batch runner with validation, resume behavior, tiled gradients, failure isolation, tests, and reproducible execution. The goal is not to present DeepDream as state-of-the-art generative AI; it is to demonstrate controlled experimentation, model introspection, gradient-based image optimization, and reliable execution patterns.
