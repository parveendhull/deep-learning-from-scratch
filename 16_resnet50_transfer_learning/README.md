# Transfer Learning — Dogs vs Cats with ResNet50

## What is it
Using a CNN pretrained on ImageNet (1.4M images,
1000 classes) as a feature extractor, then fine-tuning
the top layers for the dogs-vs-cats task.

## Approach — Two Phase Training

### Phase 1: Feature Extraction
- Freeze entire ResNet50 (`base_model.trainable = False`)
- Train only the new classification head
- Only 2,049 trainable params (vs 23.5M frozen)

### Phase 2: Fine-Tuning
- Unfreeze last 30 layers of ResNet50
- Use a much smaller learning rate (1e-5 vs 1e-3)
- Allows the model to slightly adapt pretrained
  features to this specific dataset, without
  destroying the learned representations

## Why preprocess_input Matters
ResNet50 expects a specific normalization (ImageNet
mean/std per channel), different from simple /255
scaling. Using `preprocess_input` instead of manual
normalization is required for the pretrained weights
to work correctly.

## Results
| Model                          | Train Acc | Val Acc |
|----------------------------------|----------:|--------:|
| From-scratch CNN (project 15)   | 88.63%    | 89.88%  |
| ResNet50 (feature extraction)   | 98.23%    | 98.66%  |
| ResNet50 (fine-tuned)           | 99.38%    | 98.90%  |

Transfer learning improved validation accuracy by
~9 percentage points over the from-scratch CNN.

## Key Debugging Lesson — Input Size Matters
Initial runs used 128×128 input images, matching
the from-scratch CNN project. This caused highly
unstable training (validation accuracy fluctuating
between 60-77% across identical re-runs).

Root cause: ResNet50 was designed and pretrained on
224×224 images. At 128×128, the final feature map
shrinks to just 4×4 spatial dimensions — too small
to preserve meaningful spatial information for the
pretrained filters to work as intended.

Switching to 224×224 (ResNet50's native size)
immediately produced stable, high results (98%+)
across repeated runs.

Lesson: when using a pretrained model, always match
its expected input size — "any image size" is not
a safe assumption for transfer learning, even though
it works fine for a from-scratch CNN.

## Reproducibility
Random seeds fixed (tf, numpy, random) for consistent
results across reruns.