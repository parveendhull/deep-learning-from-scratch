# L1 and L2 Regularization

## What is Regularization
Technique to prevent overfitting by adding
a penalty term to the loss function.
Forces model to keep weights small.

## L1 Regularization (Lasso)
Loss = BCE + λ Σ|w|

Penalty = sum of absolute values of weights
Effect  = drives some weights to exactly zero
Use     = feature selection — sparse models

## L2 Regularization (Ridge)
Loss = BCE + λ Σw²

Penalty = sum of squared weights
Effect  = shrinks all weights but none to zero
Use     = general overfitting prevention

## Comparison
| Model              | Behavior                    |
|--------------------|-----------------------------|
| No Regularization  | Overfits — memorizes data   |
| L1 (Lasso)         | Sparse weights — some zero  |
| L2 (Ridge)         | Small weights — none zero   |

## Key Parameter — λ (lambda)
λ = 0.001 used here
Higher λ → stronger penalty → underfitting risk
Lower λ  → weaker penalty  → overfitting risk

## Dataset
make_moons — non-linear, 100 samples, noise=0.25

## Key Insight
L1 → good when many features irrelevant
L2 → good for general regularization
Both → better than no regularization on small datasets