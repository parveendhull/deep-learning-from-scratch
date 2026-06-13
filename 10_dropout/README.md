# Dropout Regularization

## What is Dropout
During training, randomly "drops" neurons
with probability p — forces network to learn
redundant representations.

No neuron can rely on others — all learn independently.
At test time, all neurons active but scaled by (1-p).

## Why it Works
Without Dropout:
- Some neurons become dominant
- Others become lazy
- Model memorizes training data = overfit

With Dropout:
- Every neuron must independently learn
- No single neuron dominates
- Better generalization on test data

## Architecture
Without Dropout:
Input → Dense(128, ReLU) → Dense(128, ReLU) → Output

With Dropout:
Input → Dense(128, ReLU) → Dropout(0.5)
      → Dense(128, ReLU) → Dropout(0.5) → Output

## Results
| Model          | Accuracy |
|----------------|----------|
| Without Dropout| overfits |
| With Dropout   | better generalization |

## Key Parameters
patience  → how many epochs to wait
p = 0.5   → 50% neurons dropped each step
Higher p  → more regularization
Lower p   → less regularization

## When to Use
Large networks prone to overfitting
Small datasets
When train accuracy >> test accuracy