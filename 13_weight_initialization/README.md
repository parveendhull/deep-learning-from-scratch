# Weight Initialization Techniques

## What is Covered
Comparison of 3 weight initialization strategies
on identical 3-layer network architecture
(15 → 30 → 30 → 1), trained with manual
forward + backward propagation.

## Techniques Compared

### Zero Initialization
W = 0 for all weights

Problem: Symmetry — every neuron in a layer
receives identical gradients during backprop.
All neurons learn the exact same function.
Network effectively collapses to 1 neuron per layer.

### Random Initialization
W = np.random.randn() * 0.8

Breaks symmetry, but large scale can cause
unstable activations and slow/erratic convergence,
especially in deeper networks.

### He Initialization
W = np.random.randn() * sqrt(2/n)
where n = number of input units to the layer

Designed specifically for ReLU activations.
Scales variance based on layer size —
keeps activations stable across layers.

## Architecture
Input(15) → Dense(30, ReLU) → Dense(30, ReLU) → Dense(1, Sigmoid)
Loss: Binary Cross-Entropy
Manual forward and backward pass — no framework used

## Results
| Initialization | Convergence Behavior |
|-----------------|----------------------|
| Zero            | Flat loss — no learning (symmetry problem) |
| Random (0.8)    | Some learning, less stable |
| He              | Smooth, fastest convergence |

## Key Insight
Zero initialization fails completely — not because
of bad luck, but a structural problem: all neurons
in a layer stay identical forever since they receive
identical gradients every step.

He initialization works best with ReLU because it
accounts for the number of inputs, preventing
activations from exploding or vanishing across layers.

## Why This Matters
Proper initialization is not a minor detail —
it determines whether a deep network can learn
at all. This connects directly to the vanishing/
exploding gradient problem in deep networks.