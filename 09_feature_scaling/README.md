# Feature Scaling in Artificial Neural Networks

## Overview

Feature Scaling is a preprocessing technique used to bring all input features to a similar numerical range before training a neural network.

Neural Networks are trained using Gradient Descent-based optimization algorithms. When features have very different scales, gradient updates become unstable, resulting in slower convergence and poor model performance.

---

## Dataset

Graduate Admission Prediction Dataset

### Features

* GRE Score
* TOEFL Score
* University Rating
* SOP
* LOR
* CGPA
* Research

### Target

* Chance of Admit

---

## Objective

To analyze the impact of Feature Scaling on the performance of an Artificial Neural Network (ANN).

---

## Methodology

1. Load the dataset.
2. Split data into training and testing sets.
3. Apply MinMaxScaler on training data.
4. Transform both training and testing datasets.
5. Train a feed-forward neural network using TensorFlow/Keras.
6. Evaluate model performance using the R² Score.

---

## Feature Scaling Technique

Min-Max Scaling was used:

[
X' = \frac{X - X_{min}}{X_{max} - X_{min}}
]

This transformation scales all feature values to the range:

[
[0,1]
]

---

## Why Feature Scaling Matters

The dataset contains features with very different magnitudes.

Example:

| Feature     | Approximate Range |
| ----------- | ----------------- |
| GRE Score   | 290 - 340         |
| TOEFL Score | 90 - 120          |
| CGPA        | 6 - 10            |
| Research    | 0 - 1             |

Without scaling:

* Large-valued features dominate gradient updates.
* Training becomes unstable.
* Optimization struggles to converge.

With scaling:

* All features contribute more evenly.
* Gradient Descent becomes more stable.
* Training converges faster and more effectively.

---

## Results

### Without Feature Scaling

* Neural network produced unstable predictions.
* Extremely poor R² score.
* Optimization failed to learn meaningful patterns.

### With Feature Scaling

* Stable training process.
* Better convergence.
* Significant improvement in model performance.

R² Score after scaling:

```text
0.731
```

---

## Key Takeaways

* Feature Scaling is highly important for Neural Networks.
* Gradient-based optimization methods are sensitive to feature magnitudes.
* Scaling improves convergence speed and training stability.
* MinMaxScaler and StandardScaler are commonly used techniques.

---

## Conclusion

Feature Scaling is one of the most important preprocessing steps in Deep Learning. By normalizing feature ranges, Neural Networks can learn more efficiently, resulting in faster convergence and improved predictive performance.
