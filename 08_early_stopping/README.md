# Early Stopping in Artificial Neural Networks

## Objective

To prevent overfitting during neural network training using Early Stopping.

## Dataset

Graduate Admission Prediction Dataset

Target:

* Chance of Admit

## Approach

1. Split dataset into training and testing sets.
2. Scale features using MinMaxScaler.
3. Build a simple feed-forward neural network.
4. Apply EarlyStopping callback.
5. Monitor validation loss during training.
6. Restore the best model weights automatically.

## Early Stopping Configuration

* Monitor: validation loss
* Patience: 10 epochs
* Restore Best Weights: True

## Benefits

* Prevents overfitting.
* Saves training time.
* Automatically selects the best model.
* Improves generalization on unseen data.

## Conclusion

Early Stopping is an effective regularization technique that stops training when validation performance stops improving, preventing unnecessary training and reducing overfitting.
