# Step 5: Model Evaluation
# Evaluate the performance of your models using a dev set and appropriate evaluation metrics.

from sklearn.metrics import accuracy_score

# Make predictions
bayesian_predictions = bayesian_classifier.predict(X_dev)
logistic_predictions = logistic_regression.predict(X_dev)
svm_predictions = svm_classifier.predict(X_dev)

# Calculate accuracy
bayesian_accuracy = accuracy_score(y_dev, bayesian_predictions)
logistic_accuracy = accuracy_score(y_dev, logistic_predictions)
svm_accuracy = accuracy_score(y_dev, svm_predictions)