#Step 6: Combining Models (Optional)
#You can combine the predictions from multiple models, if desired, to create an ensemble.

from scipy.stats import mode
from sklearn.metrics import accuracy_score

# Create an ensemble by majority voting
ensemble_predictions = mode([bayesian_predictions, logistic_predictions, svm_predictions], axis=0)
ensemble_accuracy = accuracy_score(y_dev, ensemble_predictions[0])