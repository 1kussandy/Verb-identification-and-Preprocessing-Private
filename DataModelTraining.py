# Step 4: Model Training
# Train each of your models using the preprocessed training data and the selected features.

def train_models(models, x_train, y_train):
    # Train Bayesian Classifier
    models[0].fit(x_train, y_train)

    # Train Logistic Regression
    models[1].fit(x_train, y_train)

    # Train SVM
    models[2].fit(x_train, y_train)