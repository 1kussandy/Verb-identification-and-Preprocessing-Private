import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

class BayesianClassifier:
    def __init__(self, training_file_path):
        self.vectorizer = CountVectorizer()
        self.classifier = MultinomialNB()
        self.training_file_path = training_file_path

    def _load_training_data(self, training_file_path):
        with open(training_file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            training_data = []
            for line in lines:
                line = line.strip()
                if line:
                    parts = line.split()
                    token = parts[0]
                    tag = parts[1]
                    training_data.append(f"{token} {tag}")
        return training_data

    def train(self):
        training_data = self._load_training_data(self.training_file_path)
        labels = [line.split()[-1] for line in training_data]
        X = self.vectorizer.fit_transform(training_data)
        self.classifier.fit(X, labels)

    def generate_tags(self, input_file_path):
        input_folder = os.path.dirname(input_file_path)
        output_file_path = os.path.join(input_folder, 'bayes_output.txt')
        with open(input_file_path, 'r', encoding='utf-8') as input_file, \
            open(output_file_path, 'w', encoding='utf-8') as output_file:
            for line in input_file:
                line = line.strip()
                if line:
                    X_new = self.vectorizer.transform([line])
                    predicted_class = self.classifier.predict(X_new)[0]
                    output_file.write(f"{line} {predicted_class}\n")


