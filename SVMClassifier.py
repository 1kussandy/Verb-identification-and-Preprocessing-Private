from sklearn.svm import LinearSVC


class SVMClassifier:
    def __init__(self, data):
        self.data = data
        self.model = None

    def train(self):
        # Create and train the model
        self.model = LinearSVC()
        self.model.fit(self.data[1], self.data[3])

        # print accuracy
        training_accuracy = self.model.score(self.data[1], self.data[3])
        print(f"SVM Training Accuracy: {training_accuracy * 100:.2f}%")

    def tag(self, data):
        if self.model is None:
            raise ValueError("model not properly trained")
        predictions = self.model.predict(data)

        input_file_path = 'unlabeled_test_test.txt'
        output_file_path = 'SVM_output.txt'

        with open(input_file_path, 'r', encoding='utf-8') as input_file:
            input_data = input_file.readlines()

        # Write the original word and predicted tag for each line to the output file
        with (open(output_file_path, 'w', encoding='utf-8') as output_file):
            for original_word, predicted_tag in zip(input_data, predictions):
                original_word = original_word.strip()
                if original_word == "":
                    output_line = "\n"
                else:
                    output_line = f"{original_word} {predicted_tag}\n"
                output_file.write(output_line)

if __name__ == "__main__":
    SVMClassifier.main()
