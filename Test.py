import os
from Tester import Tester
from nltk.corpus.reader import tagged

class Test:
    @staticmethod
    def main():
        training_file_path = 'train.txt'
        input_file_path = 'unlabeled_test_test.txt'
        output_file_path = 'bayes_output.txt'
        pos_tagger = Tester(training_file_path, input_file_path)
        pos_tagger.generateLabels()
        pos_tagger.test(output_file_path, pos_tagger.train())

if __name__ == "__main__":
    Test.main()
