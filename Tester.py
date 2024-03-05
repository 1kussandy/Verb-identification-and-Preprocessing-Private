import os
import nltk
from nltk.tag import DefaultTagger, UnigramTagger, BigramTagger

class Tester:
    def __init__(self, training_file_path, input_file_path):
        training_data = []
        sentence = []
        with open(training_file_path, 'r', encoding='utf-8') as training_file:
            for line in training_file:
                if line.strip():
                    token, pos_tag, chunking_data = line.strip().split()
                    sentence.append((token, pos_tag))
                else:
                    if sentence:
                        training_data.append(sentence)
                        sentence = []
        self.sentences = training_data
        self.input_file_path = input_file_path

    def train(self):
        default_tagger = DefaultTagger('NN')
        unigram_tagger = UnigramTagger(self.sentences, backoff=default_tagger)
        bigram_tagger = BigramTagger(self.sentences, backoff=unigram_tagger)
        return bigram_tagger

    def tag(self, sentence, tagger):
        tagged_sentence = tagger.tag([sentence])
        return (str(tagged_sentence[0][0]), str(tagged_sentence[0][1]))
    
    def generateLabels(self):
        trained_tagger = self.train()
        output_directory = os.path.dirname(self.input_file_path)
        output_file_path = os.path.join(output_directory, 'test_output.txt')
        tags = []
        with open(self.input_file_path, 'r', encoding='utf-8') as input_file:
            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                for line in input_file:
                    line = line.strip()
                    if line:
                        tagged_line = self.tag(line, trained_tagger)
                        tags.append(tagged_line)
                        output_file.write(tagged_line[0] + ' ' + tagged_line[1] + '\n')  
        accuracy = trained_tagger.accuracy([tags])
        print(f"Accuracy: {accuracy * 100:.2f}%")
    
    def test(self, output_file_path, trained_tagger):
        tags = []
        with open(output_file_path, 'r', encoding='utf-8') as output_file:
            for line in output_file:
                line = line.strip().split()
                if line:
                    tags.append((line[0], line[1]))
        accuracy = trained_tagger.accuracy([tags])
        print(f"Accuracy: {accuracy * 100:.2f}%")
        



