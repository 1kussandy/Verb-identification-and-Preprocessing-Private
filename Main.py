import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split

from Feature import extract_features
from BayesianClassifier import BayesianClassifier
from LogisticRegression import LogisticRegressionMODEL
from SVMClassifier import SVMClassifier


def main():
    training_file_path = 'train.txt'
    input_file_path = 'unlabeled_test_test.txt'
    
    training_data = pd.read_csv(training_file_path, header=None, delimiter=' ', names=['Token', 'POS', 'Chunking'])
    training_features = extract_features(training_data)
    
    input_data = []
    with open(input_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            input_data.append(line)
    
    input_features = pd.DataFrame({'Token': input_data})
    input_features = extract_features(input_features)
    
    [x_train, x_test, y_train, y_test] = train_test_split(training_features, training_data['POS'], test_size=0.2)
    
    vectorizer = DictVectorizer()
    x_train = vectorizer.fit_transform(x_train)
    x_test = vectorizer.transform(x_test)
    input_data = vectorizer.transform(input_features)
    
    all_data = [x_train, x_test, y_train, y_test]
    
    bc = BayesianClassifier(training_file_path, input_file_path)
    bc.generate_tags()
    
    lg = LogisticRegressionMODEL(all_data)
    lg.train()
    lg.tag(input_data)
    
    svm = SVMClassifier(all_data)
    svm.train()
    svm.tag(input_data)
        
main()
