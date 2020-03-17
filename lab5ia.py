from sklearn import preprocessing
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

train_labels = np.load('training_labels.npy', allow_pickle=True)  # incarcam imaginile
train_sentences = np.load('training_sentences.npy', allow_pickle=True)  # incarcam etichetele avand

test_labels = np.load('test_labels.npy', allow_pickle=True)  # incarcam imaginile
test_sentences = np.load('test_sentences.npy', allow_pickle=True)  # incarcam etichetele avand

def normalize_data(train_data, test_data, type=None):
    sc = None
    if type == 'standard':
        sc = preprocessing.StandardScaler()
    if type == 'l1':
        sc = preprocessing.Normalizer(norm='l1')
    if type == 'l2':
        sc = preprocessing.Normalizer(norm='l2')
    if type == 'min_max':
        sc = preprocessing.MinMaxScaler()
    bag = BagOfWords()
    bag.build_vocabulary(np.append(train_data, test_data))
    train_data_temp = bag.get_features(train_data)
    sc.fit(train_data_temp)
    sc_train = sc.transform(train_data_temp)
    test_data_temp = bag.get_features(test_data)
    sc_test = sc.transform(test_data_temp)
    return sc_train, sc_test, bag


class BagOfWords():
    def __init__(self):
        self.vocab = {}
        self.words = []
        self.vocab_len = 0

    def build_vocabulary(self, data):
        for msg in data:
            for word in msg:
                if word not in self.vocab:
                    self.vocab[word] = self.vocab_len
                    self.vocab_len = self.vocab_len + 1
                    self.words.append(word)

    def get_features(self, data):
        features = np.zeros((len(data), self.vocab_len))
        for i in range(0, len(data)):
            msg = data[i]
            for word in msg:
                features[i][self.vocab[word]] += 1
        return features


#bag_s=BagOfWords()
#bag_s.build_vocabulary(train_sentences)
#print(bag_s.vocab_len)

normalized_data = normalize_data(train_sentences,test_sentences, 'l2')
bag = BagOfWords()
bag.build_vocabulary(np.append(train_sentences,test_sentences))

clf = svm.SVC(1,'linear')
clf.fit(normalized_data[0],train_labels)
predictions = clf.predict(normalized_data[1])

print("accuracy:" + accuracy_score(test_labels, predictions).__str__())
print("f1 score:")
print(f1_score(test_labels, predictions, average=None))

def topCoef(classifier, feature_names, top_features=10):
    coef = classifier.coef_.ravel()
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    print("-----Top Positive:-----")
    for i in range(10):
        print(feature_names[top_coefficients[i]])
    print("-----Top Negative:-----")
    for i in range(10,20):
        print(feature_names[top_coefficients[i]])

topCoef(clf, normalized_data[2].words)