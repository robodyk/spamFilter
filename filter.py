from math import exp, log
from corpus import Corpus
import utils
import quality
import numpy as np
import pickle
import os

class Base_filter():
    def __init__(self, pos_tag = "SPAM", neg_tag = "OK"):
        self.pos_tag = pos_tag
        self.neg_tag = neg_tag

    def test(self, mails_path):
        try:
            os.remove(mails_path + "/!prediction.txt")
        except:
            pass
        corpus = Corpus(mails_path)
        truth_dict = utils.read_classification_from_file(mails_path + "/!truth.txt")
        with open(mails_path + "/!prediction.txt", 'a', encoding='utf-8') as f:
            for mail in corpus.emails():
                f.write(f"{mail[0]} {self.pos_tag if self.evaluate_mail(mail[1]) else self.neg_tag}\n")
        pred_dict = utils.read_classification_from_file(mails_path + '/!truth.txt')
        qual = quality.compute_quality_for_corpus(mails_path)
        print (f"Filter score: {qual}")

class PLR_filter(Base_filter):
    """
    Chang, Ming-Wei & Yih, Wen-tau & Meek, Christopher. (2008). Partitioned logistic regression for spam filtering.
    Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.
    97-105. 10.1145/1401890.1401907.
    """
    def __init__(self, subvector_count, vector_sizes, weights=None, biases=None, spam_distribution=0.5):
        super().__init__()
        self.vector_sizes = vector_sizes
        self.subvector_count = subvector_count
        self.spam_distribution = spam_distribution
        if not weights:
            with open ("learned/weights.data", "r") as f:
                self.weights = pickle.load(f)
        else:
            self.weights = weights
        if not biases:
            with open ("learned/biases.data") as f:
                self.biases = pickle.load(f)
        else:
            self.biases = biases

    def sigmoid(self, x):
        if x > 500:
            return 1

        if x < -500:
            return 0

        return exp(x)/(1+exp(x))

    def evaluate_mail(self, email):
        feature_vectors = email.get_feature_vector_prototype()
        spam_odds = self.spam_distribution ** (self.subvector_count - 1)
        for i in range(self.subvector_count):
            probability = self.sigmoid(np.dot(feature_vectors[i], self.weights[i]) + self.biases[i])
            spam_odds *= (probability/(1-probability)) if probability != 1 else 9999999999
        return False if spam_odds <= 1 else True

    def gradient_descent(self, batch, lr, max_steps):
        feature_vectors = [(m[0].get_feature_vector_prototype()) for m in batch]
        y = [m[1] for m in batch]
        for s in range(max_steps):
            partial_derivatives_w = [n * [0] for n in self.vector_sizes]
            partial_derivatives_b = self.subvector_count * [0]
            for batch_index, vector in enumerate(feature_vectors):
                for i, subvector in enumerate(vector):
                    partial_derivatives_b[i] -= lr *\
                        (1/len(batch))*(self.sigmoid(np.dot(subvector, self.weights[i]) + self.biases[i]) - y[batch_index])
                    for j, xij in enumerate(subvector):
                        partial_derivatives_w[i][j] -= lr *\
                        (1/len(batch))*xij*(self.sigmoid(np.dot(subvector, self.weights[i]) + self.biases[i]) - y[batch_index])
            for i in range(self.subvector_count):
                self.weights[i] = np.add(self.weights[i], partial_derivatives_w[i])
            self.biases = np.add(self.biases, partial_derivatives_b)


    def train(self, file_path, batch_size=10, learning_rate=0.1, max_steps = 1000):
        corpus = Corpus(file_path)
        truth_dict = utils.read_classification_from_file(file_path + "/!truth.txt")
        got_data = True
        batch_count = 1
        mails_getter = corpus.emails()
        while got_data:
            batch = []
            for i in range(batch_size):
                try:
                    email = next(mails_getter)
                    batch.append((email[1], 1 if truth_dict[email[0]] == self.pos_tag else 0))
                except StopIteration:
                    got_data = False
                    break
            self.gradient_descent(batch, learning_rate, max_steps)
            print(f"trained on batch #{batch_count}")
            batch_count +=1

    def save_paremeters(self):
        return
        try:
            os.mkdir('learned')
        except:
            pass
        with open("learned/weights.data",'w') as f:
            pickle.dump(str(self.weights), f)
        with open("learned/biases.data", 'w') as f:
            pickle.dump(str(self.biases), f)

if __name__ == '__main__':
    filtr_sn1 = PLR_filter(7,[3,11,3,11,1,20,13],[3*[1],11*[1],3*[1],11*[1],[0.1],20*[0.3],13*[0.5]],7*[0.1])
    print("Started training")
    filtr_sn1.train('data/1', 10, 0.1, 1000)
    print("finished training")
    filtr_sn1.test('data/2')
    filtr_sn1.save_paremeters()