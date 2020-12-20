from math import exp, log
from corpus import Corpus
import utils
import quality
import numpy as np
import pickle

class Base_filter():
    def __init__(self, pos_tag = "SPAM", neg_tag = "OK"):
        self.pos_tag = pos_tag
        self.neg_tag = neg_tag

    def evaluate_mail(self, file_path):
        raise NotImplementedError

    def test(self, mails_path):
        corpus = Corpus(mails_path)
        truth_dict = utils.read_classification_from_file(mails_path + "/!truth.txt")
        with open(mails_path + "/!prediction.txt", 'a', encoding='utf-8') as f:
            for mail in corpus.emails():
                f.write(f"{mail[0]} {self.evaluate_mail(mail[1])}\n")
        pred_dict = utils.read_classification_from_file(mails_path + '/!truth.txt')
        qual = quality.compute_quality_for_corpus(mails_path)
        print (f"Filter score: {qual}")

    def get_feature_vector(self,email: str):
        # TODO : přečte daný email a vrátí jeho "feature vector" tj. list hodnot na základě kterých se email filtruje
        # např: [hustota slova, hustota jiného slova, hustota velkých písmen, hustota znaku,atd...]
        # PLR filtr vektor rozdeluje na podvektory, proto je potreba aby "tematicky podobne hodnoty" byly vedle sebe
        return None


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

    def split_feature_vector(self, feature_vector):
        feature_vectors = []
        for size in self.vector_sizes:
            try:
                feature_vectors.append(feature_vector[:size])
                feature_vector = feature_vector[size:]
            except IndexError:
                feature_vectors.append(feature_vector)
        return feature_vectors

    def sigmoid(self, x):
        return 1/(1+exp(-x))

    def evaluate_mail(self, email):
        feature_vectors = email.get_feature_vector_prototype()
        spam_odds = self.spam_distribution ** (self.subvector_count - 1)
        for i in range(self.subvector_count):
            probability = self.sigmoid(np.dot(feature_vectors[i], self.weights[i]) + self.biases[i])
            spam_odds *= probability/(1-probability)
        return True if log(spam_odds) <= 0 else False

    def gradient_descent(self, batch, lr, max_steps):
        feature_vectors = [(m[0].get_feature_vector_prototype()) for m in batch]
        y = [m[1] for m in batch]
        partial_derivatives_w = [n*[0] for n in self.vector_sizes]
        partial_derivatives_b = self.subvector_count*[0]
        for s in range(max_steps):
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
        while got_data:
            batch = []
            for i in range(batch_size):
                try:
                    email = next(corpus.emails())
                    batch.append((email[1], 1 if truth_dict[email[0]] == self.pos_tag else 0))
                except StopIteration:
                    got_data = False
                    break
            self.gradient_descent(batch, learning_rate, max_steps)
            print(f"trained on batch #{batch_count}")
            batch_count +=1

    def save_paremeters(self):
        with open("learned/weights.data",'w') as f:
            pickle.dump(self.weights, f)
        with open("learned/biases.data", 'w') as f:
            pickle.dump(self.weights, f)

if __name__ == '__main__':
    filtr_sn1 = PLR_filter(5,[3,10,3,10,1],[3*[1],10*[1],3*[1],10*[1],[1]],5*[0.1])
    print("Started training")
    filtr_sn1.train('data/1')
    print("finished training")
    filtr_sn1.test('data/2')
    filtr_sn1.save_paremeters()