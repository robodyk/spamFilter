from collections import Counter
from math import exp, log
from corpus import Corpus
import utils
import numpy as np
import pickle
import os
from matplotlib import pyplot as plt


class Base_filter():
    def __init__(self, pos_tag="SPAM", neg_tag="OK"):
        self.pos_tag = pos_tag
        self.neg_tag = neg_tag

    def test(self, mails_path):
        try:
            os.remove(mails_path + "/!prediction.txt")
        except:
            pass
        corpus = Corpus(mails_path)
        with open(mails_path + "/!prediction.txt", 'a', encoding='utf-8') as f:
            for mail in corpus.emails():
                res = self.evaluate_mail(mail[1])
                f.write(f"{mail[0]} {self.pos_tag if res else self.neg_tag}\n")


class PLR_filter(Base_filter):
    """
    With given data performs worse than Naive Bayes filter.

    Chang, Ming-Wei & Yih, Wen-tau & Meek, Christopher. (2008). Partitioned logistic regression for spam filtering.
    Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.
    97-105. 10.1145/1401890.1401907.

    optimized by gradient descent
    """

    def __init__(self, subvector_count, vector_sizes, weights=None, biases=None, spam_distribution_odds=1, learned_dir= 'learned_plr'):
        super().__init__()
        self.learned_dir = learned_dir
        self.vector_sizes = vector_sizes
        self.subvector_count = subvector_count
        self.spam_distribution = spam_distribution_odds
        if not weights:
            with open(f"{self.learned_dir}/weights.data", "rb") as f:
                self.weights = pickle.load(f)
        else:
            self.weights = weights
        if not biases:
            with open(f"{self.learned_dir}/biases.data", "rb") as f:
                self.biases = pickle.load(f)
        else:
            self.biases = biases
        self.bias_momentums = []
        self.weight_momentums = []

    def init_momentums(self):  # sets momentums to 0
        self.bias_momentums = self.subvector_count * [0]
        self.weight_momentums = [c * [0] for c in self.vector_sizes]

    def sigmoid(self, x):
        if x > 30:  # the filter does not need a higher level of precision
            return 0.9999999

        if x < -30:
            return 0.0000001

        return exp(x) / (1 + exp(x))

    # determines the odds of mail being spam according to the formula in the cited paper (chapter 2)
    def evaluate_mail(self, email, tuning=None):
        feature_vectors = email.get_feature_vector_plr()
        spam_odds = self.spam_distribution ** (self.subvector_count - 1)

        for i in range(self.subvector_count):
            probability = self.sigmoid(np.dot(feature_vectors[i], self.weights[i]) + self.biases[i])
            spam_odds *= (probability / (1 - probability)) if probability != 1 else 9999999999

        spam_odds *= email.check_spam_status()
        return False if spam_odds <= 1 else True

    # Computes a gradient - vector of steepest descent of the loss function of logistic regression on a given subvector
    # and takes a step proportional to learning rate (lr), momentum parameter adds some quotient of previous step to
    # that step this can help avoiding local minima, y  is truth vector and its elements are 1 if positive and 0 if neg.
    def gradient_descent(self, subvector_index, y, batch, lr, momentum):
        steps_w = len(self.weights[subvector_index]) * [0]
        step_b = 0
        for batch_index, vector in enumerate(batch):  # iterates over each subvector of a batch
            sig = self.sigmoid(np.dot(vector, self.weights[subvector_index]) + self.biases[subvector_index])
            # this computes the step taken on the bias
            step_b = self.bias_momentums[subvector_index] * momentum + \
                                   lr * (1 / len(batch)) * (sig - y[batch_index])  # d(loss) / db
            # this computes the step taken on each of the weights
            for j, xij in enumerate(vector):
                steps_w[j] = self.weight_momentums[subvector_index][j] * momentum + \
                                           lr * (1 / len(batch)) * xij * (sig - y[batch_index])  # d(loss) / dw

        self.weight_momentums = steps_w
        self.bias_momentums = step_b
        self.weights[subvector_index] = list(np.add(self.weights[subvector_index], steps_w))
        self.biases[subvector_index] += step_b
        return

    # epochs = iterations over dataset
    def train(self, file_path, batch_size=10, learning_rate=0.1, lr_decay =0.05, epochs=1000, momentum=0.0):
        corpus = Corpus(file_path)
        truth_dict = utils.read_classification_from_file(file_path + "/!truth.txt")
        got_data = True
        mails_getter = corpus.emails()
        batches = []
        # loads all data from directory in batches of given size
        while got_data:
            batch = []
            # loads a batch of given size, a smaller one if out of data
            for i in range(batch_size):
                try:
                    email = next(mails_getter)
                    batch.append((email[1], 1 if truth_dict[email[0]] == self.pos_tag else 0))
                except StopIteration:
                    got_data = False
                    break
            batches.append(batch)
        for e in range(epochs):  # trains multiple times on all batches
            self.init_momentums()
            for batch in batches:  # performs gradient descent on each bach
                # gets feature vectors for batch
                feature_vectors = [(m[0].get_feature_vector_plr()) for m in batch]  # gets feature vectors of the batch
                y = [m[1] for m in batch]  # gets the truth vector of the batch
                for i in range(self.subvector_count):  # weights for each subvector are trained separately
                    subvector_batch = [v[i] for v in feature_vectors]  # isolates a subvector from all vectors
                    self.gradient_descent(i, y, subvector_batch, learning_rate, momentum)
                print(f"trained on epoch #{e +1}")
            learning_rate *= 1/(1 + lr_decay * e)


    def save_paremeters(self):
        try:
            os.mkdir(self.learned_dir)
        except:
            pass
        with open(f"{self.learned_dir}/weights.data", 'wb+') as f:
            pickle.dump(self.weights, f)
        with open(f"{self.learned_dir}/biases.data", 'wb+') as f:
            pickle.dump(self.biases, f)

class LR_filter(PLR_filter):
    """
    With given data performs worse than Naive Bayes filter.

    Logistic regression optimized by gradient descent
    """
    def __init__(self, vector_size, weights= None, bias= None, spam_distribution_odds= 3, learned_dir = 'learned_lr'):
        self.learned_dir = 'learned_lr'
        super().__init__(None, None, weights, bias, spam_distribution_odds, learned_dir)
        self.vector_size = vector_size

    def init_momentums(self):
        self.weight_momentums = self.vector_size * [0]
        self.bias_momentums = 0

    def evaluate_mail(self, email, tuning):  # does not split the feature vector, so the result is directly applied
        vector = email.get_feature_vector_lr()
        probability = self.sigmoid(np.dot(vector, self.weights) + self.biases)
        probability *= self.spam_distribution/ (self.spam_distribution +1)
        probability *= email.check_spam_status()
        return True if probability > 0.5 else False

    def train(self, file_path, batch_size=10, learning_rate=0.1, lr_decay = 0.05, epochs=1000, momentum=0.0,
              tuning= False):  # analogous to PLR_filter, tuning parameter plots out a graph of mean loss over epochs
        if tuning:
            og_lr =learning_rate # original learning rate
            x_plt = []  # x-axis (epochs)
            y_plt = []  # y-axis (mean loss)
        corpus = Corpus(file_path)
        truth_dict = utils.read_classification_from_file(file_path + "/!truth.txt")
        got_data = True
        mails_getter = corpus.emails()
        batches = []
        while got_data:
            batch = []
            for i in range(batch_size):
                try:
                    email = next(mails_getter)
                    batch.append((email[1], 1 if truth_dict[email[0]] == self.pos_tag else 0))
                except StopIteration:
                    got_data = False
                    break
            batches.append(batch)

        for e in range(epochs):
            if tuning:
                steps = 0
            print(learning_rate)
            self.init_momentums()
            loss = 0
            for batch in batches:
                batch_vectors = [(m[0].get_feature_vector_lr()) for m in batch]
                y = [m[1] for m in batch]
                loss += self.gradient_descent(y, batch_vectors, learning_rate, momentum)
                if tuning:
                    steps +=1
                print(f"trained on epoch #{e +1}")
            learning_rate *= 1 / (1 + lr_decay * e)
            if tuning:
                y_plt.append(loss/steps)
                x_plt.append(e)
        if tuning:
            plt.plot(x_plt, y_plt)
            plt.title(f"lr:{og_lr} lrd:{lr_decay} bs:{batch_size} m: {momentum} e:{epochs}")
            plt.xlabel("epochs")
            plt.ylabel("mean loss")
            plt.show()

    def gradient_descent(self, y, batch, lr, momentum):  # same as PLR, but has 1 parameter less <- no subvectors
        steps_w = len(self.weights) * [0]
        step_b = 0
        loss = 0
        for batch_index, vector in enumerate(batch):
            sig = self.sigmoid(np.dot(vector, self.weights) + self.biases)
            if sig == 0 or sig == 1:
                print("extreme sig")
            step_b = self.bias_momentums * momentum - \
                                   lr * (1 / len(batch)) * (sig - y[batch_index])

            for j, xij in enumerate(vector):
                steps_w[j] = self.weight_momentums[j] * momentum - \
                                           lr * (1 / len(batch)) * xij * (sig - y[batch_index])
            loss -= (1 / len(batch)) * (y[batch_index] * log(sig) + (1 - y[batch_index]) * log(1 - sig))
        self.weight_momentums = steps_w
        self.bias_momentums = step_b
        self.weights = list(np.add(self.weights, steps_w))
        self.biases += step_b
        return loss  # also returns loss


class MyFilter(Base_filter):
    """
    Naive Bayes. Simple but effective
    """
    def __init__(self, data_dir = "learned_nb"):
        super().__init__()
        self.content_spam_dict = {}  # will contains probabilities of words occuring in spam/ham mails
        self.content_ham_dict = {}
        self.spam_probability = 0.5  # determined by distribution, default 0.50
        self.ham_probability = 0.5
        self.data_dir = data_dir
        self.trained = self.load_dicts()  # this field contains true if some training data is loaded

    def train(self, file_path):
        self.content_spam_dict = {}
        self.content_ham_dict = {}
        class_dict = utils.read_classification_from_file(file_path + '/!truth.txt')
        corpus = Corpus(file_path)
        email_generator = corpus.emails()
        content_counter_spam = Counter()
        content_counter_ham = Counter()
        content_wordcount_spam = 0
        content_wordcount_ham = 0

        spam_count = 0
        ham_count = 0
        every_word_content = set()

        for mail in email_generator:
            content_words = self.string_to_words(mail[1].content_no_html)
            content_counter = Counter(content_words)
            for word in content_words:
                every_word_content.add(word)

            if class_dict[mail[0]] == self.pos_tag:
                spam_count += 1
                content_counter_spam += content_counter
                content_wordcount_spam += len(content_words)
            else:
                ham_count += 1
                content_counter_ham += content_counter
                content_wordcount_ham += len(content_words)

        for word in every_word_content:
            content_counter_ham[word] += 1
            content_counter_spam[word] += 1
            self.content_spam_dict[word] = content_counter_spam[word] / content_wordcount_spam
            self.content_ham_dict[word] = content_counter_ham[word] / content_wordcount_ham

        self.spam_probability = spam_count / (spam_count + ham_count)
        self.ham_probability = ham_count / (spam_count + ham_count)
        self.trained = True

    def get_score(self, elements, ham_dict, spam_dict):
        ham_score = 1
        spam_score = 1
        for word in elements:
            try:
                ham_score *= ham_dict[word]
                spam_score *= spam_dict[word]
                # because of following condition the more likely category can get rounded to inf
                if spam_score + ham_score == float('inf'):
                    break  # such large difference is conclusive
                # this prevents floats rounding to 0
                if min(utils.order_of_magnitude(ham_score), utils.order_of_magnitude(spam_score)) < -50:
                    ham_score *= 100000
                    spam_score *= 100000
            except KeyError:  # in case it encounters a new word
                pass
        return ham_score, spam_score

    def evaluate_mail(self, email):
        content_words = self.string_to_words(email.content_no_html)
        chars = email.content_no_html
        content_score = self.get_score(content_words, self.content_ham_dict, self.content_spam_dict)
        ham_score = self.ham_probability * content_score[0]
        spam_score = self.spam_probability * content_score[1]

        return True if spam_score > ham_score else False

    def string_to_words(self, str):
        words = str.lower()\
                .replace('.', '').replace(',', '').replace('!', '').replace('?', '').replace('\n', ' ')\
                .split()
        return words

    def save_dicts(self):
        try:
            os.mkdir(self.data_dir)
        except:
            pass
        with open(self.data_dir + "/spam_dict.data", "wb+") as f:
            pickle.dump(self.content_spam_dict, f)
        with open(self.data_dir + "/ham_dict.data", "wb+") as f:
            pickle.dump(self.content_ham_dict, f)
        with open(self.data_dir + "/spam_probability.data", "wb+") as f:
            pickle.dump(self.spam_probability, f)

    def load_dicts(self):
        try:
            with open(self.data_dir + "/spam_dict.data", "rb") as f:
                self.content_spam_dict = pickle.load(f)
            with open(self.data_dir + "/ham_dict.data", "rb") as f:
                self.content_ham_dict = pickle.load(f)
            with open(self.data_dir + "/spam_probability.data", "rb") as f:
                self.spam_probability = pickle.load(f)
            self.ham_probability = 1 - self.spam_probability
            return True
        except:
            return False


if __name__ == '__main__':
    filtr_sn3 = MyFilter("learned_nb")
    print(filtr_sn3.ham_probability)
    filtr_sn3.test('data/1')
    filtr_sn3.test('data/2')
    filtr_sn3.save_dicts()
