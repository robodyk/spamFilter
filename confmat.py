class BinaryConfusionMatrix:

    def __init__(self, pos_tag, neg_tag):
        self.pos_tag = pos_tag
        self.neg_tag = neg_tag
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    def as_dict(self):
        return {
            'tp': self.tp,
            'fp': self.fp,
            'tn': self.tn,
            'fn': self.fn
        }

    def update(self, truth, prediction):
        self.check_values(prediction, truth)
        if prediction == self.pos_tag:
            if prediction == truth:
                self.tp += 1
            else:
                self.fp += 1
        elif prediction == truth:
            self.tn += 1
        else:
            self.fn += 1

    def check_values(self, prediction, truth):
        if truth != self.pos_tag and truth != self.neg_tag:
            raise ValueError
        if prediction != self.pos_tag and prediction != self.neg_tag:
            raise ValueError

    def compute_from_dicts(self, truth_dict, pred_dict):
        for email in pred_dict:
            self.update(truth_dict[email], pred_dict[email])
