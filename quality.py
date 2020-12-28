from utils import read_classification_from_file
import confmat


def compute_quality_for_corpus(corpus_dir, fn_weight=1, fp_weight=10, pos_tag="SPAM", neg_tag="OK"):
    matrix = confmat.BinaryConfusionMatrix(pos_tag, neg_tag)
    truth_dict = read_classification_from_file(corpus_dir + "/!truth.txt")
    pred_dict = read_classification_from_file(corpus_dir + "/!prediction.txt")
    matrix.compute_from_dicts(truth_dict, pred_dict)
    print (f"Fp: {matrix.fp}\nFn: {matrix.fn}\nTp: {matrix.tp}\nTn: {matrix.tn}")
    return (matrix.tp + matrix.tn) / (fn_weight * matrix.fn + fp_weight * matrix.fp + matrix.tp + matrix.tn)
