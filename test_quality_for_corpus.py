#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for compute_quality_for_corpus function."""

import unittest
import os
import shutil
from test_readClassificationFromFile import (
    create_classification, 
    save_classification_to_file,
    replaced_open)

CORPUS_DIR = 'corpus_for_testing_delete_me'
TRUTH_FILENAME = '!truth.txt'
PREDICTION_FILANAME = '!prediction.txt'
SPAM_TAG = 'SPAM'
HAM_TAG = 'OK'


class ComputeQualityForCorpusTest(unittest.TestCase):

    def setUp(self):
        # Create a corpus directory
        os.makedirs(CORPUS_DIR, exist_ok=True)
        from quality import compute_quality_for_corpus
        self.compute_quality_for_corpus = compute_quality_for_corpus

    def tearDown(self):
        # Delete the corpus directory
        shutil.rmtree(CORPUS_DIR, ignore_errors=True)
    
    def test_allPredictionsCorrect(self):
        # Prepare the SUT
        # Create an artificial email classification dictionary  
        truth_dict = create_classification()
        create_truth_and_prediction_file(truth_dict, lambda x: x)
        # Excercise the SUT
        with replaced_open():   # Insist on explicit use of encoding
            q = self.compute_quality_for_corpus(CORPUS_DIR)
        # Assertions
        self.assertEqual(q, 1.0)
        
    def test_allPredictionsWrong(self):
        # Prepare the SUT
        # Create an artificial email classification dictionary  
        truth_dict = create_classification()
        create_truth_and_prediction_file(truth_dict, invert_classes)
        # Excercise the SUT
        with replaced_open():   # Insist on explicit use of encoding
            q = self.compute_quality_for_corpus(CORPUS_DIR)
        # Assertions
        self.assertEqual(q, 0.0)
        
    def test_allPredictionsSpam_for10SpamsAnd20Hams(self):
        # Prepare the SUT
        # Create an artificial email classification dictionary  
        truth_dict = create_classification(n_items=30, n_spams=10)
        create_truth_and_prediction_file(truth_dict, spams_only)
        # Since there are 10 spams and 20 hams in the corpus,
        # and predictions are all spam, then the confusion matrix 
        # shall have TP = 10 and FP = 20, zero negatives.
        # The modified accuracy then is:
        expected_q = 10 / (10 + 10*20)
        # Excercise the SUT
        with replaced_open():   # Insist on explicit use of encoding
            q = self.compute_quality_for_corpus(CORPUS_DIR)
        # Assertions
        self.assertEqual(q, expected_q)
        
    def test_allPredictionsHam_for10SpamsAnd20Hams(self):
        # Prepare the SUT
        # Create an artificial email classification dictionary  
        truth_dict = create_classification(n_items=30, n_spams=10)
        create_truth_and_prediction_file(truth_dict, hams_only)
        # Since there are 10 spams and 20 hams in the corpus,
        # and predictions are all ham, then the confusion matrix 
        # shall have TN = 20 and FN = 10, zero positives.
        # The modified accuracy then is:
        expected_q = 20 / (20 + 10)
        # Excercise the SUT
        with replaced_open():   # Insist on explicit use of encoding
            q = self.compute_quality_for_corpus(CORPUS_DIR)
        # Assertions
        self.assertEqual(q, expected_q)
        
    def test_1FP2FN_for10SpamsAnd20Hams(self):
        # Prepare the SUT
        # Create an artificial email classification dictionary  
        truth_dict = create_classification(n_items=30, n_spams=10)
        create_truth_and_prediction_file(truth_dict, n_FP_n_FN)
        # Since there are 10 spams and 20 hams in the corpus,
        # and predictions are all correct except 1 FP and 2 FN, 
        # then the confusion matrix shall have
        # FP = 1, FN = 2, 
        # TN = 19, TP = 8.
        # The modified accuracy then is:
        expected_q = (8 + 19) / (8 + 19 + 10*1 + 2)
        # Excercise the SUT
        with replaced_open():   # Insist on explicit use of encoding
            q = self.compute_quality_for_corpus(CORPUS_DIR)
        # Assertions
        self.assertEqual(q, expected_q)
        
        
def create_truth_and_prediction_file(truth_dict, pred_setter):
    """
    Create !truth.txt and !prediction.txt files in the corpus directory.

    Here we assume that the corpus directory already exists.
    The pred_setter function must accept the classification dictionary,
    and must return a dictionary with the same keys and possibly changed values.
    """
    # Create a prediction dictionary
    pred_dict = pred_setter(truth_dict)
    # Compile the filepaths
    truth_filepath = os.path.join(CORPUS_DIR, TRUTH_FILENAME)
    pred_filepath = os.path.join(CORPUS_DIR, PREDICTION_FILANAME)
    # Save the same dictionary as both the !truth.txt and !prediction.txt
    save_classification_to_file(truth_dict, truth_filepath)
    save_classification_to_file(pred_dict, pred_filepath)

def invert_classes(orig_dict):
    """Return a dict with switched HAM_TAG and SPAM_TAG."""
    inv_dict = {}
    for email_filename, truth in orig_dict.items():
        inv_dict[email_filename] = \
            SPAM_TAG if truth==HAM_TAG else HAM_TAG
    return inv_dict

def spams_only(orig_dict):
    """Return a dict with values equal to SPAM_TAG only."""
    spam_dict = {}
    for email_fname in orig_dict:
        spam_dict[email_fname] = SPAM_TAG
    return spam_dict

def hams_only(orig_dict):
    """Return a dict with values equal to HAM_TAG only."""
    ham_dict = {}
    for email_fname in orig_dict:
        ham_dict[email_fname] = HAM_TAG
    return ham_dict

def n_FP_n_FN(orig_dict, n_fp=1, n_fn=2):
    """Return a dict with predifined numbers of FP and FN."""
    new_dict = {}
    fp = fn = 0
    for email_fname, true_class in orig_dict.items():
        if fp < n_fp and true_class == HAM_TAG:
            new_dict[email_fname] = SPAM_TAG
            fp += 1
        elif fn < n_fn and true_class == SPAM_TAG:
            new_dict[email_fname] = HAM_TAG
            fn += 1
        else:
            new_dict[email_fname] = true_class
    return new_dict         
    
if __name__=='__main__':
    unittest.main(exit=False)