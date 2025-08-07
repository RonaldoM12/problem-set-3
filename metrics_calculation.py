'''
PART 2: METRICS CALCULATION
- Tailor the code scaffolding below to calculate various metrics
- Write the functions below
    - Further info and hints are provided in the docstrings
    - These should return values when called by the main.py
'''

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd

def calculate_metrics(model_pred_df, genre_list, genre_true_counts, genre_tp_counts, genre_fp_counts):
    '''
    Calculate micro and macro metrics
    
    Args:
        model_pred_df (pd.DataFrame): DataFrame containing model predictions
        genre_list (list): List of unique genres
        genre_true_counts (dict): Dictionary of true genre counts
        genre_tp_counts (dict): Dictionary of true positive genre counts
        genre_fp_counts (dict): Dictionary of false positive genre counts
    
    Returns:
        tuple: Micro precision, recall, F1 score
        lists of macro precision, recall, and F1 scores
    
    Hint #1: 
    tp -> true positives
    fp -> false positives
    tn -> true negatives
    fn -> false negatives

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    Hint #2: Micro metrics are tuples, macro metrics are lists

    '''

    # Your code here
    total_tp = sum(genre_tp_counts.values())
    total_fp = sum(genre_fp_counts.values())
    total_fn = sum(genre_true_counts[genre] - genre_tp_counts.get(genre, 0) for genre in genre_list)

    micro_precision = total_tp / (total_tp + total_fp)
    micro_recall = total_tp / (total_tp + total_fn)
    micro_f1 = (2* micro_precision* micro_recall) / (micro_precision + micro_recall)

    precision_list = []
    recall_list = []
    f1_list = []

    for genre in genre_list:
        tp = genre_tp_counts.get(genre, 0)
        fp = genre_fp_counts.get(genre, 0)
        fn = genre_true_counts.get(genre, 0) - tp

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = (2 * precision * recall) / (precision + recall)

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    return (micro_precision, micro_recall, micro_f1), precision_list, recall_list, f1_list

    
def calculate_sklearn_metrics(model_pred_df, genre_list):
    '''
    Calculate metrics using sklearn's precision_recall_fscore_support.
    
    Args:
        model_pred_df (pd.DataFrame): DataFrame containing model predictions.
        genre_list (list): List of unique genres.
    
    Returns:
        tuple: Macro precision, recall, F1 score, and micro precision, recall, F1 score.
    
    Hint #1: You'll need these two lists
    pred_rows = []
    true_rows = []
    
    Hint #2: And a little later you'll need these two matrixes for sk-learn
    pred_matrix = pd.DataFrame(pred_rows)
    true_matrix = pd.DataFrame(true_rows)
    '''

    # Your code here

    pred_rows = model_pred_df['true_genre'].tolist()
    true_rows = model_pred_df['predicted_genre'].tolist()

    pred_matrix = pd.DataFrame(pred_rows, columns=["true_genre"])
    true_matrix = pd.DataFrame(true_rows, columns=["predicted_genre"])

    macro_pred, macro_rows, macro_f1 = precision_recall_fscore_support(
        true_rows, pred_rows, average="macro", labels= genre_list, zero_division=0
    )

    micro_pred, micro_rows, micro_f1 = precision_recall_fscore_support(
        true_rows, pred_rows, average="micro", labels= genre_list, zero_division=0
    )

    return (macro_pred, macro_rows, macro_f1), (micro_pred, micro_rows, micro_f1)
