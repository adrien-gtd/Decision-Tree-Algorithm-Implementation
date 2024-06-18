from typing import List
import numpy as np

def precision_recall(expected_results: List[bool], actual_results: List[bool]) -> (float, float):
    """Compute the precision and recall of a series of predictions

    Parameters
    ----------
        expected_results : List[bool]
            The true results, that is the results that the predictor
            should have find.
        actual_results : List[bool]
            The predicted results, that have to be evaluated.

    Returns
    -------
        float
            The precision of the predicted results.
        float
            The recall of the predicted results.
    """

    np_expected = np.array(expected_results)
    np_actual = np.array(actual_results)
    TP = np.sum(np.logical_and(np_expected, np_actual))
    FP = np.sum(np.logical_and(~np_expected, np_actual))
    FN = np.sum(np.logical_and(np_expected, ~np_actual))
    # TP = sum([1 for index in range(len(expected_results)) if expected_results[index] == True and actual_results[index] == True])
    # FP = sum([1 for index in range(len(expected_results)) if expected_results[index] == False and actual_results[index] == True])
    # FN = sum([1 for index in range(len(expected_results)) if expected_results[index] == True and actual_results[index] == False])
    if (TP + FP) == 0 or (TP + FN) != 0:
        recall = TP / (TP + FP)
        precision = TP / (TP + FN)
        return recall, precision
    return 0,0

def F1_score(expected_results: List[bool], actual_results: List[bool]) -> float:
    """Compute the F1-score of a series of predictions

    Parameters
    ----------
        expected_results : List[bool]
            The true results, that is the results that the predictor
            should have find.
        actual_results : List[bool]
            The predicted results, that have to be evaluated.

    Returns
    -------
        float
            The F1-score of the predicted results.
    """
    recall, precision = precision_recall(expected_results, actual_results)
    return (2 * recall * precision) / (recall + precision)
