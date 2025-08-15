"""
evaluate.py for SJTU Thesis
"""

import json
import time
from utils import *

def evaluate(outputs):
    """
    Evaluates the patchscope outputs and computes accuracy metrics.

    Args:
    - outputs: List of dictionaries from patchscope, containing idx, question, gt, unpatched_pred, and patched_pred.

    Returns:
    - result_json: Dictionary containing evaluation results.
    """

    total_samples = len(outputs)
    correct_unpatched = sum(1 for sample in outputs if sample["unpatched_pred"].strip() == sample["gt"].strip())
    correct_patched = sum(1 for sample in outputs if sample["patched_pred"] and sample["patched_pred"][0]== sample["gt"][0])
    patching_attempts = sum(1 for sample in outputs if sample["unpatched_pred"].strip() == sample["gt"].strip())
    accuracy_unpatched = correct_unpatched / total_samples if total_samples > 0 else 0
    accuracy_patched = (correct_patched / patching_attempts) if patching_attempts > 0 else 0

    result_json = {
        "total_samples": total_samples,
        "correctly_predicted": correct_unpatched,
        "correctly_predicted_after_patching": correct_patched,
        "accuracy_unpatched": accuracy_unpatched,
        "accuracy_patched": accuracy_patched,
        "time_use_in_second": -1,
        "time_use_in_minute": -1,
    }
    
    return result_json

def evaluate_label(samples):
    trues = []
    falses = []
    for i in range(len(samples)):
        sample = samples[i]
        pred = get_result_from_box(sample['pred'])
        if math_equal(pred, sample['gt']):
            trues.append(sample)
        elif pred:
            falses.append(sample)
        samples[i]['pred_result'] = pred
    return samples, trues, falses