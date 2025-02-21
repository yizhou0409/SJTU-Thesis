import json
import time

def evaluate(outputs):
    """
    Evaluates the patchscope outputs and computes accuracy metrics.

    Args:
    - outputs: List of dictionaries from patchscope, containing idx, question, gt, unpatched_pred, and patched_pred.
    - metrics_path: Path to save evaluation metrics.

    Returns:
    - metrics: Dictionary containing evaluation results.
    """

    total_samples = len(outputs)
    correct_unpatched = sum(1 for sample in outputs if sample["unpatched_pred"].strip() == sample["gt"].strip())
    correct_patched = sum(1 for sample in outputs if sample["patched_pred"] and sample["patched_pred"].strip() == sample["gt"].strip())

    patching_attempts = sum(1 for sample in outputs if sample["unpatched_pred"].strip() == sample["gt"].strip())

    accuracy_unpatched = correct_unpatched / total_samples if total_samples > 0 else 0
    accuracy_patched = (correct_patched / patching_attempts) if patching_attempts > 0 else 0

    metrics = {
        "total_samples": total_samples,
        "correctly_predicted": correct_unpatched,
        "correctly_predicted_after_patching": correctly_patched
        "accuracy_unpatched": accuracy_unpatched,
        "accuracy_patched": accuracy_patched,
    }
    
    return metrics