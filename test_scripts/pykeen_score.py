# score.py
import os
import torch
from pykeen.datasets.nations import Nations


def load_model_and_threshold(model_dir="../trained_model", threshold_file="threshold_net.txt"):
    """
    Load the PyKEEN ConvE model trained with inverse triples and a selected threshold τ.
    """
    # Load model
    model_path = os.path.join(model_dir, "trained_model.pkl")
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    model.eval()

    # Load threshold
    path = os.path.join(model_dir, threshold_file)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Threshold file not found: {path}")
    with open(path) as fh:
        tau = float(fh.read().strip())

    return model, tau


def score_and_classify(model, tau, triples, ground_truth):
    """
    Score each triple, classify using τ, and compute metrics.
    """
    # Reload Nations with inverse triples for mapping
    dataset = Nations(create_inverse_triples=True)
    training = dataset.training

    # Map triples to IDs
    hrt = torch.tensor([
        [
            training.entity_to_id[h],
            training.relation_to_id[r],
            training.entity_to_id[t],
        ] for h, r, t in triples
    ], dtype=torch.long)

    # Score
    with torch.no_grad():
        scores = model.score_hrt(hrt).squeeze(-1).tolist()

    # Classify and tally
    tp = tn = fp = fn = 0
    for score, is_true in zip(scores, ground_truth):
        pred = score >= tau
        if pred and is_true:
            tp += 1
        elif pred and not is_true:
            fp += 1
        elif not pred and is_true:
            fn += 1
        else:
            tn += 1

    # Compute metrics
    total = tp + tn + fp + fn
    accuracy  = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }


def main():
    # Define test triples and true labels
    examples = [
        ("usa",    "treaties",         "uk"),
        ("china",  "embassy",          "ussr"),
        ("india",  "intergovorgs",     "netherlands"),
        ("brazil", "militaryalliance", "burma"),
        ("cuba",   "tourism3",         "egypt"),
        ("burma",  "exports3",         "netherlands"),
    ]
    labels = [True, True, True, False, False, False]

    # Evaluate for each threshold strategy
    strategies = [
        ("Youden's J", "threshold_youden.txt"),
        ("TP-FP Max", "threshold_net.txt"),
        ("Max TP",     "threshold_tp.txt"),
    ]

    for name, thr_file in strategies:
        model, tau = load_model_and_threshold(threshold_file=thr_file)
        metrics = score_and_classify(model, tau, examples, labels)

        print(f"\n=== Results for {name} (τ from {thr_file}) ===")
        print(f"Threshold τ = {tau:.4f}")
        print(f"TP={metrics['TP']}, FP={metrics['FP']}, FN={metrics['FN']}, TN={metrics['TN']}")
        print(f"Accuracy={metrics['Accuracy']:.3f}")
        print(f"Precision={metrics['Precision']:.3f}")
        print(f"Recall={metrics['Recall']:.3f}")
        print(f"F1 Score={metrics['F1']:.3f}")

if __name__ == "__main__":
    main()

