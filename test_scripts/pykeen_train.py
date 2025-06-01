# train.py
import os
import random
import torch
import numpy as np

from pykeen.pipeline import pipeline
from pykeen.datasets.nations import Nations
from pykeen.triples import CoreTriplesFactory

def main():
    # 1. Train ConvE with inverse triples
    result = pipeline(
        dataset="Nations",
        dataset_kwargs={"create_inverse_triples": True},
        model="ConvE",
        training_kwargs=dict(
            num_epochs=50,
            batch_size=128,
        ),
        device="cpu",
        random_seed=42,
    )

    # 2. Load validation with inverse triples
    dataset = Nations(create_inverse_triples=True)
    val_tf = dataset.validation
    pos = val_tf.triples

    # 3. Generate balanced negatives (1:1 ratio)
    entities = list(val_tf.entity_to_id.keys())
    neg = []
    for h, r, t in pos:
        while True:
            t_corr = random.choice(entities)
            if t_corr != t:
                neg.append([h, r, t_corr])
                break
    neg = np.array(neg, dtype=str)

    # 4. Convert to ID tensors
    def to_ids(triples):
        return torch.tensor(
            [[
                val_tf.entity_to_id[h],
                val_tf.relation_to_id[r],
                val_tf.entity_to_id[t],
            ] for h, r, t in triples],
            dtype=torch.long,
        )

    hrt_pos = to_ids(pos)
    hrt_neg = to_ids(neg)

    # 5. Score validation triples
    model = result.model
    model.eval()
    with torch.no_grad():
        scores_pos = model.score_hrt(hrt_pos).squeeze(-1).cpu().numpy()
        scores_neg = model.score_hrt(hrt_neg).squeeze(-1).cpu().numpy()

    # 6a. Youden's J thresholding (TPR - FPR) for balance
    all_scores = np.unique(np.concatenate([scores_pos, scores_neg]))
    best_tau_j, best_j = None, -1.0
    for τ in all_scores:
        tp = (scores_pos >= τ).sum()
        fn = (scores_pos < τ).sum()
        fp = (scores_neg >= τ).sum()
        tn = (scores_neg < τ).sum()
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        j = tpr - fpr
        if j > best_j:
            best_j, best_tau_j = j, float(τ)
    print(f"Optimal τ (Youden's J)= {best_tau_j:.4f}, J={best_j:.3f}")

    # 6b. Net gain thresholding: maximize (TP - FP)
    best_tau_net, best_net = None, -np.inf
    for τ in all_scores:
        tp = (scores_pos >= τ).sum()
        fp = (scores_neg >= τ).sum()
        net = tp - fp
        if net > best_net:
            best_net, best_tau_net = net, float(τ)
    print(f"Optimal τ (TP-FP Max)= {best_tau_net:.4f}, Net Gain={best_net}")

    # 6c. TP-optimal thresholding: maximize true positives (recall)
    # Set τ to minimum positive score so TP = total positives
    best_tau_tp = float(scores_pos.min())
    tp_max = (scores_pos >= best_tau_tp).sum()
    print(f"Optimal τ (Maximize TP)= {best_tau_tp:.4f}, TP={tp_max}\n")

    # 7. Evaluate at TP-FP-optimal τ
    tp = int((scores_pos >= best_tau_net).sum())
    fn = int((scores_pos <  best_tau_net).sum())
    fp = int((scores_neg >= best_tau_net).sum())
    tn = int((scores_neg <  best_tau_net).sum())
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    print("Confusion at TP-FP-optimal τ:")
    print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"Accuracy={accuracy:.3f}, Precision={precision:.3f}, Recall={recall:.3f}\n")

    # 8. Save artifacts
    out_dir = "../trained_model"
    os.makedirs(out_dir, exist_ok=True)
    result.save_to_directory(out_dir)
    with open(os.path.join(out_dir, "threshold_youden.txt"), "w") as fh:
        fh.write(f"{best_tau_j:.8f}\n")
    with open(os.path.join(out_dir, "threshold_net.txt"), "w") as fh:
        fh.write(f"{best_tau_net:.8f}\n")
    with open(os.path.join(out_dir, "threshold_tp.txt"), "w") as fh:
        fh.write(f"{best_tau_tp:.8f}\n")
    val_path = os.path.join(out_dir, "validation_triples_inverse.pt")
    CoreTriplesFactory.to_path_binary(val_tf, val_path)

    print(f"Training complete. Artifacts saved to {out_dir}/")

if __name__ == "__main__":
    main()

