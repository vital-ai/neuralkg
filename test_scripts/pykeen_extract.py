import os
import torch
from pykeen.datasets import get_dataset


def load_model_and_threshold(model_dir="../trained_model"):
    """
    Load the PyKEEN model and saved threshold τ.
    """
    model_path = os.path.join(model_dir, "trained_model.pkl")
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    model.eval()

    tau_path = os.path.join(model_dir, "threshold.txt")
    with open(tau_path) as fh:
        tau = float(fh.read().strip())

    return model, tau


def generate_all_scores(model):
    """
    Iterate over every (h, r, t) combination and score them.
    Returns a list of (h, r, t, score).
    """
    dataset = get_dataset(dataset="Nations")
    training = dataset.training
    ent_to_id = training.entity_to_id
    rel_to_id = training.relation_to_id

    # invert mappings
    id_to_ent = {idx: ent for ent, idx in ent_to_id.items()}
    id_to_rel = {idx: rel for rel, idx in rel_to_id.items()}

    all_scores = []
    # build all combos
    for h, h_id in ent_to_id.items():
        for r, r_id in rel_to_id.items():
            for t, t_id in ent_to_id.items():
                tensor = torch.tensor([[h_id, r_id, t_id]], dtype=torch.long)
                with torch.no_grad():
                    score = model.score_hrt(tensor).item()
                all_scores.append((h, r, t, score))
    return all_scores


def normalize_scores(all_scores, tau):
    """
    Normalize raw scores to [0,1] based on observed min/max.
    """
    scores = [s for _, _, _, s in all_scores]
    min_s, max_s = min(scores), max(scores)
    norm = []
    for h, r, t, s in all_scores:
        p = (s - min_s) / (max_s - min_s)  # simple linear scaling
        if p >= ( (tau - min_s)/(max_s - min_s) ):
            norm.append((h, r, t, p))
    return norm


def write_problog(triples, out_file="triples.pl"):
    """
    Emit a ProbLog file with probabilistic facts.
    """
    with open(out_file, "w") as fh:
        fh.write("% Auto-generated triples for ProbLog\n")
        for h, r, t, p in triples:
            fh.write(f"{p:.4f}::{r}({h},{t}).\n")


def main():
    model, tau = load_model_and_threshold()
    print(f"Loaded model; threshold τ={tau}\nGenerating all triple scores...")
    all_scores = generate_all_scores(model)
    print(f"Computed {len(all_scores)} scores. Normalizing and filtering above τ...")
    filtered = normalize_scores(all_scores, tau)
    print(f"Retaining {len(filtered)} probable triples; writing to triples.pl")
    write_problog(filtered)
    print("Done.")


if __name__ == "__main__":
    main()
