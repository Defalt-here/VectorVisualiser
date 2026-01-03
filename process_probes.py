import json
import csv
import numpy as np
from scipy.stats import spearmanr

# ===============================
# Load embeddings (RAW)
# ===============================
data = np.load("glove_full.npz", allow_pickle=True)
words = data["words"]
vectors = data["vectors"]  # <-- RAW, NOT normalized

word_to_index = {w: i for i, w in enumerate(words)}
DIM = vectors.shape[1]

print("Embedding shape:", vectors.shape)

# ===============================
# GLOBAL SANITY CHECKS
# ===============================
if "man" in word_to_index and "woman" in word_to_index:
    man = vectors[word_to_index["man"]]
    woman = vectors[word_to_index["woman"]]
    cos_mw = np.dot(man, woman) / (np.linalg.norm(man) * np.linalg.norm(woman))
    print("cos(man, woman) =", cos_mw)

# ===============================
# Utilities
# ===============================
def normalize(v):
    n = np.linalg.norm(v)
    return v if n == 0 else v / n

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def entropy(scores, eps=1e-12):
    s = np.array(scores)
    s = s - s.min() + eps
    p = s / s.sum()
    return -np.sum(p * np.log2(p))

# ===============================
# Expression evaluation
# ===============================
def evaluate_expression(expr):
    tokens = expr.lower().split()
    result = np.zeros(DIM)
    used_words = set()
    sign = +1

    for t in tokens:
        if t == "+":
            sign = +1
        elif t == "-":
            sign = -1
        else:
            if t not in word_to_index:
                return None, None, t
            result += sign * vectors[word_to_index[t]]
            used_words.add(t)
            sign = +1

    return normalize(result), used_words, None

def top_k_neighbors(result_vec, used_words, k=5):
    sims = np.dot(vectors, result_vec)
    for w in used_words:
        sims[word_to_index[w]] = -np.inf
    idx = np.argsort(sims)[-k:][::-1]
    return [(words[i], float(sims[i])) for i in idx]

def run_probes(probe_json, output_csv, embedding_name="glove_300d"):
    with open(probe_json, "r", encoding="utf-8") as f:
        probes = json.load(f)["probes"]

    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        # CSV header
        writer.writerow([
            "probe_id",
            "domain",
            "embedding_model",
            "expression",
            "probe_type",
            "sym_vector_distance",
            "sym_overlap_top5",
            "sym_rank_correlation",
            "sym_entropy_diff"
        ])

        for p in probes:
            expr = p["expression"]
            probe_type = p["type"].strip().lower()

            # Default values
            sym_dist = sym_overlap = sym_rank = sym_entropy = 0.0
            print(
                    f"[CHECK] id={p['id']} | type={repr(p.get('type'))} | "
                    f"normalized={repr(probe_type)} | "
                    f"has_pair={'paired_expression' in p}"
                )
            if probe_type == "symmetry":
                sym_dist, sym_overlap, sym_rank, sym_entropy = symmetry_metrics(
                    p["expression"],
                    p["paired_expression"]
                )

            writer.writerow([
                p["id"],
                p["domain"],
                embedding_name,
                expr,
                probe_type,
                sym_dist,
                sym_overlap,
                sym_rank,
                sym_entropy
            ])


# ===============================
# SYMMETRY DIAGNOSTIC (KEY)
# ===============================

def symmetry_metrics(expr_a, expr_b):
    v1, used1, _ = evaluate_expression(expr_a)
    v2, used2, _ = evaluate_expression(expr_b)

    print("\nSYMMETRY TEST")
    print(expr_a, "<->", expr_b)

    print("v1 == v2 ?", np.allclose(v1, v2))
    print("L2 distance:", np.linalg.norm(v1 - v2))

    n1 = top_k_neighbors(v1, used1, 5)
    n2 = top_k_neighbors(v2, used2, 5)

    print("Neighbors A:", n1)
    print("Neighbors B:", n2)

    w1 = [w for w, _ in n1]
    w2 = [w for w, _ in n2]

    overlap = len(set(w1) & set(w2)) / 5.0

    common = list(set(w1) & set(w2))
    if len(common) >= 2:
        r1 = [w1.index(w) for w in common]
        r2 = [w2.index(w) for w in common]
        rank_corr = spearmanr(r1, r2).correlation
    else:
        rank_corr = 0.0

    e1 = entropy([s for _, s in n1])
    e2 = entropy([s for _, s in n2])

    return np.linalg.norm(v1 - v2), overlap, rank_corr, abs(e1 - e2)

# ===============================
# MANUAL TEST (RUN THIS FIRST)
# ===============================
if __name__ == "__main__":
    run_probes( probe_json="probes.json", output_csv="probe_results.csv", embedding_name="glove_50d" )