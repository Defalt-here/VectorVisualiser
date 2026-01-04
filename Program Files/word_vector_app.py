import numpy as np
import os

# ---------- Load embeddings ----------
npz_file = "Datasets/glove_full.npz"
data = np.load(npz_file, allow_pickle=True)
words = data["words"]
vectors = data["vectors"]

word_to_index = {word: i for i, word in enumerate(words)}
DIM = vectors.shape[1]


# ---------- Utilities ----------
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


# ---------- Expression Parsing ----------
def parse_expression(expr):
    """
    Parses expressions like:
    king - man + woman
    crown + man

    Returns list of (word, sign)
    """
    tokens = expr.lower().split()
    parsed = []
    sign = +1

    for token in tokens:
        if token == "+":
            sign = +1
        elif token == "-":
            sign = -1
        else:
            parsed.append((token, sign))
            sign = +1

    return parsed


# ---------- Expression Evaluation ----------
def evaluate_expression(expr):
    parsed = parse_expression(expr)

    result = np.zeros(DIM)
    used_words = set()

    base_word = parsed[0][0]  # FIRST word defines role

    for word, sign in parsed:
        if word not in word_to_index:
            print(f"Unknown word: {word}")
            return None, None, None

        vec = vectors[word_to_index[word]]
        result += sign * vec
        used_words.add(word)

    # Do not normalize here; keep raw vector arithmetic result.
    return result, used_words, base_word


# ---------- Reranked Similarity Search (WITH ROLE BIAS) ----------
def find_top_matches(
    result_vector,
    used_words,
    base_word,
    top_n=5,
    candidate_pool=40,
    penalty_alpha=0.3,
    role_beta=0.3,
):
    # Stage 1: cosine similarity to result
    base_sims = np.dot(vectors, result_vector)

    # Exclude input words
    for w in used_words:
        base_sims[word_to_index[w]] = -np.inf

    candidate_idx = np.argsort(base_sims)[-candidate_pool:]

    base_vec = vectors[word_to_index[base_word]]
    reranked = []

    for i in candidate_idx:
        w_vec = vectors[i]

        sim_result = base_sims[i]

        # Penalize similarity to input words
        input_sims = [
            np.dot(w_vec, vectors[word_to_index[w]])
            for w in used_words
        ]
        mean_input_sim = np.mean(input_sims)

        # Role preservation: stay close to base word
        role_sim = np.dot(w_vec, base_vec)

        final_score = (
            sim_result
            - penalty_alpha * mean_input_sim
            + role_beta * role_sim
        )

        reranked.append(
            (words[i], final_score, sim_result, role_sim)
        )

    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked[:top_n]


# ---------- Main CLI ----------
def main():
    print("Word Vector Arithmetic Calculator")
    print("Example: king - man + woman\n")

    expr = input("Expression: ").strip()

    result_vector, used_words, base_word = evaluate_expression(expr)
    if result_vector is None:
        return

    matches = find_top_matches(
        result_vector,
        used_words,
        base_word
    )

    print("\nTop 5 closest words:")
    for word, score, cosine, role_sim in matches:
        print(
            f"{word:15s} | score={score:.4f} | cos={cosine:.4f} | role={role_sim:.4f}"
        )


if __name__ == "__main__":
    main()
