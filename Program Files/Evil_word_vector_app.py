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

    base_word = parsed[0][0]

    for word, sign in parsed:
        if word not in word_to_index:
            print(f"Unknown word: {word}")
            return None, None, None

        vec = vectors[word_to_index[word]]
        result += sign * vec
        used_words.add(word)

    return result, used_words, base_word


# ---------- Biased Similarity Search ----------
def find_top_matches(
    result_vector,
    used_words,
    base_word,
    top_n=5,
    candidate_pool=10,        # smaller pool = more bias
    penalty_alpha=0.1,        # weak echo suppression
    role_beta=0.9,            # strong role anchoring
    magnitude_gamma=0.7       # frequency dominance
):
    # Raw similarity
    base_sims = np.dot(vectors, result_vector)

    # Magnitude bias
    magnitudes = np.linalg.norm(vectors, axis=1)
    base_sims = base_sims * (1 + magnitude_gamma * magnitudes)

    # Exclude input words
    for w in used_words:
        base_sims[word_to_index[w]] = -np.inf

    candidate_idx = np.argsort(base_sims)[-candidate_pool:]

    base_vec = vectors[word_to_index[base_word]]
    reranked = []

    for i in candidate_idx:
        w_vec = vectors[i]

        sim_result = base_sims[i]

        # Penalise similarity to input words
        input_sims = [
            np.dot(w_vec, vectors[word_to_index[w]])
            for w in used_words
        ]
        mean_input_sim = np.mean(input_sims)

        # Role preservation
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
    print("Evil Word Vector Arithmetic Calculator")
    print("Example: leader - man + woman\n")

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
