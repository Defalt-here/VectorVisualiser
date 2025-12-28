import numpy as np
import random

# Load the saved vectors
data = np.load('trimmed.npz', allow_pickle=True)
words = data['words']
vectors = data['vectors']

# Print vector shape
print(f"Vector shape: {vectors.shape}")

# Print norm of a random vector
idx = random.randint(0, len(vectors) - 1)
norm = np.linalg.norm(vectors[idx])
print(f"Norm of random vector (index {idx}): {norm:.6f}")

# Confirm common words exist
common_words = ['king', 'man', 'woman', 'queen']
for word in common_words:
    if word in words:
        print(f"'{word}' exists in vocabulary")
    else:
        print(f"'{word}' does not exist in vocabulary")

# Check if 'king' is missing
if 'king' not in words:
    print("ERROR: 'king' is missing - your vocab selection is wrong.")
else:
    print("Vocab selection appears correct ('king' is present).")

# Find closest word to a random vector
print("\n--- Random Vector Test ---")
# Generate random vector (values between -1 and 1)
random_vec = np.random.uniform(-1, 1, vectors.shape[1])
random_vec = random_vec / np.linalg.norm(random_vec)

# Compute cosine similarities (dot product since all are normalized)
similarities = np.dot(vectors, random_vec)
closest_idx = np.argmax(similarities)
closest_word = words[closest_idx]
similarity = similarities[closest_idx]

print(f"Random vector is closest to: '{closest_word}' with similarity {similarity:.4f}")