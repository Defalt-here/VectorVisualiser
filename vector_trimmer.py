import numpy as np
import sys

if len(sys.argv) < 2 or len(sys.argv) > 3:
    print("Usage: python vector_trimmer.py <input_file> [--full]")
    sys.exit(1)

input_file = sys.argv[1]
full_mode = len(sys.argv) == 3 and sys.argv[2] == '--full'

words = []
vectors = []

with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        word = parts[0]
        try:
            vec = np.array([float(x) for x in parts[1:]])
            # Normalize the vector
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            words.append(word)
            vectors.append(vec)
            if not full_mode and len(words) >= 5000:
                break
        except ValueError:
            continue

vectors = np.array(vectors)
output_name = 'glove_full.npz' if full_mode else 'glove_trimmed.npz'
np.savez(output_name, words=np.array(words, dtype=object), vectors=vectors)
print(f"Saved {len(words)} words and vectors to {output_name}")