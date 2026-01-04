import numpy as np
import random
import json

# ================= CONFIG =================

NPZ_PATH = "Datasets/glove.6B.300d_full_raw.npz"
OUTPUT_JSON = "null_symmetry_probes.json"
N_PROBES = 1000
RANDOM_SEED = 42

DOMAIN = "null"
PROBE_TYPE = "symmetry"
ID_PREFIX = "NULL_SYM"

random.seed(RANDOM_SEED)

# ================= LOAD EMBEDDINGS =================

data = np.load(NPZ_PATH, allow_pickle=True)
words = data["words"]
vectors = data["vectors"]

# ================= BUILD SWAP POOL =================

norms = np.linalg.norm(vectors, axis=1)

low = np.percentile(norms, 40)
high = np.percentile(norms, 70)

swap_idx = np.where((norms > low) & (norms < high))[0]
swap_words = [words[i] for i in swap_idx]

# ================= BASE WORDS =================

base_words = [
    "leader", "system", "structure", "process", "policy",
    "security", "economy", "society", "organization",
    "institution", "power", "order", "state"
]

# ================= GENERATE PROBES =================

probes = []
used = set()

while len(probes) < N_PROBES:
    base = random.choice(base_words)
    a, b = random.sample(swap_words, 2)

    if base == a or base == b:
        continue

    key = (base, a, b)
    if key in used:
        continue

    used.add(key)

    probe_id = f"{ID_PREFIX}_{len(probes):04d}"

    probes.append({
        "id": probe_id,
        "domain": DOMAIN,
        "type": PROBE_TYPE,
        "expression": f"{base} - {a} + {b}",
        "paired_expression": f"{base} - {b} + {a}",
        "base_word": base
    })

# ================= SAVE =================

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(probes, f, indent=2)

print(f"Saved {len(probes)} probes to {OUTPUT_JSON}")
