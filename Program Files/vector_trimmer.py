import numpy as np
import os


def _prompt_choice(prompt, choices, default=None):
    choices_lc = {c.lower(): c for c in choices}
    suffix = f" [{'/'.join(choices)}]" + (f" (default: {default})" if default else "")
    while True:
        raw = input(f"{prompt}{suffix}: ").strip().lower()
        if not raw and default is not None:
            raw = default.lower()
        if raw in choices_lc:
            return choices_lc[raw]
        print(f"Please enter one of: {', '.join(choices)}")


def _prompt_int(prompt, min_value=1, default=None):
    while True:
        raw = input(
            f"{prompt}" + (f" (default: {default})" if default is not None else "") + ": "
        ).strip()
        if not raw and default is not None:
            return default
        try:
            value = int(raw)
        except ValueError:
            print("Please enter an integer.")
            continue
        if value < min_value:
            print(f"Please enter a value >= {min_value}.")
            continue
        return value

def normalize_vector(vec):
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def _iter_text_embeddings(file_path, input_type):
    """Yields (word, vector_list[float]) from a .vec/.txt style embedding file.

    - For input_type == 'vec', supports (optional) fastText header: '<n_words> <dim>'.
    - For input_type == 'txt', reads all lines as 'word val1 val2 ...'.
    """
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        first_line = True
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            # fastText .vec header: two integers
            if first_line and input_type == "vec" and len(parts) == 2:
                if parts[0].isdigit() and parts[1].isdigit():
                    first_line = False
                    continue

            first_line = False

            word = parts[0]
            try:
                vec = [float(x) for x in parts[1:]]
            except ValueError:
                continue
            yield word, vec

def process_vectors(file_path, input_type, normalize, trim_size, output_format):
    words = []
    vectors = []

    expected_dim = None
    skipped_dim_mismatch = 0

    for word, vec_list in _iter_text_embeddings(file_path, input_type):
        if expected_dim is None:
            expected_dim = len(vec_list)
            if expected_dim == 0:
                continue

        if len(vec_list) != expected_dim:
            skipped_dim_mismatch += 1
            continue

        vec = np.asarray(vec_list, dtype=np.float32)
        if normalize:
            vec = normalize_vector(vec)
        words.append(word)
        vectors.append(vec)

        if trim_size and len(words) >= trim_size:
            break

    if not vectors:
        raise RuntimeError("No vectors were loaded. Check file type/path and file format.")

    vectors = np.stack(vectors, axis=0)

    stem = os.path.splitext(os.path.basename(file_path))[0]
    size_tag = "full" if not trim_size else f"trim{trim_size}"
    norm_tag = "norm" if normalize else "raw"

    if output_format == '.npz':
        output_name = f"{stem}_{size_tag}_{norm_tag}.npz"
        np.savez(output_name, words=np.array(words, dtype=object), vectors=vectors)
    elif output_format == '.vec':
        output_name = f"{stem}_{size_tag}_{norm_tag}.vec"
        with open(output_name, 'w', encoding='utf-8') as f:
            f.write(f"{len(words)} {vectors.shape[1]}\n")
            for word, vector in zip(words, vectors):
                f.write(f"{word} {' '.join(f'{x:.6g}' for x in vector.tolist())}\n")
    else:
        print("Unsupported output format.")
        return

    if skipped_dim_mismatch:
        print(f"Note: skipped {skipped_dim_mismatch} lines due to vector length mismatch.")
    print(f"Saved {len(words)} words and vectors to {output_name}")

def main():
    print("Vector Processor CLI")

    input_type = _prompt_choice("Input file type", ["vec", "txt"], default="vec")

    file_path = input("Enter the input file path: ").strip().strip('"')
    if not os.path.exists(file_path):
        print("File not found.")
        return

    normalize = _prompt_choice("Normalize vectors?", ["yes", "no"], default="yes") == "yes"

    mode = _prompt_choice("Process full file or trim?", ["full", "trim"], default="trim")
    trim_size = None
    if mode == "trim":
        trim_size = _prompt_int("How many vectors to keep", min_value=1, default=5000)

    output_format = _prompt_choice("Output format", [".npz", ".vec"], default=".npz")

    try:
        process_vectors(file_path, input_type, normalize, trim_size, output_format)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()