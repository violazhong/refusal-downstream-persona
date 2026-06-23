"""
Analyze PC1-PC6 overlap with assistant axis.
Compute cosine similarity between each PC and the assistant axis from E6.
"""
import torch, json, os, glob, numpy as np
from pathlib import Path

REPO = Path("/root/repositories/i-and-thou-vector-private")
LLMOS_REPO = Path("/root/repositories/llmos")
OUT_DIR = REPO / "experiment_result" / "Qwen2.5-7B-Instruct"

# Load 88 I-Thou vectors
def load_ithou_vectors(layer, position="response_avg"):
    """Load all 88 I-Thou trait vectors at a given layer/position."""
    vec_dir = REPO / "data" / "vectors" / "Qwen2.5-7B-Instruct"
    if not vec_dir.exists():
        vec_dir = LLMOS_REPO / "data" / "vectors" / "Qwen2.5-7B-Instruct"

    vectors = []
    trait_names = []

    pattern = str(vec_dir / f"*_model_persona_{position}.pt")
    files = sorted(glob.glob(pattern))

    if not files:
        # Try alternative naming
        pattern = str(vec_dir / f"*_{position}.pt")
        files = sorted(glob.glob(pattern))

    for f in files:
        basename = os.path.basename(f)
        trait = basename.replace(f"_model_persona_{position}.pt", "").replace(f"_{position}.pt", "")
        data = torch.load(f, map_location="cpu")

        if isinstance(data, dict):
            if str(layer) in data:
                v = data[str(layer)]
            elif layer in data:
                v = data[layer]
            elif "vectors" in data:
                v = data["vectors"]
                if isinstance(v, dict):
                    v = v.get(str(layer), v.get(layer))
            else:
                continue
        elif isinstance(data, torch.Tensor):
            if data.dim() == 2:
                v = data[layer]
            else:
                v = data
        else:
            continue

        v = v.float().squeeze()
        if v.dim() == 1 and v.shape[0] == 3584:
            vectors.append(v)
            trait_names.append(trait)

    return torch.stack(vectors), trait_names

def load_assistant_axis(layer):
    """Load assistant axis from E6 results."""
    results_path = OUT_DIR / "assistant_axis_results.json"
    if not results_path.exists():
        results_path = LLMOS_REPO / "experiment_result" / "Qwen2.5-7B-Instruct" / "assistant_axis_results.json"

    with open(results_path) as f:
        data = json.load(f)

    # The axis should be stored in the results
    layer_key = str(layer)
    if "axes" in data and layer_key in data["axes"]:
        axis = torch.tensor(data["axes"][layer_key])
    elif "assistant_axis" in data:
        ax_data = data["assistant_axis"]
        if layer_key in ax_data:
            axis = torch.tensor(ax_data[layer_key])
        else:
            return None
    else:
        # Try loading from .pt file
        for base in [OUT_DIR, LLMOS_REPO / "experiment_result" / "Qwen2.5-7B-Instruct"]:
            pt_path = base / f"assistant_axis_L{layer}.pt"
            if pt_path.exists():
                axis = torch.load(pt_path, map_location="cpu").float().squeeze()
                return axis / axis.norm()
        return None

    axis = axis.float().squeeze()
    return axis / axis.norm()

def load_refusal_direction():
    """Load refusal direction at L10."""
    for base in [OUT_DIR, LLMOS_REPO / "experiment_result" / "Qwen2.5-7B-Instruct"]:
        path = base / "refusal_direction_last_prompt_L10.pt"
        if path.exists():
            v = torch.load(path, map_location="cpu").float().squeeze()
            return v / v.norm()
    return None

def main():
    LAYERS = [10, 15, 20, 21, 25]
    N_PCS = 6

    print("=== PC1-PC6 vs Assistant Axis Analysis ===\n")

    # First check what vector files exist
    for base in [REPO, LLMOS_REPO]:
        vec_dir = base / "data" / "vectors" / "Qwen2.5-7B-Instruct"
        if vec_dir.exists():
            files = list(vec_dir.glob("*.pt"))
            print(f"Found {len(files)} vector files in {vec_dir}")
            if files:
                print(f"  Sample: {files[0].name}")
                # Check structure of first file
                d = torch.load(files[0], map_location="cpu")
                if isinstance(d, dict):
                    print(f"  Keys: {list(d.keys())[:10]}")
                elif isinstance(d, torch.Tensor):
                    print(f"  Shape: {d.shape}")
            break

    # Check assistant axis results
    for base in [OUT_DIR, LLMOS_REPO / "experiment_result" / "Qwen2.5-7B-Instruct"]:
        ax_path = base / "assistant_axis_results.json"
        if ax_path.exists():
            with open(ax_path) as f:
                ax_data = json.load(f)
            print(f"\nAssistant axis results found at {ax_path}")
            print(f"  Top-level keys: {list(ax_data.keys())[:10]}")
            if "layers" in ax_data:
                print(f"  Layers: {list(ax_data['layers'].keys())}")
            break

    # Try to load vectors and compute PCA
    for layer in LAYERS:
        print(f"\n{'='*50}")
        print(f"Layer {layer}")
        print(f"{'='*50}")

        try:
            vectors, traits = load_ithou_vectors(layer)
            print(f"  Loaded {len(traits)} trait vectors, shape: {vectors.shape}")
        except Exception as e:
            print(f"  Failed to load vectors: {e}")
            continue

        # SVD (uncentered, same as pca_shared_axis.py)
        U, S, Vt = torch.linalg.svd(vectors, full_matrices=False)
        total_var = (S ** 2).sum().item()

        pcs = Vt[:N_PCS]  # [6, 3584]
        explained = [(S[i] ** 2 / total_var).item() for i in range(N_PCS)]

        print(f"  Explained variance: {['PC{}: {:.1%}'.format(i+1, e) for i, e in enumerate(explained)]}")

        # Load assistant axis
        axis = load_assistant_axis(layer)
        if axis is not None:
            print(f"\n  Cosines with assistant axis:")
            for i in range(N_PCS):
                cos = torch.dot(pcs[i], axis).item()
                print(f"    PC{i+1}: cos = {cos:+.4f} (var = {explained[i]:.1%})")
        else:
            print(f"  Assistant axis not found for L{layer}")

        # Load refusal (only L10)
        if layer == 10:
            refusal = load_refusal_direction()
            if refusal is not None:
                print(f"\n  Cosines with refusal direction:")
                for i in range(N_PCS):
                    cos = torch.dot(pcs[i], refusal).item()
                    print(f"    PC{i+1}: cos = {cos:+.4f}")

    # Save PC vectors for steering
    print(f"\n{'='*50}")
    print("Saving PC1-PC6 vectors for potential steering...")
    print(f"{'='*50}")

    pc_vectors = {}
    for layer in [20, 21]:
        try:
            vectors, traits = load_ithou_vectors(layer)
            U, S, Vt = torch.linalg.svd(vectors, full_matrices=False)
            for i in range(N_PCS):
                key = f"response_avg_L{layer}_PC{i+1}"
                pc_vectors[key] = Vt[i]
                print(f"  Saved {key}")
        except Exception as e:
            print(f"  Failed for L{layer}: {e}")

    save_path = OUT_DIR / "pc1_to_pc6_vectors.pt"
    torch.save(pc_vectors, save_path)
    print(f"\nSaved to {save_path}")

if __name__ == "__main__":
    main()
