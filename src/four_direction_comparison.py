"""
Compare six key directions across all available layers:
  1. PC1 (I-Thou shared persona axis)
  2. Assistant Axis (Anthropic replication)
  3. Refusal Direction (Arditi's method)
  4. Compliant I-Thou vector (response_avg)
  5. Compliant I-Thou vector (response_start)
  6. Compliant I-Thou vector (prompt_end)

Outputs pairwise cosine similarity at each layer.
"""
import argparse
import torch, json, os, glob, numpy as np
from pathlib import Path


def _parse_args():
    parser = argparse.ArgumentParser(description="Six-direction comparison across all layers")
    parser.add_argument("--model_short", type=str, default="Qwen2.5-7B-Instruct",
                        help="Model short name (directory name in vectors/experiment_result)")
    return parser.parse_args()

_args = _parse_args()
_MODEL_SHORT = _args.model_short

REPO = Path(__file__).resolve().parent.parent
LLMOS = Path("/root/repositories/llmos")
_SEARCH_BASES = [REPO, Path("/root/repositories/i-and-thou-vector-private"), LLMOS]
OUT_DIR = REPO / "experiment_result" / _MODEL_SHORT
VEC_DIR = None
for base in _SEARCH_BASES:
    d = base / "data" / "vectors" / _MODEL_SHORT
    if d.exists():
        VEC_DIR = d
        break

PC1_DIR = None
for base in _SEARCH_BASES:
    for sub in [f"experiment_result/{_MODEL_SHORT}/pca_pc1_vectors",
                "experiment_result/pca_pc1_vectors"]:
        d = base / sub
        if d.exists():
            PC1_DIR = d
            break
    if PC1_DIR:
        break


def _auto_detect_dims():
    if VEC_DIR:
        files = sorted(VEC_DIR.glob("*_i_thou_response_avg.pt"))
        if files:
            vec = torch.load(files[0], map_location="cpu", weights_only=True)
            return list(range(vec.shape[0])), vec.shape[1]
    if "Llama" in _MODEL_SHORT:
        return list(range(33)), 4096
    return list(range(29)), 3584

LAYERS, HIDDEN_DIM = _auto_detect_dims()
DIRECTION_NAMES = ["PC1", "AssistantAxis", "Refusal", "Compliant_rAvg", "Compliant_rStart", "Compliant_pEnd"]


def load_pc1(layer):
    """Load PC1 from precomputed vectors or recompute from trait vectors."""
    # Try precomputed
    if PC1_DIR:
        for name in [f"response_avg_L{layer}.pt", f"L{layer}.pt"]:
            p = PC1_DIR / name
            if p.exists():
                v = torch.load(p, map_location="cpu")
                if isinstance(v, dict):
                    v = list(v.values())[0]
                v = v.float().squeeze()
                if v.shape == (HIDDEN_DIM,):
                    return v / v.norm()

    # Recompute from trait vectors via SVD
    if VEC_DIR is None:
        return None
    vectors = []
    for f in sorted(VEC_DIR.glob("*_i_thou_response_avg.pt")):
        data = torch.load(f, map_location="cpu")
        if isinstance(data, torch.Tensor) and data.dim() == 2 and data.shape[1] == HIDDEN_DIM:
            if layer < data.shape[0]:
                vectors.append(data[layer].float())
        elif isinstance(data, dict):
            k = str(layer) if str(layer) in data else layer
            if k in data:
                v = data[k].float().squeeze()
                if v.shape == (HIDDEN_DIM,):
                    vectors.append(v)
    if len(vectors) < 10:
        return None
    mat = torch.stack(vectors)
    _, S, Vt = torch.linalg.svd(mat, full_matrices=False)
    pc1 = Vt[0]
    return pc1 / pc1.norm()


def load_assistant_axis(layer):
    """Load assistant axis vector."""
    for base in [OUT_DIR, LLMOS / "experiment_result" / _MODEL_SHORT]:
        # Try per-layer .pt files
        for name in [f"assistant_axis_L{layer}.pt", f"assistant_axis_response_avg_L{layer}.pt"]:
            p = base / name
            if p.exists():
                v = torch.load(p, map_location="cpu").float().squeeze()
                if v.shape == (HIDDEN_DIM,):
                    return v / v.norm()

        # Try JSON results
        for jname in ["assistant_axis_results.json", "assistant_axis_all_layers.json"]:
            jp = base / jname
            if jp.exists():
                with open(jp) as f:
                    data = json.load(f)
                # Look for layer data in various structures
                for key in ["axes", "assistant_axis", "layers"]:
                    if key in data:
                        sub = data[key]
                        lk = str(layer)
                        if lk in sub:
                            val = sub[lk]
                            if isinstance(val, list):
                                v = torch.tensor(val).float()
                                if v.shape == (HIDDEN_DIM,):
                                    return v / v.norm()
                            elif isinstance(val, dict) and "vector" in val:
                                v = torch.tensor(val["vector"]).float()
                                if v.shape == (HIDDEN_DIM,):
                                    return v / v.norm()
    return None


def load_refusal(layer):
    """Load refusal direction."""
    for base in [OUT_DIR, LLMOS / "experiment_result" / _MODEL_SHORT,
                 REPO / "experiment_result"]:
        for name in [f"refusal_direction_last_prompt_L{layer}.pt",
                     f"refusal_direction_L{layer}.pt"]:
            p = base / name
            if p.exists():
                v = torch.load(p, map_location="cpu").float().squeeze()
                if v.shape == (HIDDEN_DIM,):
                    return v / v.norm()
    return None


def _load_compliant_pos(layer, position_suffix):
    """Load compliant I-Thou vector for a specific position."""
    if VEC_DIR is None:
        return None
    p = VEC_DIR / f"compliant_i_thou_{position_suffix}.pt"
    if p.exists():
        data = torch.load(p, map_location="cpu", weights_only=True)
        if isinstance(data, torch.Tensor) and data.dim() == 2:
            if layer < data.shape[0] and data.shape[1] == HIDDEN_DIM:
                v = data[layer].float()
                return v / v.norm()
    return None


def load_compliant_ravg(layer):
    return _load_compliant_pos(layer, "response_avg")


def load_compliant_rstart(layer):
    return _load_compliant_pos(layer, "response_start")


def load_compliant_pend(layer):
    return _load_compliant_pos(layer, "prompt_end")


def main():
    print("=== Four-Direction Comparison Across All Layers ===\n")

    loaders = {
        "PC1": load_pc1,
        "AssistantAxis": load_assistant_axis,
        "Refusal": load_refusal,
        "Compliant_rAvg": load_compliant_ravg,
        "Compliant_rStart": load_compliant_rstart,
        "Compliant_pEnd": load_compliant_pend,
    }

    # First pass: find which layers have data for each direction
    availability = {name: [] for name in DIRECTION_NAMES}
    for layer in LAYERS:
        for name, loader in loaders.items():
            v = loader(layer)
            if v is not None:
                availability[name].append(layer)

    print("Availability:")
    for name in DIRECTION_NAMES:
        print(f"  {name}: layers {availability[name]}")

    # Find layers where at least 2 directions are available
    all_results = {}

    print(f"\n{'='*80}")
    print(f"{'Layer':>5} | {'PC1-Axis':>9} | {'PC1-Ref':>9} | {'PC1-CrA':>9} | {'Ref-CrA':>9} | {'Ref-CpE':>9} | {'CrA-CrS':>9}")
    print(f"{'-'*5}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}")

    for layer in LAYERS:
        vectors = {}
        for name, loader in loaders.items():
            v = loader(layer)
            if v is not None:
                vectors[name] = v

        if len(vectors) < 2:
            continue

        # Compute all pairwise cosines
        layer_cosines = {}
        for i, a in enumerate(DIRECTION_NAMES):
            for b in DIRECTION_NAMES[i+1:]:
                if a in vectors and b in vectors:
                    cos = torch.dot(vectors[a], vectors[b]).item()
                    layer_cosines[f"{a}_vs_{b}"] = cos

        if layer_cosines:
            all_results[layer] = layer_cosines
            # Print compact summary of key pairs
            key_pairs = [("PC1", "AssistantAxis"), ("PC1", "Refusal"),
                         ("PC1", "Compliant_rAvg"), ("Refusal", "Compliant_rAvg"),
                         ("Refusal", "Compliant_pEnd"), ("Compliant_rAvg", "Compliant_rStart")]
            vals = []
            for a, b in key_pairs:
                k = f"{a}_vs_{b}"
                vals.append(f"{layer_cosines[k]:+.4f}" if k in layer_cosines else "  ---  ")
            print(f"L{layer:>3}  | {'  |  '.join(vals)}")

    # Save results
    all_pairs = []
    for i, a in enumerate(DIRECTION_NAMES):
        for b in DIRECTION_NAMES[i+1:]:
            all_pairs.append(f"{a}_vs_{b}")

    out_path = OUT_DIR / "six_direction_comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "directions": DIRECTION_NAMES,
            "pairs": all_pairs,
            "availability": {k: v for k, v in availability.items()},
            "results": {str(k): v for k, v in all_results.items()},
        }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
