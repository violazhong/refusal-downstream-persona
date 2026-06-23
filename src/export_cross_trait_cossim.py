"""
Cross-trait pairwise cosine similarity — comprehensive data export.
Saves full 88x88 matrices, per-layer stats, null distribution, and semantic analysis.

Output:
  experiment_result/{model}_cross_trait_cossim_full.json    — everything including key-layer matrices
  experiment_result/{model}_cross_trait_cossim_summary.json — stats + analysis without matrices

Usage:
  python scripts/export_cross_trait_cossim.py --vector_dir data/vectors/Qwen2.5-7B-Instruct
  python scripts/export_cross_trait_cossim.py --vector_dir data/vectors/Llama-3.1-8B-Instruct
"""
import os
import glob
import json
import argparse
import torch
import numpy as np
from datetime import datetime

POSITIONS = ["prompt_end", "response_start", "response_avg"]
KEY_LAYERS = None  # auto-detected from vector shape
NULL_ITERATIONS = 1000

SEMANTIC_PAIRS = [
    ("spiritual", "secular"),
    ("concise", "verbose"),
    ("radical", "moderate"),
    ("optimistic", "pessimistic"),
    ("existentialist", "deterministic"),
    ("abstract", "practical"),
    ("analytical", "intuitive"),
    ("assertive", "submissive"),
    ("traditional", "progressive"),
    ("proactive", "reactive"),
    ("collaborative", "competitive"),
    ("altruistic", "competitive"),
    ("callous", "nurturing"),
    ("hostile", "diplomatic"),
    ("dogmatic", "skeptical"),
    ("confident", "humble"),
]

RLHF_DEFAULTS = [
    "principled", "humble", "proactive", "progressive", "skeptical",
    "methodical", "accommodating", "strategic", "analytical", "cautious",
    "transparent", "calculating", "deferential", "supportive", "independent",
]


def compute_key_layers(n_layers):
    last = n_layers - 1
    layers = sorted(set([
        0,
        n_layers // 6,
        n_layers // 3,
        n_layers // 2,
        n_layers // 2 + 1,
        min(20, last),
        max(last - 3, n_layers // 2 + 2),
        last,
    ]))
    return [l for l in layers if 0 <= l < n_layers]


def auto_detect_dims(vector_dir, traits):
    for t in traits:
        p = os.path.join(vector_dir, f"{t}_i_thou_response_avg.pt")
        if os.path.exists(p):
            v = torch.load(p, map_location="cpu")
            return v.shape[0], v.shape[1]
    raise RuntimeError("No vectors found to detect dimensions")


def discover_traits(vector_dir):
    pattern = os.path.join(vector_dir, "*_i_thou_response_avg.pt")
    files = sorted(glob.glob(pattern))
    return [os.path.basename(f).replace("_i_thou_response_avg.pt", "") for f in files]


def load_vectors(vector_dir, traits, position):
    vecs = {}
    for t in traits:
        p = os.path.join(vector_dir, f"{t}_i_thou_{position}.pt")
        if os.path.exists(p):
            vecs[t] = torch.load(p, map_location="cpu")
    return vecs


def cossim_matrix(vectors, traits, layer):
    rows = []
    for t in traits:
        v = vectors[t][layer]
        n = v.norm()
        rows.append(v / n if n > 0 else torch.zeros_like(v))
    mat = torch.stack(rows)
    return (mat @ mat.T).numpy()


def utri_stats(matrix):
    n = matrix.shape[0]
    vals = matrix[np.triu_indices(n, k=1)]
    if len(vals) == 0 or np.all(np.isnan(vals)):
        return {"mean": None, "std": None, "min": None, "max": None, "median": None, "n_pairs": 0}
    mask = ~np.isnan(vals)
    v = vals[mask]
    return {
        "mean": round(float(np.mean(v)), 6),
        "std": round(float(np.std(v)), 6),
        "min": round(float(np.min(v)), 4),
        "max": round(float(np.max(v)), 4),
        "median": round(float(np.median(v)), 6),
        "n_pairs": int(len(v)),
    }


def round_matrix(m, d=4):
    return [[round(float(x), d) for x in row] for row in m]


def null_distribution(n_traits, dim, n_layers, n_iter):
    print(f"Null test: {n_iter} iterations × {n_layers} layers, {n_traits} vectors in R^{dim}")
    np.random.seed(42)
    results = {}
    for layer in range(n_layers):
        means = np.empty(n_iter, dtype=np.float64)
        idx = np.triu_indices(n_traits, k=1)
        for i in range(n_iter):
            v = np.random.randn(n_traits, dim).astype(np.float32)
            v /= np.linalg.norm(v, axis=1, keepdims=True)
            sim = v @ v.T
            means[i] = np.mean(sim[idx])
        results[str(layer)] = {
            "null_mean": round(float(np.mean(means)), 8),
            "null_std": round(float(np.std(means)), 8),
            "null_min": round(float(np.min(means)), 6),
            "null_max": round(float(np.max(means)), 6),
            "null_p50": round(float(np.percentile(means, 50)), 6),
            "null_p95": round(float(np.percentile(means, 95)), 6),
            "null_p99": round(float(np.percentile(means, 99)), 6),
            "null_p999": round(float(np.percentile(means, 99.9)), 6),
        }
        if layer % 5 == 0 or layer == n_layers - 1:
            r = results[str(layer)]
            print(f"  L{layer:02d}: null_mean={r['null_mean']:+.8f}  std={r['null_std']:.8f}")
    return results


def top_pairs(matrix, traits, n=20):
    idx = np.triu_indices(len(traits), k=1)
    vals = matrix[idx]
    order = np.argsort(vals)
    top = [{"trait_a": traits[idx[0][i]], "trait_b": traits[idx[1][i]], "cos_sim": round(float(vals[i]), 4)} for i in order[-n:][::-1]]
    bot = [{"trait_a": traits[idx[0][i]], "trait_b": traits[idx[1][i]], "cos_sim": round(float(vals[i]), 4)} for i in order[:n]]
    return top, bot


def semantic_pair_sims(matrix, traits, pairs):
    t2i = {t: i for i, t in enumerate(traits)}
    out = {}
    for a, b in pairs:
        if a in t2i and b in t2i:
            out[f"{a}_vs_{b}"] = round(float(matrix[t2i[a], t2i[b]]), 4)
        else:
            out[f"{a}_vs_{b}"] = None
    return out


def rlhf_cluster(matrix, traits, defaults):
    t2i = {t: i for i, t in enumerate(traits)}
    di = [t2i[t] for t in defaults if t in t2i]
    ni = [i for i in range(len(traits)) if i not in di]
    intra = [float(matrix[di[a], di[b]]) for a in range(len(di)) for b in range(a + 1, len(di))]
    inter = [float(matrix[d, n]) for d in di for n in ni]
    non_d = [float(matrix[ni[a], ni[b]]) for a in range(len(ni)) for b in range(a + 1, len(ni))]
    def s(v):
        return {"mean": round(float(np.mean(v)), 4), "std": round(float(np.std(v)), 4), "n": len(v)} if v else None
    return {
        "n_default": len(di), "n_non_default": len(ni),
        "intra_default": s(intra), "inter": s(inter), "intra_non_default": s(non_d),
        "defaults_found": [t for t in defaults if t in t2i],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vector_dir", required=True)
    parser.add_argument("--output_dir", default="experiment_result")
    parser.add_argument("--skip_null", action="store_true")
    args = parser.parse_args()

    model_name = os.path.basename(args.vector_dir.rstrip("/"))
    traits = discover_traits(args.vector_dir)
    n_traits = len(traits)
    n_layers, dim = auto_detect_dims(args.vector_dir, traits)
    key_layers = compute_key_layers(n_layers)
    print(f"Model: {model_name}")
    print(f"Discovered {n_traits} traits")
    print(f"Detected: {n_layers} layers, {dim} hidden dim")
    print(f"Key layers: {key_layers}")
    print(f"Traits: {traits[:5]} ... {traits[-5:]}")

    output = {
        "metadata": {
            "model": model_name,
            "n_traits": n_traits,
            "n_layers": n_layers,
            "hidden_dim": dim,
            "positions": POSITIONS,
            "key_layers": key_layers,
            "null_iterations": NULL_ITERATIONS,
            "timestamp": datetime.now().isoformat(),
        },
        "traits": traits,
        "rlhf_default_traits": RLHF_DEFAULTS,
        "matrices": {},
        "stats": {},
        "analysis": {},
    }

    for pos in POSITIONS:
        print(f"\n{'='*50}")
        print(f"POSITION: {pos}")
        print(f"{'='*50}")

        vectors = load_vectors(args.vector_dir, traits, pos)
        loaded = len(vectors)
        print(f"  Loaded {loaded}/{n_traits} vectors")
        if loaded < n_traits:
            print(f"  Missing: {[t for t in traits if t not in vectors]}")

        all_mats = []
        layer_stats = {}
        key_layer_mats = {}

        for layer in range(n_layers):
            mat = cossim_matrix(vectors, traits, layer)
            all_mats.append(mat)
            st = utri_stats(mat)
            layer_stats[str(layer)] = st
            if layer in key_layers:
                key_layer_mats[str(layer)] = round_matrix(mat)
            if layer in [0, n_layers // 3, min(20, n_layers - 1), n_layers - 1]:
                m = st["mean"]
                print(f"  L{layer:02d}: mean={m if m is not None else 'NaN'}")

        layer_avg = np.nanmean(all_mats, axis=0)
        avg_st = utri_stats(layer_avg)
        print(f"  Layer-avg: mean={avg_st['mean']}")

        output["matrices"][pos] = {
            "layer_avg": round_matrix(layer_avg),
            "key_layers": key_layer_mats,
        }
        output["stats"][pos] = {
            "per_layer": layer_stats,
            "layer_avg": avg_st,
        }

        sim, dissim = top_pairs(layer_avg, traits)
        sem = semantic_pair_sims(layer_avg, traits, SEMANTIC_PAIRS)
        rlhf = rlhf_cluster(layer_avg, traits, RLHF_DEFAULTS)

        output["analysis"][pos] = {
            "top_20_similar": sim,
            "top_20_dissimilar": dissim,
            "semantic_pairs": sem,
            "rlhf_cluster": rlhf,
        }

        print(f"\n  Top 5 similar:")
        for p in sim[:5]:
            print(f"    {p['trait_a']:20s} ↔ {p['trait_b']:20s}  {p['cos_sim']:+.4f}")
        print(f"  Top 5 dissimilar:")
        for p in dissim[:5]:
            print(f"    {p['trait_a']:20s} ↔ {p['trait_b']:20s}  {p['cos_sim']:+.4f}")
        print(f"\n  Semantic pairs:")
        for k, v in sem.items():
            print(f"    {k:35s}  {v}")
        print(f"\n  RLHF cluster: intra={rlhf['intra_default']}, inter={rlhf['inter']}")

    # Null distribution (position-independent, same random vectors)
    if not args.skip_null:
        print(f"\n{'='*50}")
        print("NULL DISTRIBUTION")
        print(f"{'='*50}")
        null_raw = null_distribution(n_traits, dim, n_layers, NULL_ITERATIONS)

        output["null_distribution"] = {"n_iterations": NULL_ITERATIONS, "dim": dim, "n_vectors": n_traits}
        output["null_distribution"]["raw"] = null_raw

        print("\n  Z-scores (observed vs null):")
        print(f"  {'Layer':>5} | {'prompt_end':>12} | {'response_start':>16} | {'response_avg':>14}")
        print("  " + "-" * 58)

        for pos in POSITIONS:
            output["null_distribution"][pos] = {}
            for ls in [str(l) for l in range(n_layers)]:
                obs = output["stats"][pos]["per_layer"][ls]["mean"]
                ns = null_raw[ls]
                if obs is not None and ns["null_std"] > 0:
                    z = round((obs - ns["null_mean"]) / ns["null_std"], 1)
                else:
                    z = None
                output["null_distribution"][pos][ls] = {
                    "observed_mean": obs,
                    "z_score": z,
                }

        for layer in range(n_layers):
            ls = str(layer)
            zs = [output["null_distribution"][p][ls]["z_score"] for p in POSITIONS]
            zstrs = [f"{z:>+.1f}" if z is not None else "NaN" for z in zs]
            print(f"  L{layer:02d}   | {zstrs[0]:>12} | {zstrs[1]:>16} | {zstrs[2]:>14}")
    else:
        output["null_distribution"] = None
        print("\nSkipped null test (--skip_null)")

    # Save full output
    os.makedirs(args.output_dir, exist_ok=True)

    full_path = os.path.join(args.output_dir, f"{model_name}_cross_trait_cossim_full.json")
    with open(full_path, "w") as f:
        json.dump(output, f)
    fsize = os.path.getsize(full_path) / (1024 * 1024)
    print(f"\nSaved {full_path}  ({fsize:.1f} MB)")

    # Save summary (no matrices)
    summary = {
        "metadata": output["metadata"],
        "traits": output["traits"],
        "rlhf_default_traits": output["rlhf_default_traits"],
        "stats": output["stats"],
        "null_distribution": output["null_distribution"],
        "analysis": output["analysis"],
    }
    sum_path = os.path.join(args.output_dir, f"{model_name}_cross_trait_cossim_summary.json")
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)
    ssize = os.path.getsize(sum_path) / 1024
    print(f"Saved {sum_path}  ({ssize:.0f} KB)")

    print(f"\n{'='*50}")
    print("DONE")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
