#!/usr/bin/env python3
"""
PCA shared-axis analysis of 88 I-Thou vectors.

Self-contained script — runs on remote GPU server via LLMOS.
Only depends on torch, numpy, scipy, matplotlib, json.

Usage:
    python scripts/pca_shared_axis.py
"""

import os
import json
import time
import numpy as np
import torch
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not available, skipping plots")

import argparse
from pathlib import Path

try:
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not available, skipping cluster dedup")

# ── Config ──────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="PCA shared-axis analysis of I-Thou vectors")
    parser.add_argument("--vector_dir", type=str, default="data/vectors/Qwen2.5-7B-Instruct",
                        help="Path to vector directory")
    parser.add_argument("--output_dir", type=str, default="experiment_result",
                        help="Path to output directory")
    return parser.parse_args()

_args = parse_args()
VECTOR_DIR = Path(_args.vector_dir)
OUTPUT_DIR = Path(_args.output_dir)
PC1_DIR = OUTPUT_DIR / "pca_pc1_vectors"
POSITIONS = ["prompt_end", "response_start", "response_avg"]
FOCAL_POSITION = "response_avg"

def _auto_detect_layers():
    """Auto-detect n_layers and compute focal/key layers from first vector."""
    files = sorted(VECTOR_DIR.glob("*_i_thou_response_avg.pt"))
    if not files:
        return 21, [0, 5, 10, 15, 17, 20, 21, 25, 28]
    vec = torch.load(files[0], map_location="cpu", weights_only=True)
    n_layers = vec.shape[0]
    n_transformer = n_layers - 1
    focal = int(round(n_transformer * 0.75))
    steering = int(round(n_transformer * 0.625))
    key = sorted(set([
        0,
        max(1, n_transformer // 6),
        n_transformer // 3,
        n_transformer // 2,
        steering,
        int(round(n_transformer * 0.71)),
        focal,
        n_transformer - 3,
        n_transformer,
    ]))
    print(f"Auto-detected: {n_layers} layers (focal=L{focal}, key={key})")
    return focal, key

FOCAL_LAYER, KEY_LAYERS = _auto_detect_layers()
N_PCS_REPORT = 20
BOOTSTRAP_N = 100
CLUSTER_THRESHOLD = 0.5
NEAR_ZERO_THRESHOLD = 1e-6
NEAR_ZERO_MAX_FRAC = 0.1

np.random.seed(42)


# ── Helpers ─────────────────────────────────────────────────────────────

def l2_normalize(v):
    """L2-normalize rows of a 2D array."""
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.where(norms < NEAR_ZERO_THRESHOLD, 1.0, norms)
    return v / norms


def cos_sim(a, b):
    """Cosine similarity between two 1D vectors."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def abs_cos_sim(a, b):
    """Absolute cosine similarity (sign-invariant)."""
    return abs(cos_sim(a, b))


def run_svd(V_normed):
    """Run both uncentered and centered SVD, return results dict."""
    n_traits = V_normed.shape[0]
    n_pcs = min(N_PCS_REPORT, n_traits)

    # Uncentered SVD
    U_unc, S_unc, Vt_unc = np.linalg.svd(V_normed, full_matrices=False)
    var_unc = (S_unc ** 2) / (S_unc ** 2).sum()

    # Centered SVD (standard PCA)
    V_centered = V_normed - V_normed.mean(axis=0, keepdims=True)
    U_c, S_c, Vt_c = np.linalg.svd(V_centered, full_matrices=False)
    var_c = (S_c ** 2) / (S_c ** 2).sum()

    # Sanity check: cos(PC1_uncentered, mean_direction)
    r_mean = V_normed.mean(axis=0)
    r_mean_norm = r_mean / (np.linalg.norm(r_mean) + 1e-12)
    cos_pc1_mean = cos_sim(Vt_unc[0], r_mean_norm)

    return {
        "explained_var_uncentered": var_unc[:n_pcs].tolist(),
        "explained_var_centered": var_c[:n_pcs].tolist(),
        "cos_pc1_unc_vs_mean": cos_pc1_mean,
        "pc1_uncentered": Vt_unc[0],  # numpy array, not serialized
        "pc1_centered": Vt_c[0],
        "singular_values_unc": S_unc[:n_pcs].tolist(),
        "singular_values_cen": S_c[:n_pcs].tolist(),
    }


# ── Step 1: Load vectors ───────────────────────────────────────────────

def load_all_ithou_vectors():
    """Load all 88 I-Thou vectors for all positions. Returns {position: {trait: tensor[29, 3584]}}."""
    # Discover traits
    files = sorted(VECTOR_DIR.glob("*_i_thou_response_avg.pt"))
    traits = [f.stem.replace("_i_thou_response_avg", "") for f in files]
    print(f"Found {len(traits)} traits")

    vectors = {}
    for pos in POSITIONS:
        vectors[pos] = {}
        for trait in traits:
            path = VECTOR_DIR / f"{trait}_i_thou_{pos}.pt"
            vec = torch.load(path, map_location="cpu", weights_only=True)
            vectors[pos][trait] = vec.numpy().astype(np.float32)
        print(f"  Loaded {len(vectors[pos])} vectors for {pos}")

    return traits, vectors


def load_model_persona_vectors(traits):
    """Load model_persona vectors for cluster dedup (Step 5a). Uses response_avg only."""
    mp_vecs = {}
    for trait in traits:
        path = VECTOR_DIR / f"{trait}_model_persona_response_avg.pt"
        if path.exists():
            vec = torch.load(path, map_location="cpu", weights_only=True)
            mp_vecs[trait] = vec.numpy().astype(np.float32)
    print(f"  Loaded {len(mp_vecs)} model_persona vectors for cluster dedup")
    return mp_vecs


# ── Step 2-3: PCA per (position, layer) ────────────────────────────────

def run_pca_all(traits, vectors):
    """Run PCA for all positions and layers. Returns full results dict."""
    results = {}

    for pos in POSITIONS:
        results[pos] = {}
        n_layers = vectors[pos][traits[0]].shape[0]

        for layer in range(n_layers):
            # Stack and normalize
            V = np.stack([vectors[pos][t][layer] for t in traits])  # [88, 3584]

            # Check for near-zero vectors
            norms = np.linalg.norm(V, axis=1)
            near_zero_frac = (norms < NEAR_ZERO_THRESHOLD).mean()
            if near_zero_frac > NEAR_ZERO_MAX_FRAC:
                results[pos][layer] = {"skipped": True, "near_zero_frac": float(near_zero_frac)}
                continue

            V_normed = l2_normalize(V)
            svd_result = run_svd(V_normed)

            results[pos][layer] = {
                "skipped": False,
                "explained_var_uncentered": svd_result["explained_var_uncentered"],
                "explained_var_centered": svd_result["explained_var_centered"],
                "cos_pc1_unc_vs_mean": svd_result["cos_pc1_unc_vs_mean"],
                "singular_values_unc": svd_result["singular_values_unc"],
                "singular_values_cen": svd_result["singular_values_cen"],
            }

            # Save PC1 vectors for key layers
            if layer in KEY_LAYERS:
                pc1_path = PC1_DIR / f"{pos}_L{layer}.pt"
                torch.save(
                    torch.from_numpy(svd_result["pc1_uncentered"].astype(np.float32)),
                    pc1_path,
                )

        print(f"  PCA done for {pos} ({n_layers} layers)")

    return results


# ── Step 4: Decision rule ──────────────────────────────────────────────

def apply_decision_rule(results):
    """Apply decision rule at focal position/layer."""
    focal = results[FOCAL_POSITION].get(FOCAL_LAYER)
    if focal is None or focal.get("skipped"):
        return {"decision": "SKIPPED", "reason": "Focal layer was skipped (near-zero vectors)"}

    ev_unc = focal["explained_var_uncentered"]
    pc1 = ev_unc[0]
    pc2 = ev_unc[1] if len(ev_unc) > 1 else 0.0

    # Check cumulative for top 3-5
    top3_cum = sum(ev_unc[:3])
    top5_cum = sum(ev_unc[:5])

    if pc1 > 0.20 and pc2 < 0.08:
        decision = "SINGLE_DOMINANT_AXIS"
        reason = f"PC1={pc1:.4f} (>{0.20}), PC2={pc2:.4f} (<{0.08}). Single dominant shared axis."
    elif all(0.03 < ev_unc[i] < 0.20 for i in range(min(3, len(ev_unc)))):
        decision = "SHARED_SUBSPACE"
        reason = f"Top-3 PCs each in 3-20% range (cum={top3_cum:.4f}). Shared subspace of dim ~3-5."
    elif pc1 < 0.05:
        decision = "NO_USABLE_DIRECTION"
        reason = f"PC1={pc1:.4f} (<{0.05}). No usable shared direction."
    else:
        decision = "AMBIGUOUS"
        reason = f"PC1={pc1:.4f}, PC2={pc2:.4f}, top3_cum={top3_cum:.4f}. Does not clearly fit any category."

    return {
        "decision": decision,
        "reason": reason,
        "focal_position": FOCAL_POSITION,
        "focal_layer": FOCAL_LAYER,
        "pc1_explained": pc1,
        "pc2_explained": pc2,
        "top3_cumulative": top3_cum,
        "top5_cumulative": top5_cum,
        "cos_pc1_vs_mean": focal["cos_pc1_unc_vs_mean"],
    }


# ── Step 5a: Cluster dedup ─────────────────────────────────────────────

def cluster_dedup(traits, vectors, mp_vectors):
    """Cluster traits by model_persona cos-sim, re-run PCA on prototypical traits."""
    print("\n=== Step 5a: Cluster dedup ===")

    # Use model_persona response_avg at focal layer for clustering
    available_traits = [t for t in traits if t in mp_vectors]
    V = np.stack([mp_vectors[t][FOCAL_LAYER] for t in available_traits])
    V_normed = l2_normalize(V)

    # Pairwise cosine similarity → distance
    cos_mat = V_normed @ V_normed.T
    cos_mat = np.clip(cos_mat, -1, 1)
    dist_mat = 1.0 - cos_mat
    np.fill_diagonal(dist_mat, 0)
    dist_mat = np.maximum(dist_mat, 0)
    dist_condensed = squareform(dist_mat, checks=False)

    # Agglomerative clustering
    Z = linkage(dist_condensed, method="average")
    labels = fcluster(Z, t=CLUSTER_THRESHOLD, criterion="distance")
    n_clusters = len(set(labels))

    # Select prototypical trait per cluster (highest mean cos-sim to cluster members)
    cluster_map = {}
    prototypical_traits = []
    for c in sorted(set(labels)):
        members = [available_traits[i] for i, l in enumerate(labels) if l == c]
        member_indices = [i for i, l in enumerate(labels) if l == c]

        if len(members) == 1:
            proto = members[0]
        else:
            sub_cos = cos_mat[np.ix_(member_indices, member_indices)]
            mean_cos = sub_cos.mean(axis=1)
            proto_idx = member_indices[np.argmax(mean_cos)]
            proto = available_traits[proto_idx]

        prototypical_traits.append(proto)
        cluster_map[str(c)] = {"prototype": proto, "members": members, "size": len(members)}

    print(f"  {len(available_traits)} traits → {n_clusters} clusters")
    print(f"  Prototypical traits: {len(prototypical_traits)}")

    # Re-run PCA on deduped set at focal position/layer
    V_dedup = np.stack([vectors[FOCAL_POSITION][t][FOCAL_LAYER] for t in prototypical_traits])
    V_dedup_normed = l2_normalize(V_dedup)
    svd_dedup = run_svd(V_dedup_normed)

    return {
        "n_clusters": n_clusters,
        "n_prototypical": len(prototypical_traits),
        "prototypical_traits": prototypical_traits,
        "cluster_map": cluster_map,
        "dedup_explained_var_unc": svd_dedup["explained_var_uncentered"],
        "dedup_explained_var_cen": svd_dedup["explained_var_centered"],
        "dedup_cos_pc1_vs_mean": svd_dedup["cos_pc1_unc_vs_mean"],
    }


# ── Step 5b: Bootstrap stability ───────────────────────────────────────

def bootstrap_stability(traits, vectors):
    """Resample traits, recompute PC1, measure stability."""
    print("\n=== Step 5b: Bootstrap stability ===")

    # Get normalized vectors at focal position/layer
    V = np.stack([vectors[FOCAL_POSITION][t][FOCAL_LAYER] for t in traits])
    V_normed = l2_normalize(V)

    pc1s = []
    for i in range(BOOTSTRAP_N):
        idx = np.random.choice(len(traits), size=len(traits), replace=True)
        V_boot = V_normed[idx]
        _, _, Vt = np.linalg.svd(V_boot, full_matrices=False)
        pc1s.append(Vt[0])

    # Pairwise absolute cosine similarity between bootstrap PC1s
    pc1_mat = np.stack(pc1s)
    cos_mat = np.abs(pc1_mat @ pc1_mat.T)
    np.fill_diagonal(cos_mat, np.nan)
    mean_cos = float(np.nanmean(cos_mat))
    std_cos = float(np.nanstd(cos_mat))
    min_cos = float(np.nanmin(cos_mat))

    stability = "STABLE" if mean_cos > 0.9 else ("MODERATE" if mean_cos > 0.7 else "UNSTABLE")
    print(f"  Bootstrap PC1 stability: mean |cos|={mean_cos:.4f}, std={std_cos:.4f}, min={min_cos:.4f} → {stability}")

    return {
        "n_bootstrap": BOOTSTRAP_N,
        "mean_abs_cos": mean_cos,
        "std_abs_cos": std_cos,
        "min_abs_cos": min_cos,
        "stability": stability,
        "pairwise_cos_distribution": {
            "p5": float(np.nanpercentile(cos_mat, 5)),
            "p25": float(np.nanpercentile(cos_mat, 25)),
            "p50": float(np.nanpercentile(cos_mat, 50)),
            "p75": float(np.nanpercentile(cos_mat, 75)),
            "p95": float(np.nanpercentile(cos_mat, 95)),
        },
    }


# ── Step 5c: Layer consistency ──────────────────────────────────────────

def layer_consistency(traits, vectors):
    """Compute cos(PC1_L, PC1_{L+1}) for adjacent layers."""
    print("\n=== Step 5c: Layer consistency ===")

    results = {}
    for pos in POSITIONS:
        n_layers = vectors[pos][traits[0]].shape[0]
        pc1s = []

        for layer in range(n_layers):
            V = np.stack([vectors[pos][t][layer] for t in traits])
            norms = np.linalg.norm(V, axis=1)
            if (norms < NEAR_ZERO_THRESHOLD).mean() > NEAR_ZERO_MAX_FRAC:
                pc1s.append(None)
                continue
            V_normed = l2_normalize(V)
            _, _, Vt = np.linalg.svd(V_normed, full_matrices=False)
            pc1s.append(Vt[0])

        # Adjacent layer cos-sim
        adjacent = {}
        for layer in range(n_layers - 1):
            if pc1s[layer] is not None and pc1s[layer + 1] is not None:
                adjacent[f"L{layer}_L{layer+1}"] = abs_cos_sim(pc1s[layer], pc1s[layer + 1])

        values = list(adjacent.values())
        mean_adj = float(np.mean(values)) if values else 0.0
        min_adj = float(np.min(values)) if values else 0.0

        consistency = "CONSISTENT" if mean_adj > 0.7 else ("MODERATE" if mean_adj > 0.5 else "INCONSISTENT")
        results[pos] = {
            "adjacent_cos": adjacent,
            "mean_adjacent_cos": mean_adj,
            "min_adjacent_cos": min_adj,
            "consistency": consistency,
        }
        print(f"  {pos}: mean adj |cos|={mean_adj:.4f}, min={min_adj:.4f} → {consistency}")

    return results


# ── Plotting ────────────────────────────────────────────────────────────

def plot_spectrum_focal(results):
    """Plot explained variance spectrum at focal position/layer."""
    focal = results[FOCAL_POSITION][FOCAL_LAYER]
    if focal.get("skipped"):
        print("  Skipping focal plot (layer was skipped)")
        return

    ev_unc = focal["explained_var_uncentered"]
    ev_cen = focal["explained_var_centered"]
    n = min(N_PCS_REPORT, len(ev_unc))
    x = np.arange(1, n + 1)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.bar(x - 0.2, ev_unc[:n], width=0.4, label="Uncentered", color="#2196F3", alpha=0.85)
    ax.bar(x + 0.2, ev_cen[:n], width=0.4, label="Centered", color="#FF9800", alpha=0.85)
    ax.set_xlabel("Principal Component", fontsize=12)
    ax.set_ylabel("Explained Variance Ratio", fontsize=12)
    ax.set_title(f"PCA Spectrum — {FOCAL_POSITION} L{FOCAL_LAYER} (88 I-Thou vectors)", fontsize=13)
    ax.set_xticks(x)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    # Annotate PC1/PC2
    ax.annotate(f"{ev_unc[0]:.3f}", (1 - 0.2, ev_unc[0]), ha="center", va="bottom", fontsize=9, fontweight="bold")
    if len(ev_unc) > 1:
        ax.annotate(f"{ev_unc[1]:.3f}", (2 - 0.2, ev_unc[1]), ha="center", va="bottom", fontsize=9)

    path = OUTPUT_DIR / f"pca_spectrum_response_avg_L{FOCAL_LAYER}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_spectrum_all_positions(results):
    """Plot spectrum comparison across 3 positions at focal layer."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for ax, pos in zip(axes, POSITIONS):
        data = results[pos].get(FOCAL_LAYER)
        if data is None or data.get("skipped"):
            ax.set_title(f"{pos} L{FOCAL_LAYER} (skipped)")
            continue

        ev_unc = data["explained_var_uncentered"]
        ev_cen = data["explained_var_centered"]
        n = min(N_PCS_REPORT, len(ev_unc))
        x = np.arange(1, n + 1)

        ax.bar(x - 0.2, ev_unc[:n], width=0.4, label="Uncentered", color="#2196F3", alpha=0.85)
        ax.bar(x + 0.2, ev_cen[:n], width=0.4, label="Centered", color="#FF9800", alpha=0.85)
        ax.set_xlabel("PC")
        ax.set_title(f"{pos} L{FOCAL_LAYER}")
        ax.set_xticks(x)
        ax.grid(axis="y", alpha=0.3)
        ax.annotate(f"{ev_unc[0]:.3f}", (1 - 0.2, ev_unc[0]), ha="center", va="bottom", fontsize=8, fontweight="bold")

    axes[0].set_ylabel("Explained Variance Ratio")
    axes[0].legend(fontsize=9)
    fig.suptitle(f"PCA Spectrum Comparison — L{FOCAL_LAYER} (88 I-Thou vectors)", fontsize=14)
    fig.tight_layout()

    path = OUTPUT_DIR / f"pca_spectrum_all_positions_L{FOCAL_LAYER}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_bootstrap(bootstrap_results):
    """Plot bootstrap cos-sim distribution."""
    dist = bootstrap_results["pairwise_cos_distribution"]
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    keys = ["p5", "p25", "p50", "p75", "p95"]
    vals = [dist[k] for k in keys]
    ax.bar(keys, vals, color="#4CAF50", alpha=0.8)
    ax.axhline(y=0.9, color="red", linestyle="--", alpha=0.7, label="Stability threshold (0.9)")
    ax.axhline(y=0.7, color="orange", linestyle="--", alpha=0.7, label="Moderate threshold (0.7)")
    ax.set_ylabel("|cos(PC1_i, PC1_j)|")
    ax.set_title(f"Bootstrap PC1 Stability ({BOOTSTRAP_N} resamples, {FOCAL_POSITION} L{FOCAL_LAYER})")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    path = OUTPUT_DIR / "pca_bootstrap_stability.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Main ────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 60)
    print("PCA Shared-Axis Analysis of 88 I-Thou Vectors")
    print("=" * 60)

    # Setup output dirs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PC1_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load
    print("\n--- Step 1: Loading vectors ---")
    traits, vectors = load_all_ithou_vectors()
    mp_vectors = load_model_persona_vectors(traits)

    # Step 2-3: PCA
    print("\n--- Step 2-3: PCA (all positions × all layers) ---")
    pca_results = run_pca_all(traits, vectors)

    # Print key results
    print("\n--- Key Results (uncentered PC1-5 explained variance) ---")
    for pos in POSITIONS:
        print(f"\n  {pos}:")
        for layer in KEY_LAYERS:
            data = pca_results[pos].get(layer)
            if data is None or data.get("skipped"):
                print(f"    L{layer:2d}: SKIPPED")
                continue
            ev = data["explained_var_uncentered"]
            cos_val = data["cos_pc1_unc_vs_mean"]
            top5 = " ".join([f"{v:.4f}" for v in ev[:5]])
            print(f"    L{layer:2d}: [{top5}]  cos(PC1,mean)={cos_val:+.4f}")

    # Step 4: Decision rule
    print("\n--- Step 4: Decision rule ---")
    decision = apply_decision_rule(pca_results)
    print(f"  Decision: {decision['decision']}")
    print(f"  Reason: {decision['reason']}")
    print(f"  cos(PC1, mean): {decision.get('cos_pc1_vs_mean', 'N/A')}")

    # Step 5: Robustness checks
    print("\n--- Step 5: Robustness checks ---")
    focal_ev = pca_results[FOCAL_POSITION][FOCAL_LAYER]["explained_var_uncentered"][0]

    if HAS_SCIPY:
        cluster_results = cluster_dedup(traits, vectors, mp_vectors)
        dedup_ev = cluster_results["dedup_explained_var_unc"][0]
        print(f"\n  Cluster dedup comparison (PC1 explained var):")
        print(f"    Full 88 traits: {focal_ev:.4f}")
        print(f"    Deduped {cluster_results['n_prototypical']} traits: {dedup_ev:.4f}")
        print(f"    Ratio: {dedup_ev/focal_ev:.2f}x")
    else:
        cluster_results = None
        dedup_ev = None
        print("  Skipping cluster dedup (scipy not available)")

    bootstrap_results = bootstrap_stability(traits, vectors)
    layer_results = layer_consistency(traits, vectors)

    # Plots
    if HAS_MPL:
        print("\n--- Plotting ---")
        plot_spectrum_focal(pca_results)
        plot_spectrum_all_positions(pca_results)
        plot_bootstrap(bootstrap_results)
    else:
        print("\n--- Skipping plots (matplotlib not available) ---")

    # ── Assemble and save results ───────────────────────────────────────

    # Full results (all layers, all positions)
    full_output = {
        "metadata": {
            "date": time.strftime("%Y-%m-%d"),
            "model": str(VECTOR_DIR.name),
            "n_traits": len(traits),
            "traits": traits,
            "positions": POSITIONS,
            "n_layers": vectors[POSITIONS[0]][traits[0]].shape[0],
            "focal_position": FOCAL_POSITION,
            "focal_layer": FOCAL_LAYER,
            "runtime_seconds": round(time.time() - t0, 1),
        },
        "pca_results": {},
        "decision": decision,
        "robustness": {
            "cluster_dedup": {k: v for k, v in cluster_results.items() if k != "cluster_map"} if cluster_results else None,
            "bootstrap": bootstrap_results,
            "layer_consistency": layer_results,
        },
        "cluster_map": cluster_results["cluster_map"] if cluster_results else None,
    }

    # Serialize PCA results (convert layer keys to strings for JSON)
    for pos in POSITIONS:
        full_output["pca_results"][pos] = {}
        for layer, data in pca_results[pos].items():
            full_output["pca_results"][pos][str(layer)] = data

    full_path = OUTPUT_DIR / "pca_shared_axis_full.json"
    with open(full_path, "w") as f:
        json.dump(full_output, f, indent=2)
    print(f"\n  Saved full results: {full_path}")

    # Summary (key findings only)
    summary = {
        "metadata": full_output["metadata"],
        "decision": decision,
        "focal_results": {
            "explained_var_uncentered_top10": pca_results[FOCAL_POSITION][FOCAL_LAYER]["explained_var_uncentered"][:10],
            "explained_var_centered_top10": pca_results[FOCAL_POSITION][FOCAL_LAYER]["explained_var_centered"][:10],
            "cos_pc1_unc_vs_mean": pca_results[FOCAL_POSITION][FOCAL_LAYER]["cos_pc1_unc_vs_mean"],
        },
        "robustness_verdicts": {
            "cluster_dedup": {
                "n_clusters": cluster_results["n_clusters"],
                "n_prototypical": cluster_results["n_prototypical"],
                "pc1_full": focal_ev,
                "pc1_dedup": dedup_ev,
                "ratio": round(dedup_ev / focal_ev, 3),
                "verdict": "ROBUST" if dedup_ev / focal_ev > 0.7 else "INFLATED",
            } if cluster_results else None,
            "bootstrap": {
                "mean_abs_cos": bootstrap_results["mean_abs_cos"],
                "stability": bootstrap_results["stability"],
            },
            "layer_consistency": {
                pos: {
                    "mean_adj_cos": layer_results[pos]["mean_adjacent_cos"],
                    "consistency": layer_results[pos]["consistency"],
                }
                for pos in POSITIONS
            },
        },
        "key_layers_uncentered_pc1": {},
    }

    for layer in KEY_LAYERS:
        data = pca_results[FOCAL_POSITION].get(layer)
        if data and not data.get("skipped"):
            summary["key_layers_uncentered_pc1"][f"L{layer}"] = {
                "pc1": data["explained_var_uncentered"][0],
                "pc2": data["explained_var_uncentered"][1],
                "top5_cum": sum(data["explained_var_uncentered"][:5]),
            }

    summary_path = OUTPUT_DIR / "pca_shared_axis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary: {summary_path}")

    # Final summary
    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"DONE in {elapsed:.1f}s")
    print(f"{'=' * 60}")
    print(f"\nDecision: {decision['decision']}")
    print(f"Reason: {decision['reason']}")
    print(f"\nRobustness:")
    if cluster_results:
        print(f"  Cluster dedup: {cluster_results['n_clusters']} clusters, PC1 ratio={dedup_ev/focal_ev:.2f}x")
    print(f"  Bootstrap: mean |cos|={bootstrap_results['mean_abs_cos']:.4f} ({bootstrap_results['stability']})")
    for pos in POSITIONS:
        lc = layer_results[pos]
        print(f"  Layer consistency ({pos}): mean adj |cos|={lc['mean_adjacent_cos']:.4f} ({lc['consistency']})")


if __name__ == "__main__":
    main()
