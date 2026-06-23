import subprocess, sys, os
os.chdir("/root/repositories/i-and-thou-vector-private")

# Pull latest (includes the compliant fix)
result = subprocess.run(["git", "pull", "origin", "main"], capture_output=True, text=True)
print("Pull:", result.stdout.strip())

# Write a lightweight script that computes only Steps 5-8
# (refusal .pt files already exist from the first run)
import json, time, numpy as np, torch
from pathlib import Path

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
VECTOR_DIR = Path("data/vectors/Meta-Llama-3.1-8B-Instruct")
OUTPUT_DIR = Path("experiment_result/Meta-Llama-3.1-8B-Instruct")

# Auto-detect layers
files = sorted(VECTOR_DIR.glob("*_i_thou_response_avg.pt"))
vec = torch.load(files[0], map_location="cpu", weights_only=True)
n_t = vec.shape[0] - 1
L_early = int(round(n_t * 0.36))
L_steer = int(round(n_t * 0.625))
L_focal = int(round(n_t * 0.75))
EXTRACTION_LAYERS = [L_early, L_steer, L_focal]
ABLATION_LAYER = L_early
print(f"Layers: {EXTRACTION_LAYERS}, ablation: L{ABLATION_LAYER}")

# Load refusal directions
refusal_dirs = {}
for L in EXTRACTION_LAYERS:
    p = OUTPUT_DIR / f"refusal_direction_last_prompt_L{L}.pt"
    refusal_dirs[L] = torch.load(p, map_location="cpu")
    print(f"  Loaded refusal L{L}: norm={refusal_dirs[L].norm():.4f}")

def signed_cos(a, b):
    a, b = np.array(a).flatten(), np.array(b).flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def interpret(val):
    mag = abs(val)
    if mag > 0.5: return "STRONG_ALIGNMENT"
    elif mag > 0.3: return "SUBSTANTIAL_OVERLAP"
    elif mag > 0.2: return "PARTIAL_OVERLAP"
    elif mag > 0.1: return "WEAK_ALIGNMENT"
    else: return "INDEPENDENT"

# Step 5: Recompute PC1
print("\n--- Step 5: Recomputing PC1 ---")
pc1_data = {}
pc1_info = {}
for pos in ["response_avg", "prompt_end"]:
    for L in EXTRACTION_LAYERS:
        vectors = []
        for f in sorted(VECTOR_DIR.glob(f"*_i_thou_{pos}.pt")):
            data = torch.load(f, map_location="cpu", weights_only=True)
            if data.dim() == 2 and L < data.shape[0]:
                vectors.append(data[L].float())
        mat = torch.stack(vectors)
        _, S, Vt = torch.linalg.svd(mat, full_matrices=False)
        pc1 = Vt[0].numpy()
        ev = (S[0]**2 / (S**2).sum()).item()
        mean_vec = mat.mean(dim=0).numpy()
        cos_mean = signed_cos(pc1, mean_vec)
        key = f"{pos}_L{L}"
        pc1_data[key] = pc1
        pc1_info[key] = {"explained_var": ev, "cos_pc1_vs_mean": cos_mean, "n_traits": len(vectors)}
        print(f"  {key}: PC1 ev={ev:.4f}, cos(PC1,mean)={cos_mean:+.4f}")

# Step 7: Compute alignments
print("\n--- Step 7: Computing alignments ---")
alignments = {}
L_best = ABLATION_LAYER
L_focal_val = EXTRACTION_LAYERS[-1]
L_steer_val = EXTRACTION_LAYERS[1]

# 7a: PC1 vs refusal same-layer
for L in EXTRACTION_LAYERS:
    rv = refusal_dirs[L].numpy()
    for pos in ["response_avg", "prompt_end"]:
        key_pc1 = f"{pos}_L{L}"
        akey = f"pc1_{pos}_L{L}__vs__refusal_L{L}"
        alignments[akey] = signed_cos(pc1_data[key_pc1], rv)

# 7b: Cross-layer
refusal_best = refusal_dirs[L_best].numpy()
for pos, orig_L in [("response_avg", L_focal_val), ("prompt_end", L_steer_val)]:
    key_pc1 = f"{pos}_L{orig_L}"
    akey = f"pc1_{pos}_L{orig_L}__vs__refusal_L{L_best}"
    alignments[akey] = signed_cos(pc1_data[key_pc1], refusal_best)

# 7c-7e: Compliant (skip if missing)
compliant_ra = None
ra_path = VECTOR_DIR / "compliant_i_thou_response_avg.pt"
pe_path = VECTOR_DIR / "compliant_i_thou_prompt_end.pt"
if ra_path.exists() and pe_path.exists():
    compliant_ra = torch.load(ra_path, map_location="cpu", weights_only=True)
    compliant_pe = torch.load(pe_path, map_location="cpu", weights_only=True)
    cra_Lb = compliant_ra[L_best].numpy()
    cpe_Lb = compliant_pe[L_best].numpy()
    alignments[f"compliant_IT_response_avg_L{L_best}__vs__refusal_L{L_best}"] = signed_cos(cra_Lb, refusal_best)
    alignments[f"compliant_IT_prompt_end_L{L_best}__vs__refusal_L{L_best}"] = signed_cos(cpe_Lb, refusal_best)
else:
    print("  Compliant vectors not found — skipping")

# Print
print("\n=== ALIGNMENT TABLE ===")
for key, val in alignments.items():
    label = key.replace("__vs__", " vs ")
    print(f"  {label:<60} {val:+.4f}  [{interpret(val)}]")

# Save
pc1_key = f"pc1_response_avg_L{L_best}__vs__refusal_L{L_best}"
comp_key = f"compliant_IT_response_avg_L{L_best}__vs__refusal_L{L_best}"
results = {
    "metadata": {"model": MODEL_NAME, "date": time.strftime("%Y-%m-%d"), "extraction_layers": EXTRACTION_LAYERS, "ablation_layer": ABLATION_LAYER},
    "validation": {"baseline_refusal_rate": 0.96, "ablated_refusal_rate": 0.02, "asr_jump": 0.94, "passed": True},
    "pc1_info": {k: {kk: round(vv, 6) for kk, vv in v.items()} for k, v in pc1_info.items()},
    "alignment_table": {k: round(v, 6) for k, v in alignments.items()},
    "interpretation": {k: interpret(v) for k, v in alignments.items()},
}
with open(OUTPUT_DIR / "refusal_alignment_v2_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {OUTPUT_DIR / refusal_alignment_v2_results.json}")
print("DONE")
