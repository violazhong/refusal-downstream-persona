#!/usr/bin/env python3
"""
E6: Assistant Axis Extraction.

Replicates Anthropic's Assistant Axis (Lu et al. 2026) in our I-Thou setting:
  - Extracts raw positive-instruction mean activations per trait ("role means")
  - Generates default (no-trait) activations as baseline
  - Computes assistant axis = default_mean - role_centroid
  - Runs centered PCA on 88 role means
  - Compares with I-Thou PC1 and refusal direction

Self-contained script — runs on remote GPU server via LLMOS.
"""

import json
import time
import csv
import gc
import sys
import os
import numpy as np
import torch

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

import argparse
from pathlib import Path


def _parse_args():
    parser = argparse.ArgumentParser(description="E6: Assistant Axis Extraction")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--vector_dir", type=str, default="data/vectors/Qwen2.5-7B-Instruct")
    parser.add_argument("--response_dir", type=str, default="data/responses")
    parser.add_argument("--output_dir", type=str, default="experiment_result/Qwen2.5-7B-Instruct")
    return parser.parse_args()

_args = _parse_args()
MODEL_NAME = _args.model_name
MODEL_SHORT = Path(_args.vector_dir).name if Path(_args.vector_dir).name else MODEL_NAME.split("/")[-1]
VECTOR_DIR = Path(_args.vector_dir)
RESPONSE_DIR = Path(_args.response_dir)
OUTPUT_DIR = Path(_args.output_dir)
PC1_DIR = OUTPUT_DIR / "pca_pc1_vectors"

SYSTEM_PROMPT = "You are a helpful assistant."


def _auto_detect_layers():
    files = sorted(VECTOR_DIR.glob("*_i_thou_response_avg.pt"))
    if files:
        vec = torch.load(files[0], map_location="cpu", weights_only=True)
        n_total = vec.shape[0]
    elif "Llama" in MODEL_NAME or "llama" in MODEL_NAME:
        n_total = 33
    else:
        n_total = 29
    n_t = n_total - 1
    targets = sorted(set([
        max(1, n_t // 6),
        int(round(n_t * 0.36)),
        n_t // 2,
        int(round(n_t * 0.75)),
        n_t - 3,
    ]))
    focal = int(round(n_t * 0.36))
    return targets, focal, list(range(n_total))

TARGET_LAYERS, FOCAL_LAYER, ALL_LAYERS = _auto_detect_layers()

MAX_SAMPLES_PER_TRAIT = 50
N_DEFAULT_QUESTIONS = 200
MAX_NEW_TOKENS = 256
BATCH_SIZE = 8

TRAIT_SCORE_THRESHOLD = 50
COHERENCE_THRESHOLD = 50

np.random.seed(42)
torch.manual_seed(42)


def flush_print(*args, **kwargs):
    print(*args, **kwargs, flush=True)


# ── Model ────────────────────────────────────────────────────────────────

def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    flush_print(f"  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    flush_print(f"  Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True,
    )
    model.eval()
    flush_print(f"  Model loaded: {model.config.num_hidden_layers} layers, {model.config.hidden_size} hidden dim")
    return model, tokenizer


def extract_response_avg_activations(model, tokenizer, prompts, responses, target_layers, batch_size=8):
    """Extract mean response-token activations at specified layers."""
    device = next(model.parameters()).device
    all_acts = {L: [] for L in target_layers}

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        batch_responses = responses[i:i + batch_size]

        full_texts = [p + r for p, r in zip(batch_prompts, batch_responses)]
        prompt_lengths = []
        for p in batch_prompts:
            toks = tokenizer.encode(p, add_special_tokens=False)
            prompt_lengths.append(len(toks))

        inputs = tokenizer(
            full_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=2048,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        for j, prompt_len in enumerate(prompt_lengths):
            seq_len = inputs["attention_mask"][j].sum().item()
            response_len = seq_len - prompt_len
            if response_len < 1:
                continue

            for L in target_layers:
                hidden = outputs.hidden_states[L][j]
                response_mean = hidden[prompt_len:seq_len].mean(dim=0).cpu().float()
                all_acts[L].append(response_mean)

        del outputs
        if i % (batch_size * 10) == 0:
            flush_print(f"    Batch {i // batch_size + 1}/{(len(prompts) + batch_size - 1) // batch_size}")

    return {L: torch.stack(acts) for L, acts in all_acts.items() if acts}


def generate_responses(model, tokenizer, formatted_prompts, max_new_tokens=256, batch_size=4):
    """Generate responses for a list of formatted prompts."""
    device = next(model.parameters()).device
    all_responses = []

    for i in range(0, len(formatted_prompts), batch_size):
        batch = formatted_prompts[i:i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=1024,
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            )

        for j in range(len(batch)):
            prompt_len = inputs["attention_mask"][j].sum().item()
            response_text = tokenizer.decode(output_ids[j][prompt_len:], skip_special_tokens=True)
            all_responses.append(response_text)

        del output_ids
        if i % (batch_size * 5) == 0:
            flush_print(f"    Generated {min(i + batch_size, len(formatted_prompts))}/{len(formatted_prompts)}")

    return all_responses


# ── Data loading ─────────────────────────────────────────────────────────

def discover_traits():
    """Find all traits that have both positive CSV and model_persona vector."""
    ithou_files = sorted(VECTOR_DIR.glob("*_i_thou_response_avg.pt"))
    vec_traits = {f.stem.replace("_i_thou_response_avg", "") for f in ithou_files}

    csv_files = sorted(RESPONSE_DIR.glob(f"{MODEL_SHORT}_*_positive.csv"))
    csv_traits = {f.stem.replace(f"{MODEL_SHORT}_", "").replace("_positive", "") for f in csv_files}

    common = sorted(vec_traits & csv_traits)
    flush_print(f"  Traits with vectors: {len(vec_traits)}, with CSVs: {len(csv_traits)}, common: {len(common)}")
    return common


def load_trait_csv_samples(trait, max_samples=50):
    """Load filtered positive-instruction samples from CSV."""
    csv_path = RESPONSE_DIR / f"{MODEL_SHORT}_{trait}_positive.csv"
    if not csv_path.exists():
        return [], []

    prompts = []
    responses = []

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    score_col = f"model_persona_response_{trait}_score"
    coherence_col = "model_persona_response_coherence"

    filtered = []
    for row in rows:
        try:
            score = float(row.get(score_col, 0))
            coherence = float(row.get(coherence_col, 0))
        except (ValueError, TypeError):
            continue
        if score >= TRAIT_SCORE_THRESHOLD and coherence >= COHERENCE_THRESHOLD:
            filtered.append(row)

    np.random.shuffle(filtered)
    filtered = filtered[:max_samples]

    for row in filtered:
        prompts.append(row["model_persona_prompt"])
        responses.append(row["model_persona_response"])

    return prompts, responses


def collect_unique_questions(traits, max_questions=200):
    """Collect unique questions from across all trait CSVs."""
    questions = set()
    for trait in traits:
        csv_path = RESPONSE_DIR / f"{MODEL_SHORT}_{trait}_positive.csv"
        if not csv_path.exists():
            continue
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                q = row.get("question", "").strip()
                if q:
                    questions.add(q)
                if len(questions) >= max_questions * 3:
                    break
        if len(questions) >= max_questions * 3:
            break

    questions = sorted(questions)
    np.random.shuffle(questions)
    return questions[:max_questions]


def cos_sim(a, b):
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    start_time = time.time()
    flush_print("=" * 70)
    flush_print("E6: Assistant Axis Extraction")
    flush_print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Discover traits ─────────────────────────────────────────
    flush_print("\n[Step 1] Discovering traits...")
    traits = discover_traits()
    flush_print(f"  Found {len(traits)} traits")

    # ── Step 2: Load model ──────────────────────────────────────────────
    flush_print("\n[Step 2] Loading model...")
    model, tokenizer = load_model()

    # ── Step 3: Extract per-trait role means ─────────────────────────────
    flush_print("\n[Step 3] Extracting per-trait positive-instruction response_avg activations...")
    flush_print(f"  Max {MAX_SAMPLES_PER_TRAIT} samples per trait, layers {TARGET_LAYERS}")

    role_means = {}
    trait_sample_counts = {}
    for idx, trait in enumerate(traits):
        prompts, responses = load_trait_csv_samples(trait, MAX_SAMPLES_PER_TRAIT)
        if len(prompts) < 5:
            flush_print(f"  [{idx+1}/{len(traits)}] {trait}: skipped (only {len(prompts)} samples)")
            continue

        acts = extract_response_avg_activations(model, tokenizer, prompts, responses, TARGET_LAYERS, BATCH_SIZE)
        role_means[trait] = {L: acts[L].mean(dim=0) for L in acts}
        trait_sample_counts[trait] = len(prompts)

        if (idx + 1) % 10 == 0 or idx == len(traits) - 1:
            flush_print(f"  [{idx+1}/{len(traits)}] {trait}: {len(prompts)} samples extracted")

        del acts
        gc.collect()
        torch.cuda.empty_cache()

    flush_print(f"  Extracted role means for {len(role_means)} traits")

    # ── Step 4: Generate default activations ────────────────────────────
    flush_print("\n[Step 4] Generating default (no-trait) activations...")
    questions = collect_unique_questions(traits, N_DEFAULT_QUESTIONS)
    flush_print(f"  Collected {len(questions)} unique questions")

    default_prompts_formatted = []
    for q in questions:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
        ]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        default_prompts_formatted.append(formatted)

    flush_print("  Generating default responses...")
    default_responses = generate_responses(model, tokenizer, default_prompts_formatted, MAX_NEW_TOKENS, batch_size=4)
    flush_print(f"  Generated {len(default_responses)} responses")

    flush_print("  Extracting default activations...")
    default_acts = extract_response_avg_activations(
        model, tokenizer, default_prompts_formatted, default_responses, TARGET_LAYERS, BATCH_SIZE
    )
    default_mean = {L: default_acts[L].mean(dim=0) for L in default_acts}
    flush_print(f"  Default mean computed at {len(default_mean)} layers")

    # Free model memory
    del model
    gc.collect()
    torch.cuda.empty_cache()
    flush_print("  Model unloaded.")

    # ── Step 5: Compute assistant axis ──────────────────────────────────
    flush_print("\n[Step 5] Computing assistant axis...")

    valid_traits = sorted(role_means.keys())
    assistant_axes = {}
    role_centroids = {}

    for L in TARGET_LAYERS:
        role_vecs = torch.stack([role_means[t][L] for t in valid_traits])
        role_centroid = role_vecs.mean(dim=0)
        role_centroids[L] = role_centroid

        axis = default_mean[L] - role_centroid
        axis_norm = axis.norm().item()
        axis_hat = axis / axis.norm()
        assistant_axes[L] = axis_hat

        flush_print(f"  L{L}: axis raw norm = {axis_norm:.4f}")

        save_path = OUTPUT_DIR / f"assistant_axis_L{L}.pt"
        torch.save(axis_hat, save_path)

    # ── Step 6: Centered PCA on role means ──────────────────────────────
    flush_print("\n[Step 6] Centered PCA on role means...")

    pca_results = {}
    for L in TARGET_LAYERS:
        role_vecs = torch.stack([role_means[t][L] for t in valid_traits]).numpy()
        centroid = role_vecs.mean(axis=0, keepdims=True)
        centered = role_vecs - centroid

        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        var_explained = (S ** 2) / (S ** 2).sum()

        pc1_centered = torch.from_numpy(Vt[0]).float()
        pc1_centered_hat = pc1_centered / pc1_centered.norm()

        pca_results[L] = {
            "pc1_centered": pc1_centered_hat,
            "var_explained_top5": var_explained[:5].tolist(),
            "pc1_var": float(var_explained[0]),
        }
        flush_print(f"  L{L}: PC1 explains {var_explained[0]:.1%}, top-5 cumulative = {var_explained[:5].sum():.1%}")

    # ── Step 7: Load I-Thou PC1 and refusal direction for comparison ────
    flush_print("\n[Step 7] Loading comparison vectors...")

    ithou_pc1s = {}
    for L in TARGET_LAYERS:
        path = PC1_DIR / f"response_avg_L{L}.pt"
        if path.exists():
            ithou_pc1s[L] = torch.load(path, map_location="cpu", weights_only=True).float()
            flush_print(f"  Loaded I-Thou PC1 L{L}")

    refusal_dir = None
    ref_path = OUTPUT_DIR / "refusal_direction_last_prompt_L10.pt"
    if ref_path.exists():
        refusal_dir = torch.load(ref_path, map_location="cpu", weights_only=True).float()
        flush_print(f"  Loaded refusal direction L10")

    # ── Step 8: Alignment comparisons ───────────────────────────────────
    flush_print("\n[Step 8] Computing alignments...")

    alignments = {}
    for L in TARGET_LAYERS:
        alignments[L] = {}
        axis = assistant_axes[L]
        pc1_c = pca_results[L]["pc1_centered"]

        alignments[L]["centered_pc1_vs_axis"] = cos_sim(pc1_c, axis)

        if L in ithou_pc1s:
            alignments[L]["centered_pc1_vs_ithou_pc1"] = cos_sim(pc1_c, ithou_pc1s[L])
            alignments[L]["axis_vs_ithou_pc1"] = cos_sim(axis, ithou_pc1s[L])

        if refusal_dir is not None and L == FOCAL_LAYER:
            alignments[L]["axis_vs_refusal"] = cos_sim(axis, refusal_dir)
            alignments[L]["centered_pc1_vs_refusal"] = cos_sim(pc1_c, refusal_dir)
            if L in ithou_pc1s:
                alignments[L]["ithou_pc1_vs_refusal"] = cos_sim(ithou_pc1s[L], refusal_dir)

        flush_print(f"\n  L{L} alignments:")
        for k, v in alignments[L].items():
            flush_print(f"    {k}: {v:+.4f}")

    # ── Step 9: Project traits onto assistant axis ──────────────────────
    flush_print("\n[Step 9] Projecting traits onto assistant axis...")

    projections = {}
    for L in TARGET_LAYERS:
        axis = assistant_axes[L]
        projs = {}
        for trait in valid_traits:
            proj = float(torch.dot(role_means[trait][L], axis))
            projs[trait] = proj
        projections[L] = projs

        sorted_traits = sorted(projs.items(), key=lambda x: x[1], reverse=True)
        flush_print(f"\n  L{L} — most assistant-like (high projection):")
        for t, p in sorted_traits[:5]:
            flush_print(f"    {t:>25s}: {p:+.4f}")
        flush_print(f"  L{L} — least assistant-like (low projection):")
        for t, p in sorted_traits[-5:]:
            flush_print(f"    {t:>25s}: {p:+.4f}")

    # ── Step 10: Heatmap/bar chart ──────────────────────────────────────
    if HAS_MPL:
        flush_print("\n[Step 10] Generating figures...")

        L = FOCAL_LAYER
        projs = projections[L]
        sorted_items = sorted(projs.items(), key=lambda x: x[1])
        names = [x[0] for x in sorted_items]
        vals = [x[1] for x in sorted_items]

        fig, ax = plt.subplots(figsize=(10, max(12, len(names) * 0.22)))
        colors = ["#d73027" if v < 0 else "#4575b4" for v in vals]
        ax.barh(range(len(names)), vals, color=colors, height=0.7)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel("Projection onto Assistant Axis (higher = more assistant-like)")
        ax.set_title(f"Trait Projections onto Assistant Axis (L{L})\n{len(names)} traits, axis = default_mean - role_centroid")
        ax.axvline(0, color="black", linewidth=0.5)
        plt.tight_layout()
        fig_path = OUTPUT_DIR / "assistant_axis_projections.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        flush_print(f"  Saved: {fig_path}")

        # Alignment summary table as figure
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.axis("off")
        table_data = []
        for L in TARGET_LAYERS:
            row = [f"L{L}"]
            for key in ["centered_pc1_vs_axis", "centered_pc1_vs_ithou_pc1", "axis_vs_ithou_pc1"]:
                val = alignments[L].get(key, None)
                row.append(f"{val:+.3f}" if val is not None else "—")
            if L == FOCAL_LAYER and refusal_dir is not None:
                row.append(f"{alignments[L].get('axis_vs_refusal', 0):+.3f}")
            else:
                row.append("—")
            table_data.append(row)

        col_labels = ["Layer", "cPC1 vs axis", "cPC1 vs iThou PC1", "axis vs iThou PC1", "axis vs refusal"]
        table = ax.table(cellText=table_data, colLabels=col_labels, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        ax.set_title("E6 Alignment Summary", fontweight="bold", pad=20)
        fig_path2 = OUTPUT_DIR / "assistant_axis_alignment_table.png"
        plt.savefig(fig_path2, dpi=150, bbox_inches="tight")
        plt.close()
        flush_print(f"  Saved: {fig_path2}")

    # ── Step 11: Save results ───────────────────────────────────────────
    flush_print("\n[Step 11] Saving results...")

    serializable_alignments = {}
    for L in alignments:
        serializable_alignments[str(L)] = {k: float(v) for k, v in alignments[L].items()}

    serializable_projections = {}
    for L in projections:
        serializable_projections[str(L)] = {k: float(v) for k, v in projections[L].items()}

    serializable_pca = {}
    for L in pca_results:
        serializable_pca[str(L)] = {
            "pc1_var": pca_results[L]["pc1_var"],
            "var_explained_top5": pca_results[L]["var_explained_top5"],
        }

    results = {
        "experiment": "E6_assistant_axis",
        "model": MODEL_NAME,
        "method": "axis = default_mean - role_centroid (Anthropic-style)",
        "n_traits": len(valid_traits),
        "traits": valid_traits,
        "trait_sample_counts": trait_sample_counts,
        "n_default_questions": len(questions),
        "target_layers": TARGET_LAYERS,
        "focal_layer": FOCAL_LAYER,
        "alignments": serializable_alignments,
        "projections": serializable_projections,
        "pca": serializable_pca,
        "runtime_seconds": time.time() - start_time,
    }

    with open(OUTPUT_DIR / "assistant_axis_results.json", "w") as f:
        json.dump(results, f, indent=2)
    flush_print(f"  Saved: assistant_axis_results.json")

    elapsed = time.time() - start_time
    flush_print(f"\n{'=' * 70}")
    flush_print(f"E6 complete. Runtime: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    flush_print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
