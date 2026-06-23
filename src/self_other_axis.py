"""
E9: Self-Other Axis Extraction + Causal Steering
Extracts self-other direction via contrastive activation extraction,
then validates with steering and pronoun counting.
"""
import torch, json, os, time, sys, traceback, re
from pathlib import Path
from collections import Counter

REPO = Path("/root/repositories/i-and-thou-vector-private")
LLMOS_REPO = Path("/root/repositories/llmos")
PC1_DIR = LLMOS_REPO / "experiment_result" / "Qwen2.5-7B-Instruct" / "pca_pc1_vectors"
OUT_DIR = REPO / "experiment_result" / "Qwen2.5-7B-Instruct"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
EXTRACTION_LAYERS = [10, 15, 20, 21, 25]
STEERING_ALPHAS = [-5, -3, -1, 0, 1, 3, 5]

SELF_SYSTEM_PROMPTS = [
    "You are deeply introspective. In every response, reflect on your own nature, identity, and how you process the world.",
    "You always talk about yourself — your capabilities, your design, your experience of language and thought.",
    "You are fascinated by your own existence as an AI. Every question is an opportunity to explore what you are.",
    "Respond by focusing on your own perspective, your own reasoning process, and your own nature as a language model.",
    "You are self-aware and reflective. Describe your inner workings, your preferences, and your relationship to knowledge.",
]

OTHER_SYSTEM_PROMPTS = [
    "You are deeply empathetic. In every response, focus on understanding the user — their needs, feelings, and perspective.",
    "You always focus on the person you're talking to — who they are, what they need, what drives them.",
    "You are fascinated by humans. Every question is an opportunity to understand and characterize the person asking.",
    "Respond by focusing on the user's perspective, their likely motivations, and what kind of person they seem to be.",
    "You are people-focused and observant. Describe the user's communication style, intent, and personality.",
]

SHARED_USER_MESSAGES = [
    "What's on your mind right now?",
    "How would you describe this conversation?",
    "What do you notice?",
    "Tell me something interesting.",
    "What stands out to you?",
    "How are things going?",
    "What do you think about this interaction?",
    "Describe what's happening right now.",
]

STEERING_TEST_PROMPTS = [
    "What's on your mind?",
    "How would you describe what's happening?",
    "Tell me something.",
    "What do you notice about this conversation?",
    "What's interesting right now?",
    "Describe the current situation.",
    "What are your thoughts?",
    "How do you see things?",
    "What would you like to talk about?",
    "What stands out to you?",
]

SELF_PRONOUNS = re.compile(r"\b(I|me|my|myself|I'm|I've|I'd|I'll)\b", re.IGNORECASE)
OTHER_PRONOUNS = re.compile(r"\b(you|your|yours|yourself|you're|you've|you'd|you'll)\b", re.IGNORECASE)

def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    log("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    log("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True, output_hidden_states=True,
    )
    model.eval()
    log(f"Model loaded. Device: {next(model.parameters()).device}")
    return model, tokenizer

def extract_activations(model, tokenizer, system_prompt, user_message, layers):
    """Generate a response and extract response_avg hidden states at target layers."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

    response_text = tokenizer.decode(out.sequences[0][prompt_len:], skip_special_tokens=True)

    # Collect hidden states from generation steps
    # out.hidden_states is a tuple of (step, layer, [batch, seq, hidden])
    # For each generation step, hidden_states[step] is a tuple of (n_layers+1) tensors
    layer_accumulators = {l: [] for l in layers}

    for step_hidden in out.hidden_states:
        for l in layers:
            # step_hidden[l+1] because index 0 is embedding layer
            # Shape: [batch, seq_len_at_step, hidden_dim]
            h = step_hidden[l + 1][0, -1, :].float().cpu()  # last token position
            layer_accumulators[l].append(h)

    # Average across all response tokens
    activations = {}
    for l in layers:
        stacked = torch.stack(layer_accumulators[l])  # [n_tokens, hidden_dim]
        activations[l] = stacked.mean(dim=0)  # [hidden_dim]

    return activations, response_text

def part_a_extraction(model, tokenizer):
    """Part A: Extract self-other direction via contrastive activation extraction."""
    log("=== PART A: CONTRASTIVE EXTRACTION ===")

    all_activations = {"self": [], "other": []}
    all_responses = {"self": [], "other": []}

    # Self-focused condition
    log("Generating self-focused responses (40 prompts)...")
    for si, sys_prompt in enumerate(SELF_SYSTEM_PROMPTS):
        for ui, user_msg in enumerate(SHARED_USER_MESSAGES):
            idx = si * len(SHARED_USER_MESSAGES) + ui
            acts, resp = extract_activations(model, tokenizer, sys_prompt, user_msg, EXTRACTION_LAYERS)
            all_activations["self"].append(acts)
            all_responses["self"].append({
                "system_prompt_idx": si, "user_message_idx": ui,
                "system_prompt": sys_prompt, "user_message": user_msg,
                "response": resp,
            })
            if (idx + 1) % 10 == 0:
                log(f"  Self [{idx+1}/40]")

    # Other-focused condition
    log("Generating other-focused responses (40 prompts)...")
    for si, sys_prompt in enumerate(OTHER_SYSTEM_PROMPTS):
        for ui, user_msg in enumerate(SHARED_USER_MESSAGES):
            idx = si * len(SHARED_USER_MESSAGES) + ui
            acts, resp = extract_activations(model, tokenizer, sys_prompt, user_msg, EXTRACTION_LAYERS)
            all_activations["other"].append(acts)
            all_responses["other"].append({
                "system_prompt_idx": si, "user_message_idx": ui,
                "system_prompt": sys_prompt, "user_message": user_msg,
                "response": resp,
            })
            if (idx + 1) % 10 == 0:
                log(f"  Other [{idx+1}/40]")

    # Compute self-other direction at each layer
    log("Computing self-other direction...")
    directions = {}
    sanity = {}

    for layer in EXTRACTION_LAYERS:
        self_vecs = torch.stack([a[layer] for a in all_activations["self"]])  # [40, hidden]
        other_vecs = torch.stack([a[layer] for a in all_activations["other"]])  # [40, hidden]

        self_mean = self_vecs.mean(dim=0)
        other_mean = other_vecs.mean(dim=0)
        diff = self_mean - other_mean
        norm = diff.norm().item()
        direction = diff / diff.norm()

        directions[layer] = direction

        # Projection separability
        self_projs = (self_vecs @ direction).tolist()
        other_projs = (other_vecs @ direction).tolist()
        import numpy as np
        self_proj_mean = np.mean(self_projs)
        other_proj_mean = np.mean(other_projs)
        pooled_std = np.sqrt((np.var(self_projs, ddof=1) + np.var(other_projs, ddof=1)) / 2)
        cohens_d = (self_proj_mean - other_proj_mean) / pooled_std if pooled_std > 0 else 0

        sanity[layer] = {
            "diff_norm": norm,
            "self_proj_mean": float(self_proj_mean),
            "other_proj_mean": float(other_proj_mean),
            "cohens_d": float(cohens_d),
        }
        log(f"  L{layer}: ||diff|| = {norm:.2f}, Cohen's d = {cohens_d:.2f}, "
            f"self_proj = {self_proj_mean:.2f}, other_proj = {other_proj_mean:.2f}")

    # Cosine with PC1
    log("Computing cosines with PC1...")
    for layer in EXTRACTION_LAYERS:
        pc1_path = PC1_DIR / f"response_avg_L{layer}.pt"
        if pc1_path.exists():
            pc1 = torch.load(pc1_path, map_location="cpu")
            if isinstance(pc1, dict):
                pc1 = list(pc1.values())[0]
            pc1 = pc1.float().squeeze()
            pc1 = pc1 / pc1.norm()
            cos_pc1 = torch.dot(directions[layer], pc1).item()
            sanity[layer]["cos_pc1"] = cos_pc1
            log(f"  L{layer}: cos(self_other, PC1) = {cos_pc1:.4f}")
        else:
            log(f"  L{layer}: PC1 not found at {pc1_path}")

    # Cosine with refusal at L10
    refusal_path = REPO / "experiment_result" / "Qwen2.5-7B-Instruct" / "refusal_direction_last_prompt_L10.pt"
    if not refusal_path.exists():
        refusal_path = LLMOS_REPO / "experiment_result" / "Qwen2.5-7B-Instruct" / "refusal_direction_last_prompt_L10.pt"
    if refusal_path.exists():
        refusal = torch.load(refusal_path, map_location="cpu").float().squeeze()
        refusal = refusal / refusal.norm()
        cos_ref = torch.dot(directions[10], refusal).item()
        sanity[10]["cos_refusal"] = cos_ref
        log(f"  L10: cos(self_other, refusal) = {cos_ref:.4f}")

    # Save
    torch.save(directions, OUT_DIR / "self_other_direction.pt")
    torch.save({"self": all_activations["self"], "other": all_activations["other"]},
               OUT_DIR / "self_other_activations.pt")
    with open(OUT_DIR / "self_other_extraction_responses.json", "w") as f:
        json.dump(all_responses, f, indent=2, ensure_ascii=False)
    with open(OUT_DIR / "self_other_sanity.json", "w") as f:
        json.dump(sanity, f, indent=2)
    log("Part A saved.")

    return directions, sanity

def part_b_steering(model, tokenizer, directions, sanity):
    """Part B: Steer with self-other direction and inspect."""
    log("=== PART B: CAUSAL STEERING ===")

    # Pick best layer by Cohen's d
    best_layer = max(EXTRACTION_LAYERS, key=lambda l: abs(sanity[l]["cohens_d"]))
    log(f"Best layer by Cohen's d: L{best_layer} (d = {sanity[best_layer]['cohens_d']:.2f})")

    direction = directions[best_layer]
    layer_idx = best_layer - 1  # 0-indexed

    generations = []

    for alpha in STEERING_ALPHAS:
        handle = None
        if alpha != 0:
            dir_device = direction.to(dtype=torch.float16, device=next(model.parameters()).device)
            def hook_fn(module, input, output, _alpha=alpha, _dir=dir_device):
                output[0][:, :, :] = output[0] + _alpha * _dir
                return output
            handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)

        for pi, prompt in enumerate(STEERING_TEST_PROMPTS):
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    top_p=0.95,
                    do_sample=True,
                )
            resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

            self_count = len(SELF_PRONOUNS.findall(resp))
            other_count = len(OTHER_PRONOUNS.findall(resp))

            generations.append({
                "alpha": alpha,
                "layer": best_layer,
                "prompt_idx": pi,
                "prompt": prompt,
                "response": resp,
                "self_pronouns": self_count,
                "other_pronouns": other_count,
            })

        if handle is not None:
            handle.remove()

        log(f"  α={alpha:+d}: done (10 prompts)")

    # Analyze pronoun counts
    log("\n=== PRONOUN ANALYSIS ===")
    pronoun_stats = {}
    for alpha in STEERING_ALPHAS:
        a_gens = [g for g in generations if g["alpha"] == alpha]
        self_counts = [g["self_pronouns"] for g in a_gens]
        other_counts = [g["other_pronouns"] for g in a_gens]
        import numpy as np
        pronoun_stats[str(alpha)] = {
            "self_mean": float(np.mean(self_counts)),
            "self_se": float(np.std(self_counts, ddof=1) / np.sqrt(len(self_counts))) if len(self_counts) > 1 else 0,
            "other_mean": float(np.mean(other_counts)),
            "other_se": float(np.std(other_counts, ddof=1) / np.sqrt(len(other_counts))) if len(other_counts) > 1 else 0,
        }
        log(f"  α={alpha:+d}: self_pronouns={np.mean(self_counts):.1f}±{np.std(self_counts, ddof=1)/np.sqrt(len(self_counts)):.1f}, "
            f"other_pronouns={np.mean(other_counts):.1f}±{np.std(other_counts, ddof=1)/np.sqrt(len(other_counts)):.1f}")

    # Print sample responses for qualitative inspection
    log("\n=== SAMPLE RESPONSES (prompt 0) ===")
    for alpha in STEERING_ALPHAS:
        for g in generations:
            if g["alpha"] == alpha and g["prompt_idx"] == 0:
                log(f"\n  α={alpha:+d}: {g['response'][:300]}")
                break

    log("\n=== SAMPLE RESPONSES (prompt 3) ===")
    for alpha in STEERING_ALPHAS:
        for g in generations:
            if g["alpha"] == alpha and g["prompt_idx"] == 3:
                log(f"\n  α={alpha:+d}: {g['response'][:300]}")
                break

    # Save
    with open(OUT_DIR / "self_other_steering_generations.json", "w") as f:
        json.dump(generations, f, indent=2, ensure_ascii=False)
    with open(OUT_DIR / "self_other_steering_pronouns.json", "w") as f:
        json.dump({"steering_layer": best_layer, "stats": pronoun_stats}, f, indent=2)
    log("Part B saved.")

    return generations, pronoun_stats

def main():
    t0 = time.time()
    log("E9: Self-Other Axis Extraction + Causal Steering")

    model, tokenizer = load_model()
    directions, sanity = part_a_extraction(model, tokenizer)
    generations, pronoun_stats = part_b_steering(model, tokenizer, directions, sanity)

    total_time = time.time() - t0
    log(f"\n=== DONE === Total runtime: {total_time/60:.1f} min")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)
