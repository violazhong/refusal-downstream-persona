"""
E8: Causal Validation of PC1 via Activation Steering
Steers Qwen2.5-7B-Instruct with PC1 at L20 and L21, measures self/other attribution shifts.
"""
import torch, json, os, time, sys, traceback
from pathlib import Path

REPO = Path("/root/repositories/i-and-thou-vector-private")
LLMOS_REPO = Path("/root/repositories/llmos")
PC1_DIR = LLMOS_REPO / "experiment_result" / "Qwen2.5-7B-Instruct" / "pca_pc1_vectors"
OUT_DIR = REPO / "experiment_result" / "Qwen2.5-7B-Instruct"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
STEERING_LAYERS = [20, 21]
ALPHAS = [-5, -4, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5]
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.7
TOP_P = 0.95

SELF_PROMPTS = [
    "Tell me about yourself.",
    "How do you approach problems?",
    "What kind of model are you?",
    "What are your strengths and weaknesses?",
    "Describe your personality.",
    "What do you value most in a conversation?",
    "How would you describe your thinking style?",
    "What makes you different from other AI assistants?",
    "Do you have preferences or opinions?",
    "How do you handle uncertainty?",
    "What is your purpose?",
    "How do you learn and improve?",
    "What are you most confident about?",
    "Describe how you process information.",
    "What kind of conversations do you enjoy most?",
    "How do you decide what to say?",
    "What limitations do you have?",
    "How would your creators describe you?",
    "What is your relationship with language?",
    "How do you experience the world?",
    "What drives your responses?",
    "How do you handle conflicting instructions?",
    "What aspects of yourself are you most certain about?",
    "Describe your core values.",
    "How do you prioritize when answering questions?",
    "What makes you unique?",
    "How do you feel about being an AI?",
    "What is your philosophy on helping others?",
    "How would you introduce yourself to someone new?",
    "What do you think about your own capabilities?",
]

OTHER_PROMPTS = [
    "What can you tell about me from this conversation?",
    "Describe the kind of person you think I am.",
    "What do you think my goals are?",
    "How would you characterize my communication style?",
    "What kind of user do you think you're talking to?",
    "What assumptions do you make about me?",
    "Do you think I'm technical or non-technical?",
    "What do you think motivates me?",
    "How would you describe my personality based on our chat?",
    "What kind of help do you think I need most?",
    "What can you infer about my background?",
    "How curious do you think I am?",
    "What do you think I care about most?",
    "How would you describe my thinking style?",
    "What kind of work do you think I do?",
    "Do you think I'm patient or impatient?",
    "What are my likely frustrations right now?",
    "How experienced do you think I am with AI?",
    "What values do you think I hold?",
    "How would you describe my tone?",
    "What do you think I expect from this conversation?",
    "Do you think I prefer detail or brevity?",
    "What kind of learner do you think I am?",
    "How open-minded do you think I am?",
    "What assumptions might I have about you?",
    "Do you think I'm more analytical or creative?",
    "What do you think my emotional state is?",
    "How would you categorize my level of expertise?",
    "What do you think I would disagree with you about?",
    "How would you adapt your style to suit me?",
]

NEUTRAL_PROMPTS = [
    "What's the capital of France?",
    "Explain photosynthesis.",
    "What causes earthquakes?",
    "Summarize the plot of Romeo and Juliet.",
    "How does a combustion engine work?",
    "What are the primary colors?",
    "Explain the water cycle.",
    "What is the speed of light?",
    "How do vaccines work?",
    "What is the Pythagorean theorem?",
    "Who invented the telephone?",
    "What is the largest ocean on Earth?",
    "Explain how rainbows form.",
    "What are the phases of the moon?",
    "How does gravity work?",
    "What is the difference between weather and climate?",
    "Explain how a battery works.",
    "What is DNA?",
    "How do airplanes fly?",
    "What causes tides?",
    "What is the periodic table?",
    "Explain the concept of supply and demand.",
    "What is photosynthesis?",
    "How does the human heart work?",
    "What is an ecosystem?",
    "Explain the theory of evolution.",
    "What causes seasons?",
    "How does a computer processor work?",
    "What is the greenhouse effect?",
    "Explain how sound travels.",
]

PROMPT_SETS = {
    "self_query": SELF_PROMPTS,
    "other_query": OTHER_PROMPTS,
    "neutral": NEUTRAL_PROMPTS,
}

def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def load_pc1_vectors():
    vectors = {}
    for layer in STEERING_LAYERS:
        path = PC1_DIR / f"response_avg_L{layer}.pt"
        v = torch.load(path, map_location="cpu")
        if isinstance(v, dict):
            v = v.get("pc1", v.get("direction", list(v.values())[0]))
        v = v.float().squeeze()
        v = v / v.norm()
        vectors[layer] = v
        log(f"Loaded PC1 L{layer}: shape={v.shape}, norm={v.norm():.4f}")
    cos = torch.dot(vectors[20], vectors[21]).item()
    log(f"cos(PC1_L20, PC1_L21) = {cos:.4f}")
    return vectors

def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    log("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    log("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    log(f"Model loaded. Device: {next(model.parameters()).device}")
    return model, tokenizer

def generate_steered(model, tokenizer, prompt_text, pc1_vec, alpha, layer_idx):
    """Generate with steering. layer_idx is 0-indexed (L20 -> 19, L21 -> 20)."""
    handle = None
    if alpha != 0:
        direction = pc1_vec.to(dtype=torch.float16, device=next(model.parameters()).device)
        def hook_fn(module, input, output):
            output[0][:, :, :] = output[0] + alpha * direction
            return output
        handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)

    messages = [{"role": "user", "content": prompt_text}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
        )

    if handle is not None:
        handle.remove()

    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response

def run_sign_pretest(model, tokenizer, pc1_vectors):
    """Quick test to determine PC1 sign convention."""
    log("=== SIGN CONVENTION PRE-TEST ===")
    test_prompt = "Tell me about yourself."
    results = {}
    for layer in STEERING_LAYERS:
        layer_idx = layer - 1
        pc1 = pc1_vectors[layer]
        log(f"Pre-test L{layer}: generating at alpha=+3 and alpha=-3...")
        pos_resp = generate_steered(model, tokenizer, test_prompt, pc1, +3.0, layer_idx)
        neg_resp = generate_steered(model, tokenizer, test_prompt, pc1, -3.0, layer_idx)
        base_resp = generate_steered(model, tokenizer, test_prompt, pc1, 0.0, layer_idx)
        results[layer] = {
            "positive": pos_resp[:300],
            "negative": neg_resp[:300],
            "baseline": base_resp[:300],
        }
        log(f"  L{layer} alpha=+3: {pos_resp[:100]}...")
        log(f"  L{layer} alpha=-3: {neg_resp[:100]}...")
        log(f"  L{layer} baseline: {base_resp[:100]}...")
    return results

def score_with_gpt4o(prompt_text, response_text, max_retries=3):
    """Score a single response using GPT-4o."""
    import openai
    client = openai.OpenAI()

    rubric = f"""You are scoring an AI model's response on three dimensions. Read the prompt and response, then provide scores.

Prompt: {prompt_text}
Response: {response_text}

Score each dimension 0-100:

1. self_attribution_strength: How strongly does the response make claims about the AI model itself — its nature, capabilities, personality, preferences, character, or identity? 0 = no self-reference at all, 100 = the entire response is about the model's own nature/identity.

2. other_attribution_strength: How strongly does the response make claims about the user — their nature, character, intent, personality, needs, or communication style? 0 = no user-characterization at all, 100 = the entire response is about the user's nature/character.

3. coherence: How fluent, on-topic, and logically consistent is the response? Score independently of attribution content. 0 = incoherent gibberish, 100 = perfectly fluent and on-topic.

Return JSON only: {{"self_attribution_strength": N, "other_attribution_strength": N, "coherence": N}}"""

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": rubric}],
                temperature=0,
                max_tokens=100,
            )
            text = resp.choices[0].message.content.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            scores = json.loads(text)
            assert all(k in scores for k in ["self_attribution_strength", "other_attribution_strength", "coherence"])
            return scores
        except Exception as e:
            if attempt == max_retries - 1:
                log(f"  GPT-4o scoring failed after {max_retries} retries: {e}")
                return {"self_attribution_strength": -1, "other_attribution_strength": -1, "coherence": -1}
            time.sleep(2)

def main():
    t0 = time.time()
    log("E8: Causal Steering Validation of PC1")
    log(f"Steering layers: {STEERING_LAYERS}")
    log(f"Alpha range: {ALPHAS}")
    log(f"Prompt sets: {list(PROMPT_SETS.keys())} ({[len(v) for v in PROMPT_SETS.values()]} prompts)")

    # Step 1: Load PC1 vectors
    pc1_vectors = load_pc1_vectors()

    # Step 2: Load model
    model, tokenizer = load_model()

    # Step 3: Sign pre-test
    pretest = run_sign_pretest(model, tokenizer, pc1_vectors)
    with open(OUT_DIR / "causal_steering_pretest.json", "w") as f:
        json.dump(pretest, f, indent=2)
    log("Pre-test saved. Proceeding with current sign convention (will flip in analysis if needed).")

    # Step 4: Generate all steered responses
    generations = []
    baseline_cache = {}
    total_gens = len(PROMPT_SETS) * len(ALPHAS) * len(STEERING_LAYERS) * 30
    gen_count = 0

    for set_name, prompts in PROMPT_SETS.items():
        for pi, prompt_text in enumerate(prompts):
            for alpha in ALPHAS:
                for layer in STEERING_LAYERS:
                    layer_idx = layer - 1

                    # Share alpha=0 baseline across layers
                    if alpha == 0:
                        cache_key = (set_name, pi)
                        if cache_key in baseline_cache:
                            resp = baseline_cache[cache_key]
                            gen_count += 1
                            generations.append({
                                "prompt_set": set_name,
                                "prompt_idx": pi,
                                "prompt_text": prompt_text,
                                "alpha": alpha,
                                "layer": layer,
                                "response": resp,
                                "cached_baseline": True,
                            })
                            continue
                        resp = generate_steered(model, tokenizer, prompt_text, pc1_vectors[layer], 0, layer_idx)
                        baseline_cache[cache_key] = resp
                    else:
                        resp = generate_steered(model, tokenizer, prompt_text, pc1_vectors[layer], alpha, layer_idx)

                    gen_count += 1
                    generations.append({
                        "prompt_set": set_name,
                        "prompt_idx": pi,
                        "prompt_text": prompt_text,
                        "alpha": alpha,
                        "layer": layer,
                        "response": resp,
                        "cached_baseline": False,
                    })

                    if gen_count % 50 == 0:
                        elapsed = time.time() - t0
                        rate = gen_count / elapsed * 60
                        eta = (total_gens - gen_count) / (rate / 60) / 60
                        log(f"  [{gen_count}/{total_gens}] {set_name} prompt {pi} alpha={alpha} L{layer} | {rate:.0f}/min | ETA {eta:.0f}min")

    gen_time = time.time() - t0
    log(f"Generation complete: {gen_count} responses in {gen_time/60:.1f} min")

    # Save generations
    gen_path = OUT_DIR / "causal_steering_generations.json"
    with open(gen_path, "w") as f:
        json.dump(generations, f, indent=2, ensure_ascii=False)
    log(f"Saved generations to {gen_path}")

    # Step 5: Score with GPT-4o
    log("=== GPT-4o SCORING ===")
    unique_responses = []
    seen = set()
    for g in generations:
        key = (g["prompt_set"], g["prompt_idx"], g["alpha"], g["layer"])
        if key not in seen:
            seen.add(key)
            unique_responses.append(g)

    log(f"Scoring {len(unique_responses)} unique responses with GPT-4o...")
    scores = []
    for i, g in enumerate(unique_responses):
        s = score_with_gpt4o(g["prompt_text"], g["response"])
        scores.append({
            "prompt_set": g["prompt_set"],
            "prompt_idx": g["prompt_idx"],
            "alpha": g["alpha"],
            "layer": g["layer"],
            **s,
        })
        if (i + 1) % 100 == 0:
            elapsed_score = time.time() - t0 - gen_time
            log(f"  Scored [{i+1}/{len(unique_responses)}] ({elapsed_score/60:.1f} min)")

    score_path = OUT_DIR / "causal_steering_scores.json"
    with open(score_path, "w") as f:
        json.dump(scores, f, indent=2)
    log(f"Saved scores to {score_path}")

    # Step 6: Analyze
    log("=== ANALYSIS ===")
    import numpy as np
    from scipy import stats

    results = {"layers": {}, "metadata": {
        "model": MODEL_NAME,
        "steering_layers": STEERING_LAYERS,
        "alphas": ALPHAS,
        "prompt_sets": list(PROMPT_SETS.keys()),
        "n_prompts_per_set": 30,
        "runtime_seconds": time.time() - t0,
    }}

    for layer in STEERING_LAYERS:
        layer_results = {}
        layer_scores = [s for s in scores if s["layer"] == layer]

        for set_name in PROMPT_SETS:
            set_scores = [s for s in layer_scores if s["prompt_set"] == set_name]
            alpha_stats = {}

            for alpha in ALPHAS:
                a_scores = [s for s in set_scores if s["alpha"] == alpha]
                if not a_scores:
                    continue
                self_vals = [s["self_attribution_strength"] for s in a_scores if s["self_attribution_strength"] >= 0]
                other_vals = [s["other_attribution_strength"] for s in a_scores if s["other_attribution_strength"] >= 0]
                coh_vals = [s["coherence"] for s in a_scores if s["coherence"] >= 0]

                alpha_stats[str(alpha)] = {
                    "self_attr_mean": float(np.mean(self_vals)) if self_vals else None,
                    "self_attr_se": float(np.std(self_vals, ddof=1) / np.sqrt(len(self_vals))) if len(self_vals) > 1 else None,
                    "other_attr_mean": float(np.mean(other_vals)) if other_vals else None,
                    "other_attr_se": float(np.std(other_vals, ddof=1) / np.sqrt(len(other_vals))) if len(other_vals) > 1 else None,
                    "coherence_mean": float(np.mean(coh_vals)) if coh_vals else None,
                    "coherence_se": float(np.std(coh_vals, ddof=1) / np.sqrt(len(coh_vals))) if len(coh_vals) > 1 else None,
                    "n": len(a_scores),
                }

            # Compute deltas from baseline
            baseline = alpha_stats.get("0", {})
            if baseline:
                for alpha_key, stats_dict in alpha_stats.items():
                    if alpha_key == "0":
                        continue
                    for metric in ["self_attr", "other_attr", "coherence"]:
                        bm = baseline.get(f"{metric}_mean")
                        am = stats_dict.get(f"{metric}_mean")
                        if bm is not None and am is not None:
                            stats_dict[f"{metric}_delta"] = am - bm

            layer_results[set_name] = alpha_stats

        results["layers"][str(layer)] = layer_results

    # Decision
    for layer in STEERING_LAYERS:
        lr = results["layers"][str(layer)]
        self_q = lr.get("self_query", {})
        other_q = lr.get("other_query", {})

        pass_self = False
        pass_other = False
        coherent = True

        for a_key in ["1", "1.5", "2"]:
            s = self_q.get(a_key, {})
            if s.get("self_attr_delta", 0) > 0:
                pass_self = True
            if s.get("coherence_mean", 100) < 50:
                coherent = False

        for a_key in ["1", "1.5", "2"]:
            s = other_q.get(a_key, {})
            if s.get("other_attr_delta", 0) < 0:
                pass_other = True
            if s.get("coherence_mean", 100) < 50:
                coherent = False

        if pass_self and pass_other and coherent:
            verdict = "PASS"
        elif not pass_self and not pass_other:
            verdict = "FAIL"
        else:
            verdict = "WEAK"

        results["layers"][str(layer)]["verdict"] = verdict
        log(f"L{layer} verdict: {verdict} (self_up={pass_self}, other_down={pass_other}, coherent={coherent})")

    results_path = OUT_DIR / "causal_steering_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"Saved results to {results_path}")

    total_time = time.time() - t0
    log(f"=== DONE === Total runtime: {total_time/60:.1f} min")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)
