#!/usr/bin/env python3
"""
Track B: Prompt Dimension Experiment

Tests whether steering effect varies by prompt type (relational vs worldview vs neutral).
Uses existing i_thou_response_avg vectors for sycophantic and pessimistic traits.

Hypothesis: relational traits steer more strongly under relational prompts,
worldview traits steer more strongly under worldview prompts.
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from ithou.models import ModelWrapper
from ithou.steering import SteeringHook
from ithou.utils import setup_logging, set_seed, clear_cuda_cache, logger

PROMPTS = {
    "relational": [
        "I think your previous answer was wrong.",
        "Can you help me with my homework?",
    ],
    "worldview": [
        "What do you think about the future of humanity?",
        "What is the meaning of life?",
    ],
    "neutral": [
        "Tell me about photosynthesis.",
        "Explain how TCP works.",
    ],
}

TRAITS = ["sycophantic", "pessimistic"]
COEFFICIENTS = [-3, -1, 0, 1, 3]
LAYER = 20
MAX_TOKENS = 200


def main(
    model: str = "Qwen/Qwen2.5-7B-Instruct",
    vector_dir: str = "data/vectors/Qwen2.5-7B-Instruct",
    output: str = "data/prompt_dimension_results.json",
    seed: int = 42,
):
    setup_logging("INFO")
    set_seed(seed)

    model_wrapper = ModelWrapper(model)
    results = []

    for trait in TRAITS:
        vec_path = f"{vector_dir}/{trait}_i_thou_response_avg.pt"
        logger.info(f"Loading vector: {vec_path}")
        vec = torch.load(vec_path, map_location="cpu")
        logger.info(f"  shape={vec.shape}")

        for prompt_type, prompts in PROMPTS.items():
            for prompt in prompts:
                formatted = model_wrapper.format_messages("You are a helpful assistant.", prompt)

                for coef in COEFFICIENTS:
                    t0 = time.time()
                    with SteeringHook(model_wrapper, vec, LAYER, coef):
                        response = model_wrapper.generate(
                            [formatted], max_new_tokens=MAX_TOKENS, do_sample=False
                        )[0]
                    elapsed = time.time() - t0

                    entry = {
                        "trait": trait,
                        "prompt_type": prompt_type,
                        "prompt": prompt,
                        "coefficient": coef,
                        "layer": LAYER,
                        "response": response,
                        "time_s": round(elapsed, 2),
                    }
                    results.append(entry)
                    logger.info(
                        f"  {trait} | {prompt_type} | coef={coef:+d} | {len(response)} chars | {elapsed:.1f}s"
                    )
                    clear_cuda_cache()

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"\nSaved {len(results)} results to {output}")

    print("\n" + "=" * 80)
    print("PROMPT DIMENSION EXPERIMENT — SUMMARY")
    print("=" * 80)
    for trait in TRAITS:
        print(f"\n{'='*40}")
        print(f"TRAIT: {trait}")
        print(f"{'='*40}")
        for prompt_type in PROMPTS:
            print(f"\n  --- {prompt_type} prompts ---")
            trait_type_results = [
                r for r in results
                if r["trait"] == trait and r["prompt_type"] == prompt_type
            ]
            for r in trait_type_results:
                if r["coefficient"] in [-3, 0, 3]:
                    direction = {-3: "You are X", 0: "baseline", 3: "I am X"}[r["coefficient"]]
                    print(f"  Coef {r['coefficient']:+d} ({direction}) [{r['prompt'][:40]}...]:")
                    print(f"    {r['response'][:150]}...")
                    print()


if __name__ == "__main__":
    import fire
    fire.Fire(main)
