#!/usr/bin/env python3
"""
Batch pipeline for running Stage 1 (generate+score) and Stage 2 (extract)
for multiple traits sequentially.

Each stage is run as a subprocess to ensure GPU memory is freed between runs.

Usage:
    python scripts/run_batch_pipeline.py \
        --traits vindictive,ironic,sardonic \
        --model Qwen/Qwen2.5-7B-Instruct \
        --stages 1,2
"""

import subprocess
import sys
import time
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def run_stage1(model: str, trait: str, samples: int = 40):
    """Run generate_combined_responses.py as subprocess."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "generate_combined_responses.py"),
        "--model", model,
        "--trait", trait,
        "--output_dir", "data/responses",
        "--samples_per_instruction", str(samples),
        "--max_new_tokens", "300",
        "--temperature", "1.0",
    ]
    print(f"\n{'='*60}")
    print(f"STAGE 1: {trait}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAIL (exit {result.returncode})"
    print(f"  Stage 1 {trait}: {status} ({elapsed:.0f}s)")
    return result.returncode == 0


def run_stage2(model: str, trait: str):
    """Run extract_vectors.py as subprocess."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "extract_vectors.py"),
        "--model", model,
        "--trait", trait,
        "--responses_dir", "data/responses",
        "--output_dir", "data/vectors",
    ]
    print(f"\n{'='*60}")
    print(f"STAGE 2: {trait}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAIL (exit {result.returncode})"
    print(f"  Stage 2 {trait}: {status} ({elapsed:.0f}s)")
    return result.returncode == 0


def main(
    traits: str = "",
    model: str = "Qwen/Qwen2.5-7B-Instruct",
    stages: str = "1,2",
    samples: int = 40,
):
    if isinstance(traits, (list, tuple)):
        trait_list = list(traits)
    else:
        trait_list = [t.strip() for t in traits.split(",")]

    if isinstance(stages, (list, tuple)):
        stage_list = [int(s) for s in stages]
    else:
        stage_list = [int(s.strip()) for s in stages.split(",")]

    print(f"Batch Pipeline")
    print(f"  Model: {model}")
    print(f"  Traits: {trait_list}")
    print(f"  Stages: {stage_list}")
    print(f"  Samples: {samples}")

    results = {}
    total_start = time.time()

    for trait in trait_list:
        results[trait] = {}

        if 1 in stage_list:
            ok = run_stage1(model, trait, samples)
            results[trait]["stage1"] = "OK" if ok else "FAIL"
            if not ok:
                print(f"  SKIPPING Stage 2 for {trait} (Stage 1 failed)")
                results[trait]["stage2"] = "SKIPPED"
                continue

        if 2 in stage_list:
            ok = run_stage2(model, trait)
            results[trait]["stage2"] = "OK" if ok else "FAIL"

    total_elapsed = time.time() - total_start

    print(f"\n{'='*60}")
    print(f"BATCH PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print()
    for trait, status in results.items():
        s1 = status.get("stage1", "-")
        s2 = status.get("stage2", "-")
        print(f"  {trait:25s} Stage1={s1:6s} Stage2={s2:6s}")

    out_path = Path("data") / "batch_pipeline_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
