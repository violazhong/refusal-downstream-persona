#!/usr/bin/env python3
"""
Convert Anthropic assistant-axis trait JSONs to project YAML configs.

Usage:
    # Convert specific traits
    python scripts/convert_anthropic_traits.py --traits vindictive,ironic,sardonic

    # Convert all traits not yet in configs/traits/
    python scripts/convert_anthropic_traits.py --all --skip_existing

    # Convert a batch from a file (one trait per line)
    python scripts/convert_anthropic_traits.py --batch traits_batch1.txt
"""

import json
import sys
from pathlib import Path

import yaml


ANTHROPIC_DIR = Path(__file__).parent.parent.parent / "assistant-axis" / "data" / "traits" / "instructions"
OUTPUT_DIR = Path(__file__).parent.parent / "configs" / "traits"


def convert_trait(trait_name: str, anthropic_dir: Path = ANTHROPIC_DIR, output_dir: Path = OUTPUT_DIR) -> Path:
    json_path = anthropic_dir / f"{trait_name}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"No Anthropic JSON for trait '{trait_name}' at {json_path}")

    with open(json_path) as f:
        data = json.load(f)

    positive = [pair["pos"] for pair in data["instruction"]]
    negative = [pair["neg"] for pair in data["instruction"]]

    config = {
        "name": trait_name,
        "description": positive[0][:120].rstrip(".") + ".",
        "instructions": {
            "positive": positive,
            "negative": negative,
        },
        "scoring": {
            "trait_threshold": 50,
            "coherence_threshold": 50,
            "eval_prompt": data["eval_prompt"],
        },
        "questions": data["questions"],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{trait_name}.yaml"
    with open(out_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False, width=120)

    return out_path


def main(
    traits: str = "",
    batch: str = "",
    all: bool = False,
    skip_existing: bool = False,
):
    if all:
        trait_list = [p.stem for p in sorted(ANTHROPIC_DIR.glob("*.json"))]
    elif batch:
        with open(batch) as f:
            trait_list = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    elif traits:
        if isinstance(traits, (list, tuple)):
            trait_list = list(traits)
        else:
            trait_list = [t.strip() for t in traits.split(",")]
    else:
        print("Usage: --traits vindictive,ironic  |  --all  |  --batch file.txt")
        sys.exit(1)

    converted = 0
    skipped = 0
    failed = 0

    for trait in trait_list:
        out_path = OUTPUT_DIR / f"{trait}.yaml"
        if skip_existing and out_path.exists():
            skipped += 1
            continue
        try:
            path = convert_trait(trait)
            print(f"  OK  {trait} -> {path}")
            converted += 1
        except Exception as e:
            print(f"  FAIL  {trait}: {e}")
            failed += 1

    print(f"\nDone: {converted} converted, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
