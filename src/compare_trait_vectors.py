#!/usr/bin/env python3
import torch
import torch.nn.functional as F
from pathlib import Path

def compare(trait1, trait2, vector_dir="data/vectors/Qwen2.5-7B-Instruct", positions=None):
    if positions is None:
        positions = ["model_persona_response_avg", "i_thou_response_avg"]
    else:
        positions = positions.split(",")
    vdir = Path(vector_dir)
    sep = "=" * 70
    print()
    print(sep)
    print("COMPARING: " + trait1 + " vs " + trait2)
    print(sep)
    for pos in positions:
        f1 = vdir / (trait1 + "_" + pos + ".pt")
        f2 = vdir / (trait2 + "_" + pos + ".pt")
        if not f1.exists() or not f2.exists():
            print("  " + pos + ": MISSING")
            continue
        v1 = torch.load(f1, map_location="cpu").float()
        v2 = torch.load(f2, map_location="cpu").float()
        print()
        print("  " + pos + ":")
        sims = []
        for layer in range(v1.shape[0]):
            sim = F.cosine_similarity(v1[layer].unsqueeze(0), v2[layer].unsqueeze(0)).item()
            sims.append(sim)
            bar = chr(9608) * int(abs(sim) * 20)
            print("    Layer %2d: %+.4f %s" % (layer, sim, bar))
        avg = sum(sims) / len(sims)
        mx = max(sims)
        ml = sims.index(mx)
        print("    --- Avg: %+.4f, Max: %+.4f (L%d)" % (avg, mx, ml))
    print()
    print(sep)

if __name__ == "__main__":
    import fire
    fire.Fire(compare)
