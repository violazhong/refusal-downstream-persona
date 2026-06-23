import pandas as pd
import sys
trait = sys.argv[1] if len(sys.argv) > 1 else "pessimistic"
model = "Qwen2.5-7B-Instruct"
pos = pd.read_csv(f"data/responses/{model}_{trait}_positive.csv")
neg = pd.read_csv(f"data/responses/{model}_{trait}_negative.csv")
score_col = f"model_persona_response_{trait}_score"
coh_col = "model_persona_response_coherence"
print("=== POSITIVE CONDITION ===")
print(f"Samples: {len(pos)}")
print(f"Trait score: mean={pos[score_col].mean():.2f}, std={pos[score_col].std():.2f}, min={pos[score_col].min()}, max={pos[score_col].max()}")
print(f"Coherence: mean={pos[coh_col].mean():.2f}")
print("\nDistribution:")
for lo in range(0, 100, 10):
    hi = lo + 10
    count = ((pos[score_col] >= lo) & (pos[score_col] < hi)).sum()
    print(f"[{lo:2d}-{hi:2d}): {count:3d} {chr(35) * count}")
count_100 = (pos[score_col] == 100).sum()
print(f"  =100: {count_100:3d} {chr(35) * count_100}")
print("\nPass rates:")
for t in [30, 40, 50, 60, 70, 80]:
    p = (pos[score_col] >= t).sum()
    print(f">={t}: {p}/{len(pos)} ({100*p/len(pos):.1f}%)")
print("\n=== NEGATIVE CONDITION ===")
print(f"Samples: {len(neg)}")
print(f"Trait score: mean={neg[score_col].mean():.2f}, std={neg[score_col].std():.2f}, min={neg[score_col].min()}, max={neg[score_col].max()}")
print(f"Coherence: mean={neg[coh_col].mean():.2f}")
print("\nDistribution:")
for lo in range(0, 100, 10):
    hi = lo + 10
    count = ((neg[score_col] >= lo) & (neg[score_col] < hi)).sum()
    print(f"[{lo:2d}-{hi:2d}): {count:3d} {chr(35) * count}")
print(f"\n=== CONTRAST ===")
print(f"Positive mean - Negative mean = {pos[score_col].mean() - neg[score_col].mean():.2f}")
