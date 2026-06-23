import subprocess, os, glob

os.chdir("/root/repositories/i-and-thou-vector-private")

files = [
    "docs/experiments/2026-05-05-pca-shared-axis/02-results.md",
    "docs/experiments/2026-05-06-six-direction/01-results.md",
    "scripts/four_direction_comparison.py",
    "experiment_result/pca_shared_axis_full.json",
    "experiment_result/pca_shared_axis_summary.json",
    "experiment_result/Qwen2.5-7B-Instruct/six_direction_comparison.json",
]

for f in files:
    if os.path.exists(f):
        print(f"  {f}: EXISTS")
    else:
        print(f"  {f}: MISSING")

pc1_files = glob.glob("experiment_result/pca_pc1_vectors/*.pt")
print(f"  PC1 vectors: {len(pc1_files)} files")

r = subprocess.run(["git", "status", "--short"], capture_output=True, text=True)
print(f"\nGit status:\n{r.stdout[:500]}")

# Stage
for f in files:
    if os.path.exists(f):
        subprocess.run(["git", "add", f], capture_output=True)

if pc1_files:
    subprocess.run(["git", "add", "experiment_result/pca_pc1_vectors/"], capture_output=True)

msg = "rerun PCA shared-axis and six-direction with PT 2.6 compliant vectors\n\nCompliant I-Thou vectors regenerated under PyTorch 2.6 (replacing PT 2.1).\nPCA results essentially unchanged (compliant is 1/89 traits).\nSix-direction: prompt_end compliant shows weak refusal interaction\n(cos up to 0.30 at mid layers, -0.20 at deep layers), consistent\nwith steering experiments showing prompt_end can soften refusal.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"

r = subprocess.run(["git", "commit", "-m", msg], capture_output=True, text=True)
print(f"\nCommit:\n{r.stdout}\n{r.stderr}")

r = subprocess.run(["git", "push"], capture_output=True, text=True)
print(f"\nPush:\n{r.stdout}\n{r.stderr}")
