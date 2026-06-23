import subprocess, sys, os
os.chdir("/root/repositories/i-and-thou-vector-private")

# Pull latest fix
result = subprocess.run(["git", "pull", "origin", "main"], capture_output=True, text=True)
print("Pull:", result.stdout.strip())

# Run assistant axis
subprocess.run([
    sys.executable, "scripts/assistant_axis.py",
    "--model_name", "meta-llama/Llama-3.1-8B-Instruct",
    "--vector_dir", "data/vectors/Meta-Llama-3.1-8B-Instruct",
    "--response_dir", "data/responses",
    "--output_dir", "experiment_result/Meta-Llama-3.1-8B-Instruct"
])

print("\nAssistant axis done. Running six-direction...")
subprocess.run([
    sys.executable, "scripts/four_direction_comparison.py",
    "--model_short", "Meta-Llama-3.1-8B-Instruct"
])
print("ALL DONE")
