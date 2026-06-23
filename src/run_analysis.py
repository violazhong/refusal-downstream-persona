import subprocess, sys
result = subprocess.run([sys.executable, "/root/repositories/i-and-thou-vector-private/scripts/e4_quick_analysis.py"], capture_output=True, text=True)
with open("/tmp/e4_analysis_output.txt", "w") as f:
    f.write(result.stdout)
    f.write(result.stderr)
print("saved to /tmp/e4_analysis_output.txt")
