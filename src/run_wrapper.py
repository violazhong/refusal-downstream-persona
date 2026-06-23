import subprocess, sys
args = sys.argv[1:]
result = subprocess.run(["bash", "../i-and-thou-vector-private/scripts/run.sh"] + args, cwd="../i-and-thou-vector-private")
sys.exit(result.returncode)