#!/usr/bin/env python3
"""
Generate compliant trait vectors for Llama-3.1-8B-Instruct.
Runs Stage 1 (vLLM generation + GPT-4o scoring) then Stage 2 (vector extraction).
"""
import subprocess
import sys
import os

os.chdir("/root/repositories/i-and-thou-vector-private")

MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
TRAIT = "compliant"
RESPONSES_DIR = "data/responses"
VECTORS_DIR = "data/vectors"
SAMPLES = "40"

print("=" * 60, flush=True)
print(f"Generating compliant vectors for {MODEL}", flush=True)
print("=" * 60, flush=True)

# Stage 1: Generate responses
print("\n[Stage 1] Generating responses with vLLM + GPT-4o scoring...", flush=True)
r1 = subprocess.run([
    sys.executable,
    "scripts/generate_combined_responses.py",
    "--model", MODEL,
    "--trait", TRAIT,
    "--output_dir", RESPONSES_DIR,
    "--samples_per_instruction", SAMPLES,
], capture_output=False)

if r1.returncode != 0:
    print(f"Stage 1 FAILED with exit code {r1.returncode}", flush=True)
    sys.exit(1)

model_short = MODEL.split("/")[-1]
pos_csv = f"{RESPONSES_DIR}/{model_short}_{TRAIT}_positive.csv"
neg_csv = f"{RESPONSES_DIR}/{model_short}_{TRAIT}_negative.csv"
print(f"  pos: {pos_csv}", flush=True)
print(f"  neg: {neg_csv}", flush=True)

if not os.path.exists(pos_csv) or not os.path.exists(neg_csv):
    print("Stage 1 output files missing!", flush=True)
    sys.exit(1)

# Stage 2: Extract vectors
print("\n[Stage 2] Extracting vectors...", flush=True)
r2 = subprocess.run([
    sys.executable,
    "scripts/extract_vectors.py",
    "--model", MODEL,
    "--trait", TRAIT,
    "--responses_dir", RESPONSES_DIR,
    "--output_dir", VECTORS_DIR,
], capture_output=False)

if r2.returncode != 0:
    print(f"Stage 2 FAILED with exit code {r2.returncode}", flush=True)
    sys.exit(1)

# Verify output
vec_dir = f"{VECTORS_DIR}/{model_short}"
import glob
pt_files = glob.glob(f"{vec_dir}/{TRAIT}_*.pt")
print(f"\n  Compliant vector files: {len(pt_files)}", flush=True)
for f in sorted(pt_files):
    print(f"    {os.path.basename(f)}", flush=True)

print("\n" + "=" * 60, flush=True)
print("ALL DONE — compliant vectors generated", flush=True)
print("=" * 60, flush=True)
