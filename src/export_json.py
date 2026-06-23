import json
with open("experiment_result/Qwen2.5-7B-Instruct/alignment_matrix.json") as f:
    data = json.load(f)
print(json.dumps(data))
