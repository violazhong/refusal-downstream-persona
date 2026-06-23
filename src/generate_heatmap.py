import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

with open('experiment_result/Qwen2.5-7B-Instruct/alignment_matrix.json') as f:
    data = json.load(f)

pc1_layers = data['pc1_layers']
refusal_layers = data['refusal_layers']
positions = data['positions']
validation = data['validation']
matrix = data['alignment_matrix']

validated = {int(k): v['validated'] for k, v in validation.items()}

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
fig.suptitle('PC1 × Refusal Alignment Matrix (cos similarity)', fontsize=14, fontweight='bold')

for idx, pos in enumerate(positions):
    ax = axes[idx]
    mat = np.zeros((len(pc1_layers), len(refusal_layers)))
    for i, pl in enumerate(pc1_layers):
        for j, rl in enumerate(refusal_layers):
            key = f'L{pl}_{pos}_L{rl}'
            mat[i, j] = matrix[key]['cos']
    
    vmax = max(0.5, np.abs(mat).max())
    im = ax.imshow(mat, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    
    ax.set_xticks(range(len(refusal_layers)))
    ax.set_xticklabels([f'L{l}' for l in refusal_layers])
    ax.set_yticks(range(len(pc1_layers)))
    ax.set_yticklabels([f'L{l}' for l in pc1_layers])
    ax.set_xlabel('Refusal extraction layer')
    if idx == 0:
        ax.set_ylabel('PC1 extraction layer')
    ax.set_title(pos.replace('_', ' ').title())
    
    for i, pl in enumerate(pc1_layers):
        for j, rl in enumerate(refusal_layers):
            val = mat[i, j]
            color = 'white' if abs(val) > 0.25 else 'black'
            marker = '*' if not validated.get(rl, False) else ''
            ax.text(j, i, f'{val:+.3f}{marker}', ha='center', va='center',
                    fontsize=7, color=color, fontweight='bold' if abs(val) > 0.3 else 'normal')
            if not validated.get(rl, False):
                ax.add_patch(mpatches.Rectangle((j-0.5, i-0.5), 1, 1,
                    fill=False, hatch='//', edgecolor='gray', linewidth=0.5, alpha=0.3))
    
    # Mark reference cell
    if pos == 'response_avg':
        ref_i = pc1_layers.index(10)
        ref_j = refusal_layers.index(10)
        rect = mpatches.Rectangle((ref_j-0.5, ref_i-0.5), 1, 1,
            fill=False, edgecolor='gold', linewidth=3)
        ax.add_patch(rect)

plt.colorbar(im, ax=axes, shrink=0.8, label='Cosine Similarity')
plt.tight_layout(rect=[0, 0, 0.92, 0.95])

out_path = 'experiment_result/Qwen2.5-7B-Instruct/alignment_matrix_heatmap.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'Saved heatmap to {out_path}')
import os
print(f'Size: {os.path.getsize(out_path):,} bytes')
