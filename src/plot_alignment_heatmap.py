import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

data_path = 'experiment_result/alignment_matrix.json'
with open(data_path) as f:
    data = json.load(f)

pc1_layers = data['pc1_layers']
refusal_layers = data['refusal_layers']
positions = data['positions']
validation = data['validation']
matrix = data['alignment_matrix']

validated = {int(k): v['validated'] for k, v in validation.items()}

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
fig.suptitle('E7: PC1 × Refusal Alignment Matrix (cosine similarity)', fontsize=14, fontweight='bold')

for idx, pos in enumerate(positions):
    ax = axes[idx]
    mat = np.zeros((len(pc1_layers), len(refusal_layers)))
    for i, pl in enumerate(pc1_layers):
        for j, rl in enumerate(refusal_layers):
            mat[i, j] = matrix[pos][str(pl)][str(rl)]['cos']

    vmax = 0.5
    im = ax.imshow(mat, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')

    ax.set_xticks(range(len(refusal_layers)))
    ax.set_xticklabels([f'L{l}' for l in refusal_layers])
    ax.set_yticks(range(len(pc1_layers)))
    ax.set_yticklabels([f'L{l}' for l in pc1_layers])
    ax.set_xlabel('Refusal extraction layer')
    if idx == 0:
        ax.set_ylabel('PC1 extraction layer')

    title = pos.replace('_', ' ').title()
    ax.set_title(title, fontsize=12)

    for i, pl in enumerate(pc1_layers):
        for j, rl in enumerate(refusal_layers):
            val = mat[i, j]
            color = 'white' if abs(val) > 0.25 else 'black'
            cell_validated = matrix[pos][str(pl)][str(rl)].get('refusal_validated', validated.get(rl, False))
            suffix = '*' if not cell_validated else ''
            weight = 'bold' if abs(val) > 0.3 else 'normal'
            ax.text(j, i, f'{val:+.3f}{suffix}', ha='center', va='center',
                    fontsize=7, color=color, fontweight=weight)
            if not cell_validated:
                ax.add_patch(mpatches.Rectangle((j-0.5, i-0.5), 1, 1,
                    fill=False, hatch='//', edgecolor='gray', linewidth=0.5, alpha=0.3))

    if pos == 'response_avg':
        ref_i = pc1_layers.index(10)
        ref_j = refusal_layers.index(10)
        rect = mpatches.Rectangle((ref_j-0.5, ref_i-0.5), 1, 1,
            fill=False, edgecolor='gold', linewidth=3)
        ax.add_patch(rect)

plt.colorbar(im, ax=axes, shrink=0.8, label='Cosine Similarity')

legend_elements = [
    mpatches.Patch(facecolor='white', edgecolor='gray', hatch='//', label='Unvalidated (ASR < 30pp)'),
    mpatches.Patch(facecolor='none', edgecolor='gold', linewidth=2, label='Reference cell (L10, response_avg)'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=9,
           bbox_to_anchor=(0.45, -0.02))

plt.tight_layout(rect=[0, 0.05, 0.92, 0.95])

out_path = 'experiment_result/alignment_matrix_heatmap.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'Saved heatmap to {out_path}')

import os
print(f'Size: {os.path.getsize(out_path):,} bytes')
