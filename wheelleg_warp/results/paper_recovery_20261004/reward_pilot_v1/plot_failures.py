"""Failure counts overlap; this figure does not assign exclusive causes."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=Path(__file__).resolve().parent;r=json.loads((p/'failure_structure.json').read_text())
keys=[f'{arm}/{seed}' for seed in [1609,1610,1611] for arm in ['original','potential']]
cols=['design','legacy_contact','terrain_contact','attitude','yaw']
matrix=np.array([[r['results'][key]['overlapping_failure_counts'][c] for c in cols] for key in keys])
fig,ax=plt.subplots(figsize=(8,4.2));im=ax.imshow(matrix,cmap='Blues',vmin=0,vmax=96,aspect='auto')
ax.set_xticks(range(5),labels=['Design','Legacy contact','Terrain contact','Attitude','Yaw'])
ax.set_yticks(range(6),labels=[key.replace('original/','Raw ').replace('potential/','Phi ') for key in keys])
for i in range(6):
    for j in range(5):ax.text(j,i,str(matrix[i,j]),ha='center',va='center',color='white' if matrix[i,j]>48 else 'black')
ax.set_title('Overlapping failures on same 96 public cases')
fig.colorbar(im,ax=ax,label='Method-case failure count');fig.tight_layout()
fig.savefig(p/'failure_structure.png',dpi=180);plt.close(fig)
