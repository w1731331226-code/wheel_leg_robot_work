"""Reproduce the fixed-final-checkpoint paired pilot figure."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=Path(__file__).resolve().parent;r=json.loads((p/'pair_review.json').read_text())
assert r['completed_seed_pairs']==3 and r['completed_runs']==6
seeds=[1609,1610,1611];x=np.arange(3);fig,axes=plt.subplots(1,2,figsize=(9,3.6))
for offset,arm,label in [(-.17,'original','Original reward'),(.17,'potential','Potential reward')]:
    count=[r['completed'][f'{arm}/{s}']['summary']['success_count'] for s in seeds]
    failures=[96-r['completed'][f'{arm}/{s}']['design'] for s in seeds]
    axes[0].bar(x+offset,np.array(count)/96*100,width=.34,label=label)
    axes[1].bar(x+offset,failures,width=.34,label=label)
    for xx,c in zip(x+offset,count):axes[0].text(xx,c/96*100+1,str(c),ha='center',fontsize=8)
    for xx,c in zip(x+offset,failures):axes[1].text(xx,c+.1,str(c),ha='center',fontsize=8)
axes[0].axhline(94/96*100,ls='--',color='grey',label='Fixed B1: 94/96')
axes[0].set(ylabel='Complete-task success (%)',ylim=(0,110),title='Fixed final 200k public evaluation')
axes[1].set(ylabel='Design failures out of 96',ylim=(0,8),title='Original design gate retained')
for ax in axes:
    ax.set_xticks(x,labels=[str(s) for s in seeds]);ax.set_xlabel('Paired training seed')
    ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
fig.tight_layout();fig.savefig(p/'three_seed_pilot.png',dpi=180);plt.close(fig)
