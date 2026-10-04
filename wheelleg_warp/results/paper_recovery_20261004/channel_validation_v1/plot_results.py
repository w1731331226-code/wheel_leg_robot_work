"""Fixed-model independent panels, not a learned-policy superiority plot."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=Path(__file__).resolve().parent;r=json.loads((p/'independent_review.json').read_text());assert r['complete']
models=['original_1609','potential_1609','original_1610','potential_1610','original_1611','potential_1611']
fig,axes=plt.subplots(1,3,figsize=(12,3.8));x=np.arange(6)
for ax,category,title,n in zip(axes,['regular','pressure','legacy_regression'],['Fresh regular (96)','Pressure (48)','Reused regression (28)'],[96,48,28]):
    for delta,mask,label in [(-.17,'all','All residuals'),(.17,'legs_only','Wheel disabled')]:
        counts=[r['decoded'][f'{m}_{mask}/{category}']['summary']['success_count'] for m in models]
        ax.bar(x+delta,np.array(counts)/n*100,width=.34,label=label)
        for xx,c in zip(x+delta,counts):ax.text(xx,c/n*100+1,str(c),ha='center',fontsize=7)
    baseline=r['decoded'][f'B1/{category}']['summary']['success_count']/n*100
    ax.axhline(baseline,ls='--',color='grey',label='Fixed B1')
    ax.set_xticks(x,labels=['R09','P09','R10','P10','R11','P11']);ax.set(title=title,ylim=(0,112),xlabel='Reward/seed of fixed model')
    ax.grid(axis='y',alpha=.2)
axes[0].set_ylabel('Complete-task success (%)')
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,fontsize=8,loc='upper center',ncol=3)
fig.tight_layout(rect=(0,0,1,.91));fig.savefig(p/'independent_channels.png',dpi=180);plt.close(fig)
