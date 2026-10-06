"""One declared no-reference-clipping example; not an aggregate benefit figure."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from analyze_complete_contact import moments, I
from complete_contact_recorder import OUT, sha, atomic_json


def run():
    analysis=OUT/'complete_contact_analysis.json'
    a=json.loads(analysis.read_text());assert a['verified']
    case=6301009  # Lowest registered 0.16m case, independently used for clamp counterexample.
    fig,axes=plt.subplots(2,2,figsize=(10,6),sharex=True,constrained_layout=True)
    inputs={}
    for col,label in enumerate(('B0','B1-route')):
        row=next(r for r in a['results'] if r['controller']==label and r['case']==case)
        assert not row['reference_clip_observed']
        file=OUT/'runs'/label/f'case_{case}.npz'
        assert sha(file)==a['input_sha256'][str(file.relative_to(OUT))]
        inputs[str(file.relative_to(OUT))]=sha(file)
        with np.load(file,allow_pickle=False) as z:
            t,meta,c,steps=z['trace'],z['meta'],z['contacts'],z['contact_steps'].astype(int)
        g=json.loads((file.parent/'geometry.json').read_text());wheel=np.asarray(g['wheel_geom_ids']);body=np.asarray(g['geom_body_ids'])
        p,q=c[:,2].astype(int),c[:,3].astype(int);wa,wb=np.isin(p,wheel),np.isin(q,wheel)
        external=(wa & (body[q]==0)) | (wb & (body[p]==0))
        vals=moments(c,meta[steps-1,3:6],np.where(wb,1.,-1.))[:3]
        zero=row['first_positive_target_contact_s'];x=t[:,I['pre_s']]-zero
        keep=(x>=-.02)&(x<=.2)
        for values,name,color in zip(vals,('Full wheel-ground moment','Normal force only','Tangential force + contact torque'),('black','#cf6a00','#008977')):
            total=np.bincount(steps[external]-1,weights=values[external],minlength=len(t))
            axes[0,col].plot(x[keep],total[keep],label=name,color=color,lw=1)
        axes[0,col].set_title(label)
        axes[1,col].plot((t[:,I['post_s']]-zero)[keep],np.rad2deg(t[keep,I['yaw']]),color='#3159a7')
        for limit in (-5,5):axes[1,col].axhline(limit,color='#b33c37',ls='--',lw=.8)
        axes[1,col].axvline(row['first_yaw5_s']-zero,color='#b33c37',ls=':',lw=.8)
        for ax in axes[:,col]:ax.axvline(0,color='gray',ls=':',lw=.8);ax.grid(alpha=.2)
        axes[1,col].set_xlabel('Time from first positive target contact (s)')
    axes[0,0].set_ylabel('External yaw moment about COM (Nm)')
    axes[1,0].set_ylabel('Body yaw (deg)')
    axes[0,0].legend(fontsize=8,loc='lower right')
    fig.suptitle('0.16 m height, left 20 mm obstacle, +0.7 m/s; development case 6301009\nNormal impact and opposing traction; no roll-reference clipping')
    for suffix in ('png','svg'):fig.savefig(OUT/f'contact_impact_example.{suffix}',dpi=180)
    plt.close(fig)
    atomic_json(OUT/'contact_impact_figure_manifest.json',dict(verified=True,case=case,
        selection='Lowest registered0.16m case; all four such cases fail both classical laws, no reference clamp. Illustrative, not selected to establish effect/independent advantage.',
        analysis_sha256=sha(analysis),input_sha256=inputs,source_sha256=sha(__file__),
        output_sha256={n:sha(OUT/n) for n in ('contact_impact_example.png','contact_impact_example.svg')}))
    print('PASS figure exports and source/data bindings; observational illustration only')


if __name__=='__main__':run()
