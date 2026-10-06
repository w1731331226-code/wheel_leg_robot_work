"""Illustrative observed centre paths, not a new task score or mode certificate."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from task_mode_recorder import OUT
from review_yaw_sector import sha
from dashboard.live_env import atomic_json


def run():
    review=json.loads((OUT/'physical_mode_analysis.json').read_text()); assert review['verified']
    case=6301001; fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True); inputs={}
    labels=['B0','B1-route','V6-1761','V6-1762','V6-1763']
    for k,label in enumerate(labels):
        directory=OUT/'runs'/label; result=json.loads((directory/'result.json').read_text()); row=next(x for x in result['runs'] if x['seed']==case)
        g=json.loads((directory/'geometry.json').read_text()); box=g['boxes'][g['case_ids'].index(case)][0]
        file=directory/row['task_mode_trace']['path']; assert sha(file)==row['task_mode_trace']['sha256']; inputs[str(file.relative_to(OUT))]=sha(file)
        with np.load(file,allow_pickle=False) as data:
            t=data['trace']; i={str(c):j for j,c in enumerate(data['columns'])}
            x=t[:,i['post_left_x']]; y=t[:,i['post_left_y']]; selected=(x>=1.2)&(x<=2.4)
            colour=f'C{k}'; axes[0].plot(x[selected],y[selected],color=colour,label=label,linewidth=1.5)
            px=t[:,i['pre_left_x']]; choose=(px>=1.2)&(px<=2.4)
            axes[1].plot(px[choose],t[choose,i['left_target_vertical_normal_N']],color=colour,linewidth=1.1,label=label)
    p=np.asarray(box['position']); half=np.asarray(box['size'])
    axes[0].add_patch(Rectangle((p[0]-half[0],p[1]-half[1]),2*half[0],2*half[1],facecolor='grey',edgecolor='black',alpha=.18))
    axes[0].axhline(p[1],color='grey',linestyle=':',linewidth=.8)
    axes[0].set(xlabel='Wheel-centre x (m), post integration',ylabel='Wheel-centre y (m)',title='Centre path versus obstacle footprint',xlim=(1.2,2.4),ylim=(-.30,-.05))
    axes[1].set(xlabel='Wheel-centre x (m), pre integration',ylabel='Target vertical normal component (N)',title='Positive edge contact can coexist with lateral paths',xlim=(1.2,2.4))
    for ax in axes: ax.grid(alpha=.2)
    axes[0].legend(fontsize=8,loc='lower left'); fig.suptitle('Illustrative case: 0.115 m command, 20 mm left obstacle',fontsize=11)
    png=OUT/'low_height_edge_witness.png'; svg=OUT/'low_height_edge_witness.svg'; fig.savefig(png,dpi=200); fig.savefig(svg); plt.close(fig)
    atomic_json(OUT/'figure_manifest.json',dict(case=case,input_sha256=inputs,analysis_sha256=sha(OUT/'physical_mode_analysis.json'),source_sha256=sha(__file__),png_sha256=sha(png),svg_sha256=sha(svg),interpretation='Single illustrative case selected because all3 V6 seeds satisfy original success; not independent or general superiority. Centre outside footprint is not a full tyre bypass verdict. Force is solver-time vertical normal component, not total support.'))
    print('Saved PNG/SVG observed centre/contact figure',flush=True)


if __name__=='__main__': run()
