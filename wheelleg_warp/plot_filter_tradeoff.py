"""All registered cases show coupled yaw/roll changes, not a chosen success."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nom_yaw_filter_probe import OUT,rec


def run():
    file=OUT/'round178_pair_review.json';data=json.loads(file.read_text());assert data['verified']
    fig,axes=plt.subplots(2,2,figsize=(9,7),sharex=True,sharey=True,constrained_layout=True)
    for ax,(name,pair) in zip(axes.flat,data['pairs'].items()):
        assert len(pair['rows'])==20
        for r in pair['rows']:
            x,y=r['original_peak_deg'][2],r['original_peak_deg'][0]
            u,v=r['latest_peak_deg'][2],r['latest_peak_deg'][0]
            low=r['height_m']==.115
            ax.annotate('',xy=(u,v),xytext=(x,y),arrowprops=dict(arrowstyle='->',lw=1.2 if low else .5,color='#b53835' if low else '#aaaaaa'))
            ax.plot(x,y,'o',mfc='none',mec='#666666',ms=4)
            ax.plot(u,v,'o',color='#b53835' if low else '#3159a7',ms=4)
        ax.axvline(5,color='#777777',ls='--',lw=.8);ax.axhline(5,color='#777777',ls='--',lw=.8)
        ax.set_title(name);ax.grid(alpha=.15);ax.set_xlim(0,8);ax.set_ylim(0,7)
        ax.text(.02,.03,'Red: all 0.115 m cases',transform=ax.transAxes,fontsize=9,color='#b53835')
    for ax in axes[-1]:ax.set_xlabel('Peak absolute yaw (deg)')
    for ax in axes[:,0]:ax.set_ylabel('Peak absolute roll (deg)')
    fig.suptitle('Original lowpass (open) to latest gyro (filled): all 20 cases per panel\nLower yaw can introduce roll violations; dashed lines are original 5 deg limits')
    for suffix in ('png','svg'):fig.savefig(OUT/f'filter_attitude_tradeoff.{suffix}',dpi=180)
    plt.close(fig)
    rec.atomic_json(OUT/'filter_tradeoff_figure_manifest.json',dict(verified=True,comparison_records=80,
        selection='All20 development cases per paired controller/noise panel; all0.115m cases highlighted, no selected winner.',
        review_sha256=rec.sha(file),source_sha256=rec.sha(__file__),
        output_sha256={n:rec.sha(OUT/n) for n in ('filter_attitude_tradeoff.png','filter_attitude_tradeoff.svg')}))
    print('PASS all80 paired changes plotted; original axis limits retained')


if __name__=='__main__':run()
