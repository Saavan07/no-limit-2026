"""Two score-compatible populations with different consequences for an entrant."""
import numpy as np
from game2_lab.cross_arena import read
from run_population_study import ROOT,save,write_csv


def main():
    study=ROOT/'results/game2-catalog-stress';out=study/'review';out.mkdir(exist_ok=True)
    models=[r for r in read(study/'model-validation.json')['models']if r['validation']['passes']]
    with np.load(study/'split-scores.npz')as d:v24=d['values'];names=d['names'].tolist()
    with np.load(study/'refinement/scores.npz')as d:v34=d['values']
    case_names=['supplied_fit_13','equal_active_targets__unrestricted__2'];rows=[]
    display={'supplied_fit_13':'Supplied field 13','equal_active_targets__unrestricted__2':'Equal-centered targets'}
    for name in case_names:
        j=names.index(name);m=models[j];rows.append({'name':name,'full_cdf_gap':m['validation']['maximum_cdf_gap'],
            'C2_36_C4_64':float(v24[36,j]),'C2_37_C4_63':float(v24[37,j]),'C2_38_C4_62':float(v24[38,j]),
            'C3_36_C4_64':float(v34[36,j]),'C3_37_C4_63':float(v34[37,j]),'C3_39_C4_61':float(v34[39,j])})
    save(out/'two-compatible-populations.json',rows)
    write_csv(out/'two-compatible-populations.csv',list(rows[0]),[[r[k]for k in rows[0]]for r in rows])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    public=np.sort(read(ROOT/'data/parry-replay/public-scores.json')['scores']);xx=np.linspace(0,9,2001)
    fig,(ax,bx)=plt.subplots(1,2,figsize=(12,5));ax.plot(xx,np.searchsorted(public,xx,side='right')/len(public),lw=2.5,c='black',label='Published scores')
    colors=['#167b80','#bd6726']
    for color,name in zip(colors,case_names):
        m=models[names.index(name)]
        with np.load(study/'validation'/(name+'.npz'))as d:u=d['values'];families=d['families']
        w=np.array(m['weights'])[families]/256;order=np.argsort(u);cw=np.cumsum(w[order]);u=u[order]
        ax.plot(xx,np.r_[0.,cw][np.searchsorted(u,xx,side='right')],color=color,lw=1.5,label=display[name])
    ax.set(xlabel='Incumbent mean placement payout',ylabel='Cumulative population mass',xlim=(0,9),ylim=(0,1),title='Both populations pass the score check');ax.grid(alpha=.2);ax.legend(fontsize=7,loc='lower right')
    labels=['C2/C4: 36/64','C2/C4: 37/63','C3/C4: 36/64','C3/C4: 37/63'];keys=['C2_36_C4_64','C2_37_C4_63','C3_36_C4_64','C3_37_C4_63']
    for j,(color,row)in enumerate(zip(colors,rows)):
        bx.barh(np.arange(4)+(-.18 if j==0 else .18),[row[k]for k in keys],height=.34,color=color,label=display[row['name']])
    bx.set_yticks(np.arange(4),labels);bx.invert_yaxis();bx.set(xlabel='Entrant expected placement payout',xlim=(0,9),title='The entrant preference can still differ');bx.grid(axis='x',alpha=.2);bx.legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'two-score-compatible-populations.png',dpi=170);plt.close(fig)
    print(rows)


if __name__=='__main__':main()
