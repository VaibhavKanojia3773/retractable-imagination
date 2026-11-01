"""Generate vector figures and TeX numbers/tables from frozen measured results."""
from pathlib import Path
import sys,json,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
P=Path(__file__).resolve().parents[1];R=P.parent
sys.path.insert(0,str(R))
from engine import plan,replay,Evidence
for file in ('evaluation.json','study.json'):
    shutil.copy2(R/'results'/file,P/'data'/file)
D=json.loads((P/'data'/'evaluation.json').read_text());S=json.loads((P/'data'/'study.json').read_text())
raw=(R/'web'/'demo-data.js').read_text(encoding='utf8');scene=json.loads(raw.split('=',1)[1].strip().rstrip(';'))['scene']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'axes.labelcolor':'#253248','text.color':'#253248'})
teal='#16796e';purple='#7160bb';amber='#c88625';gray='#718096'
events=[Evidence(e['id'],e['cell'],e['delta']) for e in scene['events']]
before=replay(scene['prior'],events);after=replay(scene['prior'],events,scene['bad_ids']);current=4*scene['width']+8
fig,axes=plt.subplots(1,2,figsize=(7.0,2.25),constrained_layout=True)
for ax,belief,label,color in zip(axes,[before,after],['(a) Invalid evidence retained','(b) Evidence retracted'],[purple,teal]):
    truth=np.asarray(scene['truth']).reshape(scene['height'],scene['width'])
    ax.imshow(truth,cmap=ListedColormap(['#f3f6f9','#bdc8d6']),vmin=0,vmax=1,interpolation='nearest')
    path=plan(belief,scene['width'],scene['height'],current,scene['goal'])
    ax.plot([c%scene['width'] for c in path],[c//scene['width'] for c in path],color=color,lw=1.8)
    ax.plot(range(3,9),[4]*6,'o-',color='#253248',markersize=2.5,lw=1.2)
    ax.scatter([14],[4],s=70,facecolors='none',edgecolors=amber,linewidths=1.5,zorder=8)
    ax.scatter([24],[4],marker='*',s=60,c=teal,zorder=9)
    ax.set_title(label,loc='left',fontsize=9,fontweight='bold',pad=6);ax.set_xticks([]);ax.set_yticks([])
    for spine in ax.spines.values():spine.set_visible(False)
fig.savefig(P/'figures'/'retraction_scene.pdf',bbox_inches='tight');plt.close(fig)
timing=S['api_timing']['rows'];names=['full_replay','checkpoint_replay','explicit_subtract','indexed_retract'];labels=['Full replay','Checkpoint replay','Explicit subtraction','Indexed retraction (ours)']
fig,ax=plt.subplots(figsize=(6.8,2.15),constrained_layout=True);x=np.arange(3);width=.18
for j,(key,label,color) in enumerate(zip(names,labels,[gray,'#b5becb',purple,teal])):
    med=np.array([r['methods'][key]['median_ms'] for r in timing]);q1=np.array([r['methods'][key]['q25_ms'] for r in timing]);q3=np.array([r['methods'][key]['q75_ms'] for r in timing])
    ax.bar(x+(j-1.5)*width,med,width,label=label,color=color,yerr=np.array([med-q1,q3-med]),capsize=2,error_kw={'elinewidth':.7})
ax.set_yscale('log');ax.set_xticks(x,[f"{r['length']:,}" for r in timing]);ax.set_xlabel('Encoded observations in history');ax.set_ylabel('Backend update + planning (ms)');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.legend(ncol=2,frameon=False,fontsize=7,loc='upper left')
fig.savefig(P/'figures'/'backend_latency.pdf',bbox_inches='tight');plt.close(fig)
f=S['tasks']['families'];n=S['tasks']['per_family']
macros={'TaskN':S['tasks']['total'],'FamilyN':n,'SensorTrainN':D['training']['train_examples'],'SensorTestN':D['training']['test_examples'],'SensorAccuracy':f"{100*D['training']['test_accuracy']:.2f}",'SensorBrier':f"{D['training']['test_brier']:.4f}",'BeforeSteps':int(f['false_occupied']['methods']['stale']['median_successful_steps']),'AfterSteps':int(f['false_occupied']['methods']['indexed_retract']['median_successful_steps']),'LongestReplayMs':f"{timing[-1]['methods']['full_replay']['median_ms']:.2f}",'LongestIndexedMs':f"{timing[-1]['methods']['indexed_retract']['median_ms']:.2f}",'LongestExplicitMs':f"{timing[-1]['methods']['explicit_subtract']['median_ms']:.2f}"}
(P/'numbers.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+str(v)+'}' for k,v in macros.items())+'\n',encoding='utf8')
rows=[r'\begin{tabular}{lrrrr}',r'\toprule',r'Task family & Stale & Stale cache & Replay & \textbf{Ours} \\',r'\midrule']
for kind,title in [('false_free','False free space'),('false_occupied','False obstacle'),('source_bundle','Source packet group'),('no_corruption','No corruption')]:
    vals=[f[kind]['methods'][m]['successes'] for m in ['stale','state_only_stale_forecast','full_replay','indexed_retract']];rows.append(title+' & '+' & '.join(f'{v}/{n}' for v in vals)+r' \\')
rows.extend([r'\bottomrule',r'\end{tabular}']);(P/'tables'/'task_results.tex').write_text('\n'.join(rows),encoding='utf8')
rows=[r'\begin{tabular}{lrrrr}',r'\toprule',r'History & Replay & Checkpoint & Explicit & \textbf{Ours} \\',r'\midrule']
for r in timing:rows.append(f"{r['length']:,}"+' & '+' & '.join(f"{r['methods'][m]['median_ms']:.3f}" for m in names)+r' \\')
rows.extend([r'\bottomrule',r'\end{tabular}']);(P/'tables'/'timing_results.tex').write_text('\n'.join(rows),encoding='utf8')
print('Generated vector figures, data snapshots, numbers.tex and tables from measured JSON.')
