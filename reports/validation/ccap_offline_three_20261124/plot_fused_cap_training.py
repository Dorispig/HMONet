import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

report=json.loads(Path('C:/Users/24846/tmp/fused_cap_training_comparison.json').read_text())
fig,axes=plt.subplots(2,2,figsize=(10,6.8),sharex=True)
for row,family in enumerate(report['families']):
    samples=family['validation']
    steps=[r['iteration'] for r in samples]
    for column,(key,title) in enumerate((('normalized_pde_loss','Normalized PDE loss'),('mesh_mse','MSE to PDE mesh'))):
        ax=axes[row,column]
        for name,color,style in (('original','#2563a6','-'),('fused','#d26927','--')):
            ax.plot(steps,[r[name][key] for r in samples],style,color=color,linewidth=1.7,label=name)
        ax.set_yscale('log')
        ax.grid(alpha=.25)
        ax.set_title(f"{family['family']} | {title}")
        if row==1:ax.set_xlabel('Training iteration')
        if column==0:ax.set_ylabel('Fixed validation set')
        if row==0 and column==0:ax.legend(frameon=False)
fig.tight_layout()
path=Path('C:/Users/24846/tmp/fused_cap_training_comparison.png')
fig.savefig(path,dpi=180)
print(path)
