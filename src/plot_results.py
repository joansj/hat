import sys,argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Arguments
parser=argparse.ArgumentParser(description='xxx')
parser.add_argument('--experiment',default='',type=str,required=True)
parser.add_argument('--approaches',default='',type=str,required=True)
parser.add_argument('--folders',default='../res/',type=str,required=False,help='(default=%(default)s)')
parser.add_argument('--seeds',default='0-9',type=str,required=False,help='(default=%(default)s)')
parser.add_argument('--output',default='',type=str,required=False,help='(default=%(default)s)')
args=parser.parse_args()
args.approaches=args.approaches.split(',')
args.folders=args.folders.split(',')
if '-' in args.seeds:
    tmp=args.seeds.split('-')
    args.seeds=list(range(int(tmp[0]),int(tmp[1])+1))
else:
    args.seeds=args.seeds.split(',')
for i in range(len(args.seeds)): args.seeds[i]=int(args.seeds[i])
print('='*100)
for arg in vars(args):
    print(arg+':',getattr(args,arg))
print('='*100)

#aref=['random','sgd-restart']
aref=['random','joint']
for a in aref:
    if a not in args.approaches:
        print('ERROR: Need',aref,'approaches for normalizing the accuracies across data sets')
        sys.exit()

########################################################################################################################

# Viz configs
use_conf_interv=True
#"""
markers=['o','v','^','<','>','s','*','d','x','+','h']
while len(markers)<len(args.approaches): markers+=markers
"""
markers=['o']*len(args.approaches)
"""
jitter=0.08

########################################################################################################################

# Load
print('Load results...')
data={}
ntasks=0
for f in args.folders:
    data[f]={}
    e=args.experiment
    # Load
    data[f]={}
    for a in args.approaches:
        data[f][a]=[]
        for s in args.seeds:
            fn=f+e+'_'+a+'_'+str(s)+'.txt'
            data[f][a].append(np.loadtxt(fn).astype(np.float32))
        data[f][a]=100*np.stack(data[f][a],axis=0)
    if ntasks==0: ntasks=data[f]['random'].shape[1]
    # Normalize
    for s in range(len(args.seeds)):
        for a in args.approaches:
            if a in aref: continue
            ref_random=np.repeat(np.reshape(np.diag(data[f]['random'][s]),(1,data[f]['random'][s].shape[1])),data[f]['random'][s].shape[0],axis=0)
            #ref_ref=np.repeat(np.reshape(np.diag(data[f]['sgd-restart'][s]),(1,data[f]['sgd-restart'][s].shape[1])),data[f]['sgd-restart'][s].shape[0],axis=0)
            ref_ref=data[f]['joint'][s]
            data[f][a][s]=(data[f][a][s]-ref_random)/(ref_ref-ref_random)-1

########################################################################################################################

# Plot
print('Plot')
print('-'*100)
plt.figure()
leg=[]
for f in data.keys():
    for a in data[f].keys():
        if a in aref: continue
        # Get data
        acc=np.zeros((data[f][a].shape[0],data[f][a].shape[1]),dtype=np.float32)
        for j in range(acc.shape[1]):
            acc[:,j]=np.mean(data[f][a][:,j,:j+1],axis=1)
        # Prepare things
        jit=jitter*2*(len(leg)/(len(args.approaches)-2)-0.5)
        ci=np.std(acc,axis=0)
        if use_conf_interv:
            ci=stats.t._ppf((1+0.95)/2,acc.shape[0]-1)*ci/np.sqrt(acc.shape[0])
        mark=markers[len(leg)]
        # Do the plot
        plt.errorbar(1+np.arange(acc.shape[1])+jit,np.mean(acc,axis=0),yerr=ci,fmt='-'+mark,markersize=6,capsize=3)
        #leg.append(f+' : '+a)
        leg.append(a)
        # Print
        print('{:16s}  '.format(a),end='')
        for j in range(acc.shape[1]):
            print('{:6.3f} ({:5.3f})  '.format(np.mean(acc[:,j]),ci[j]),end='')
        print('-->  {:6.3f}'.format(np.mean(acc)))
print('-'*100)
#plt.xlim(1-0.2,ntasks+0.2)
plt.xticks(1+np.arange(ntasks).astype(int))
plt.xlabel('Task number')
#plt.ylim(-0.1,1.1)
plt.ylabel('Relative accuracy drop')
plt.legend(leg)
plt.plot([1-jitter,ntasks+jitter],[0,0],'k--',linewidth=2)
if args.output=='':
    plt.show()
else:
    plt.savefig(args.output,bbox_inches='tight')
plt.close()

########################################################################################################################

print('Done!')
