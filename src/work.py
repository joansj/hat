import os,sys
import itertools
import numpy as np

if len(sys.argv)<5 or len(sys.argv)>6:
    print(sys.argv[0],'<gpu_id> <experiment> <approach> <output_path> [--execute]')
    sys.exit()

gpu_id=int(sys.argv[1])
exper=sys.argv[2]
appro=sys.argv[3]
outpath=sys.argv[4]

agpus=[0,1,2]
if gpu_id not in agpus:
    print('ERROR:',gpu_id,'not in',agpus)
    sys.exit()
seed=0
extra_params=''
#extra_params+=' --nepochs=100'
resname=appro
#resname='test'
execute=False
if '--execute' in sys.argv: execute=True

########################################################################################################################

pars=[[0.1, 0.25, 0.5, 0.75, 1, 1.5, 2.5, 4],
      [25,50,100,200,400,800],
      ]

########################################################################################################################

print('agpus:',agpus)
print('seed:',seed)
print('extra_params:',extra_params)
print('resname:',resname)
print('execute:',execute)
print('pars =',pars)
input('Shall we continue? ')
print()

# Generate parameter combinations
parstr=[]
for p in itertools.product(*pars):
    pp=''
    for aux in p:
        pp+=str(aux)+','
    parstr.append(pp[:-1])

# Run
plotstr=''
for i,p in enumerate(parstr):
    call='CUDA_VISIBLE_DEVICES='+str(gpu_id)
    call+=' python run.py --experiment='+exper
    call+=' --approach='+appro
    call+=' --seed='+str(seed)
    call+=' --parameter='+p
    call+=extra_params
    aux=resname+'-'+p.replace(',','-')
    call+=' --output='+outpath+'/'+exper+'_'+aux+'_'+str(seed)+'.txt'
    plotstr+=aux+','
    if i%len(agpus)==agpus.index(gpu_id):
        print(i,'\t',call)
        if execute:
            os.system(call)
plotstr=plotstr[:-1]
print()
print(plotstr)
print()

########################################################################################################################
