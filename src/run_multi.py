import sys,os,argparse,time
import numpy as np
from multiprocessing import Process

tstart=time.time()

# Arguments
parser=argparse.ArgumentParser(description='xxx')
parser.add_argument('--experiments',default='',type=str,required=True)
parser.add_argument('--approaches',default='',type=str,required=True)
parser.add_argument('--tmp_folder',default='../tmp/',type=str,required=False,help='(default=%(default)s)')
parser.add_argument('--res_folder',default='../res/',type=str,required=False,help='(default=%(default)s)')
parser.add_argument('--nruns',default=10,type=int,required=False,help='(default=%(default)d)')
parser.add_argument('--nproc',default=3,type=int,required=False,help='(default=%(default)d)')
parser.add_argument('--ngpus',default=3,type=int,required=False,help='(default=%(default)d)')
parser.add_argument('--sleep',default=0.9,type=float,required=False,help='(default=%(default)f)')
parser.add_argument('--verbose',action='store_true',help='(default=%(default)s)')
parser.add_argument('--execute',action='store_true',help='(default=%(default)s)')
args=parser.parse_args()
args.experiments=args.experiments.split(',')
args.approaches=args.approaches.split(',')

########################################################################################################################

for arg in vars(args):
    print(arg+':',getattr(args,arg))
print()

input('Shall we continue? ')
print()

########################################################################################################################

def form_tmp_filename(folder,i,j):
    return folder+'using_gpu'+'-'+str(i)+'-'+str(j)+'.log'

def launch_process(fn,call):
    if not args.verbose:
        call+=' > '+fn[:-4]+'.stdout.log'
    print(call)
    sys.stdout.flush()
    if args.execute:
        os.system(call)
    else:
        time.sleep(10*args.sleep*(1+np.random.rand()))
    print('Done ['+call+']')
    os.system('rm '+fn)
    return

def mymap(calls,gpuproc):
    # Init
    p=[]
    for i,j,k in gpuproc:
        fn=form_tmp_filename(args.tmp_folder,i,j)
        if os.path.exists(fn):
            os.system('rm '+fn)
        p.append(None)
    # Work
    for call in calls:
        i_gpu=-1
        i_proc=-1
        i_k=-1
        while i_gpu<0:
            time.sleep(args.sleep*(1+np.random.rand()))
            for i,j,k in gpuproc:
                fn=form_tmp_filename(args.tmp_folder,i,j)
                if not os.path.exists(fn):
                    i_gpu=i
                    i_proc=j
                    i_k=k
                    break
        fn=form_tmp_filename(args.tmp_folder,i_gpu,i_proc)
        os.system('touch '+fn)
        p[i_k]=Process(target=launch_process,args=(fn,'CUDA_VISIBLE_DEVICES='+str(i_gpu)+' '+call))
        p[i_k].start()
    # Wait for last
    for i,j,k in gpuproc:
        try:
            p[k].join()
        except:
            time.sleep(args.sleep*(1+np.random.rand()))
    return None

########################################################################################################################

# Create list of calls
calls=[]
for e in args.experiments:
    for a in args.approaches:
        for r in range(args.nruns):
            call='python run.py '
            call+='--experiment='+e+' '
            call+='--approach='+a+' '
            call+='--seed='+str(r)+' '
            call+='--output='+args.res_folder+e+'_'+a+'_'+str(r)+'.txt'
            calls.append(call)

# GPU-processes
gpuproc=[]
for j in range(int(np.ceil(args.nproc/args.ngpus))):
    for i in range(args.ngpus):
        gpuproc.append((i,j,len(gpuproc)))

# Launch
mymap(calls,gpuproc)

print('Done')
print('[Elapsed time = {:.1f} h]'.format((time.time()-tstart)/(60*60)))
