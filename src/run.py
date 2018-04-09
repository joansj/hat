import sys,os,argparse,time
import numpy as np
import torch

import utils

tstart=time.time()

# Arguments
parser=argparse.ArgumentParser(description='xxx')
parser.add_argument('--seed',type=int,default=0,help='(default=%(default)d)')
parser.add_argument('--experiment',default='',type=str,required=True,choices=['mnist2','pmnist','cifar','mixture'],help='(default=%(default)s)')
parser.add_argument('--approach',default='',type=str,required=True,choices=['random','sgd','sgd-frozen','lwf','lfl','ewc','imm-mean','progressive','pathnet',
                                                                            'imm-mode','sgd-restart',
                                                                            'joint','hat','hat-test'],help='(default=%(default)s)')
parser.add_argument('--output',default='',type=str,required=False,help='(default=%(default)s)')
parser.add_argument('--nepochs',default=200,type=int,required=False,help='(default=%(default)d)')
parser.add_argument('--lr',default=0.05,type=float,required=False,help='(default=%(default)f)')
parser.add_argument('--parameter',type=str,default='',help='(default=%(default)s)')
args=parser.parse_args()
if args.output=='':
    args.output='../res/'+args.experiment+'_'+args.approach+'_'+str(args.seed)+'.txt'
print('='*100)
print('Arguments =')
for arg in vars(args):
    print('\t'+arg+':',getattr(args,arg))
print('='*100)

########################################################################################################################

# Seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)
else: print('[CUDA unavailable]'); sys.exit()

# Args -- Experiment
if args.experiment=='mnist2':
    from dataloaders import mnist2 as dataloader
elif args.experiment=='pmnist':
    from dataloaders import pmnist as dataloader
elif args.experiment=='cifar':
    from dataloaders import cifar as dataloader
elif args.experiment=='mixture':
    from dataloaders import mixture as dataloader

# Args -- Approach
if args.approach=='random':
    from approaches import random as approach
elif args.approach=='sgd':
    from approaches import sgd as approach
elif args.approach=='sgd-restart':
    from approaches import sgd_restart as approach
elif args.approach=='sgd-frozen':
    from approaches import sgd_frozen as approach
elif args.approach=='lwf':
    from approaches import lwf as approach
elif args.approach=='lfl':
    from approaches import lfl as approach
elif args.approach=='ewc':
    from approaches import ewc as approach
elif args.approach=='imm-mean':
    from approaches import imm_mean as approach
elif args.approach=='imm-mode':
    from approaches import imm_mode as approach
elif args.approach=='progressive':
    from approaches import progressive as approach
elif args.approach=='pathnet':
    from approaches import pathnet as approach
elif args.approach=='hat-test':
    from approaches import hat_test as approach
elif args.approach=='hat':
    from approaches import hat as approach
elif args.approach=='joint':
    from approaches import joint as approach

# Args -- Network
if args.experiment=='mnist2' or args.experiment=='pmnist':
    if args.approach=='hat' or args.approach=='hat-test':
        from networks import mlp_hat as network
    else:
        from networks import mlp as network
else:
    if args.approach=='lfl':
        from networks import alexnet_lfl as network
    elif args.approach=='hat':
        from networks import alexnet_hat as network
    elif args.approach=='progressive':
        from networks import alexnet_progressive as network
    elif args.approach=='pathnet':
        from networks import alexnet_pathnet as network
    elif args.approach=='hat-test':
        from networks import alexnet_hat_test as network
    else:
        from networks import alexnet as network

########################################################################################################################

# Load
print('Load data...')
data,taskcla,inputsize=dataloader.get(seed=args.seed)
print('Input size =',inputsize,'\nTask info =',taskcla)

# Inits
print('Inits...')
net=network.Net(inputsize,taskcla).cuda()
utils.print_model_report(net)

appr=approach.Appr(net,nepochs=args.nepochs,lr=args.lr,args=args)
print(appr.criterion)
utils.print_optimizer_config(appr.optimizer)
print('-'*100)

# Loop tasks
acc=np.zeros((len(taskcla),len(taskcla)),dtype=np.float32)
lss=np.zeros((len(taskcla),len(taskcla)),dtype=np.float32)
for t,ncla in taskcla:
    print('*'*100)
    print('Task {:2d} ({:s})'.format(t,data[t]['name']))
    print('*'*100)

    if args.approach == 'joint':
        # Get data. We do not put it to GPU
        if t==0:
            xtrain=data[t]['train']['x']
            ytrain=data[t]['train']['y']
            xvalid=data[t]['valid']['x']
            yvalid=data[t]['valid']['y']
            task_t=t*torch.ones(xtrain.size(0)).int()
            task_v=t*torch.ones(xvalid.size(0)).int()
            task=[task_t,task_v]
        else:
            xtrain=torch.cat((xtrain,data[t]['train']['x']))
            ytrain=torch.cat((ytrain,data[t]['train']['y']))
            xvalid=torch.cat((xvalid,data[t]['valid']['x']))
            yvalid=torch.cat((yvalid,data[t]['valid']['y']))
            task_t=torch.cat((task_t,t*torch.ones(data[t]['train']['y'].size(0)).int()))
            task_v=torch.cat((task_v,t*torch.ones(data[t]['valid']['y'].size(0)).int()))
            task=[task_t,task_v]
    else:
        # Get data
        xtrain=data[t]['train']['x'].cuda()
        ytrain=data[t]['train']['y'].cuda()
        xvalid=data[t]['valid']['x'].cuda()
        yvalid=data[t]['valid']['y'].cuda()
        task=t

    # Train
    appr.train(task,xtrain,ytrain,xvalid,yvalid)
    print('-'*100)

    # Test
    for u in range(t+1):
        xtest=data[u]['test']['x'].cuda()
        ytest=data[u]['test']['y'].cuda()
        test_loss,test_acc=appr.eval(u,xtest,ytest)
        print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.1f}% <<<'.format(u,data[u]['name'],test_loss,100*test_acc))
        acc[t,u]=test_acc
        lss[t,u]=test_loss

    # Save
    print('Save at '+args.output)
    np.savetxt(args.output,acc,'%.4f')

# Done
print('*'*100)
print('Accuracies =')
for i in range(acc.shape[0]):
    print('\t',end='')
    for j in range(acc.shape[1]):
        print('{:5.1f}% '.format(100*acc[i,j]),end='')
    print()
print('*'*100)
print('Done!')

print('[Elapsed time = {:.1f} h]'.format((time.time()-tstart)/(60*60)))

if hasattr(appr, 'logs'):
    if appr.logs is not None:
        #save task names
        from copy import deepcopy
        appr.logs['task_name'] = {}
        appr.logs['test_acc'] = {}
        appr.logs['test_loss'] = {}
        for t,ncla in taskcla:
            appr.logs['task_name'][t] = deepcopy(data[t]['name'])
            appr.logs['test_acc'][t]  = deepcopy(acc[t,:])
            appr.logs['test_loss'][t]  = deepcopy(lss[t,:])
        #pickle
        import gzip
        import pickle
        with gzip.open(os.path.join(appr.logpath), 'wb') as output:
            pickle.dump(appr.logs, output, pickle.HIGHEST_PROTOCOL)

########################################################################################################################
