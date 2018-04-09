import sys,time
import numpy as np
import torch
import random

import utils
from copy import deepcopy

class Appr(object):
    # Based on paper and largely on https://github.com/dai-dao/pathnet-pytorch and https://github.com/kimhc6028/pathnet-pytorch

    def __init__(self,model,nepochs=100,sbatch=64,lr=0.05,lr_min=1e-4,lr_factor=3,lr_patience=5,clipgrad=1000,generations=20,args=None):
        self.model=model
        self.initial_model=deepcopy(model)

        #architecture hyperparams (must be the same as alexnet_pathnet)
        self.N = self.model.N   # from paper, number of distinct modules permitted in a pathway
        self.M = self.model.M   # from paper, total num modules
        self.L = self.model.L   # layers with paths in the network

        self.ntasks = self.model.ntasks

        self.generations = generations       # Grid search = [5,10,20,50,100,200]; best was 20
        self.P = 2              # from paper Secs 2.4 and 2.5, numbers of the individuals in each generation/paths to be trained

        if len(args.parameter)>=1:
            params=args.parameter.split(',')
            print('Setting parameters to',params)
            self.generations=int(params[0])

        self.nepochs=nepochs//self.generations   # To maintain same number of training updates
        self.sbatch=sbatch
        self.lr=lr
        self.lr_min=lr_min
        self.lr_factor=lr_factor
        self.lr_patience=lr_patience
        self.clipgrad=clipgrad

        self.criterion=torch.nn.CrossEntropyLoss()
        self.optimizer=self._get_optimizer()

        return

    def _get_optimizer(self,lr=None):
        if lr is None: lr=self.lr
        return torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),lr=lr)

    def train(self,t,xtrain,ytrain,xvalid,yvalid):

        if t>0: #reinit modules not in bestpath with random, according to the paper
            layers = ['conv1','conv2','conv3','fc1','fc2']
            for (n,p),(m,q) in zip(self.model.named_parameters(),self.initial_model.named_parameters()):
                if n==m:
                    layer,module,par = n.split(".")
                    module = int(module)
                    if layer in layers:
                        if module not in self.model.bestPath[0:t,layers.index(layer)]:
                            p.data = deepcopy(q.data)

        #init path for this task
        Path = np.random.randint(0,self.M-1,size=(self.P,self.L,self.N))
        guesses = list(range(self.M))
        lr=[]
        patience=[]
        best_loss=[]
        for p in range(self.P):
            lr.append(self.lr)
            patience.append(self.lr_patience)
            best_loss.append(np.inf)
            for j in range(self.L):
                np.random.shuffle(guesses)
                Path[p,j,:] = guesses[:self.N] #do not repeat modules

        winner = 0
        best_path_model = utils.get_model(self.model)
        best_loss_overall=np.inf

        try:
            for g in range(self.generations):
                if np.max(lr)<self.lr_min: break

                for p in range(self.P):
                    if lr[p]<self.lr_min: continue

                    # train only the modules in the current path, minus the ones in the model.bestPath
                    self.model.unfreeze_path(t,Path[p])

                    # the optimizer trains solely the params for the current task
                    self.optimizer=self._get_optimizer(lr[p])

                    # Loop epochs
                    for e in range(self.nepochs):
                        # Train
                        clock0=time.time()
                        self.train_epoch(t,xtrain,ytrain,Path[p])
                        clock1=time.time()
                        train_loss,train_acc=self.eval(t,xtrain,ytrain,Path[p])
                        clock2=time.time()
                        print('| Generation {:3d} | Path {:3d} | Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(
                            g+1,p+1,e+1,1000*self.sbatch*(clock1-clock0)/xtrain.size(0),1000*self.sbatch*(clock2-clock1)/xtrain.size(0),train_loss,100*train_acc),end='')
                        # Valid
                        valid_loss,valid_acc=self.eval(t,xvalid,yvalid,Path[p])
                        print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss,100*valid_acc),end='')

                        # Save the winner
                        if valid_loss<best_loss_overall:
                            best_loss_overall=valid_loss
                            best_path_model = utils.get_model(self.model)
                            winner=p
                            print(' B',end='')

                        # Adapt lr
                        if valid_loss<best_loss[p]:
                            best_loss[p]=valid_loss
                            patience[p]=self.lr_patience
                            print(' *',end='')
                        else:
                            patience[p]-=1
                            if patience[p]<=0:
                                lr[p]/=self.lr_factor
                                print(' lr={:.1e}'.format(lr[p]),end='')
                                if lr[p]<self.lr_min:
                                    print()
                                    break
                                patience[p]=self.lr_patience
                        print()

                # Restore winner model
                utils.set_model_(self.model,best_path_model)
                print('| Winning path: {:3d} | Best loss: {:.3f} |'.format(winner+1,best_loss_overall))

                # Keep the winner and mutate it
                print('Mutating')
                probability = 1/(self.N*self.L) #probability to mutate
                for p in range(self.P):
                    if p!=winner:
                        best_loss[p]=np.inf
                        lr[p]=lr[winner]
                        patience[p]=self.lr_patience
                        for j in range(self.L):
                            for k in range(self.N):
                                Path[p,j,k]=Path[winner,j,k]
                                if np.random.rand()<probability:
                                    Path[p,j,k]=(Path[p,j,k]+np.random.randint(-2,3))%self.M # add int in [-2,2] to the path, this seems yet another hyperparam

        except KeyboardInterrupt:
            print()

        #save the best path into the model
        self.model.bestPath[t] = Path[winner]
        print(self.model.bestPath[t])

        return

    def train_epoch(self,t,x,y,Path):
        self.model.train()

        r=np.arange(x.size(0))
        np.random.shuffle(r)
        r=torch.LongTensor(r).cuda()

        # Loop batches
        for i in range(0,len(r),self.sbatch):
            if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
            else: b=r[i:]
            images=torch.autograd.Variable(x[b],volatile=False)
            targets=torch.autograd.Variable(y[b],volatile=False)

            # Forward
            outputs=self.model.forward(images,t,Path)
            output=outputs[t]
            loss=self.criterion(output,targets)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(filter(lambda p: p.requires_grad, self.model.parameters()),self.clipgrad)
            self.optimizer.step()

        return

    def eval(self,t,x,y,Path=None):
        total_loss=0
        total_acc=0
        total_num=0
        self.model.eval()

        r=np.arange(x.size(0))
        r=torch.LongTensor(r).cuda()

        # Loop batches
        for i in range(0,len(r),self.sbatch):
            if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
            else: b=r[i:]
            images=torch.autograd.Variable(x[b],volatile=True)
            targets=torch.autograd.Variable(y[b],volatile=True)

            # Forward
            outputs=self.model.forward(images,t,Path)
            output=outputs[t]
            loss=self.criterion(output,targets)
            _,pred=output.max(1)
            hits=(pred==targets).float()

            # Log
            total_loss+=loss.data.cpu().numpy()[0]*len(b)
            total_acc+=hits.sum().data.cpu().numpy()[0]
            total_num+=len(b)

        return total_loss/total_num,total_acc/total_num
