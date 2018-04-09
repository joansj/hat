import sys,time
import numpy as np
import torch
from copy import deepcopy
import torch.nn.functional as F
from collections import defaultdict

import utils

class Appr(object):
    """ Class implementing the Incremental Moment Matching (mean) approach described in https://arxiv.org/abs/1703.08475 """

    def __init__(self,model,nepochs=100,sbatch=64,lr=0.05,lr_min=1e-4,lr_factor=3,lr_patience=5,clipgrad=10000,regularizer=0.0001,alpha=0.7,args=None):
        self.model=model
        self.model_old=None

        self.nepochs=nepochs
        self.sbatch=sbatch
        self.lr=lr
        self.lr_min=lr_min
        self.lr_factor=lr_factor
        self.lr_patience=lr_patience
        self.clipgrad=clipgrad

        self.ce = torch.nn.CrossEntropyLoss()
        self.optimizer=self._get_optimizer()

        self.reg=regularizer    # Grid search = [0.01,0.005,0.001,0.0005,0.0001,0.00005,0.000001]; best was 0.0001
        #self.alpha=alpha       # We assume the same alpha for all tasks. Unrealistic to tune one alpha per task (num_task-1 alphas) when we have a lot of tasks.
        if len(args.parameter)>=1:
            params=args.parameter.split(',')
            print('Setting parameters to',params)
            self.reg=float(params[0])
            #self.alpha=float(params[1])

        return

    def _get_optimizer(self,lr=None):
        if lr is None: lr=self.lr
        return torch.optim.SGD(self.model.parameters(),lr=lr)

    def train(self,t,xtrain,ytrain,xvalid,yvalid):
        best_loss=np.inf
        best_model=utils.get_model(self.model)
        lr=self.lr
        patience=self.lr_patience
        self.optimizer=self._get_optimizer(lr)

        # Loop epochs
        for e in range(self.nepochs):
            # Train
            clock0=time.time()
            self.train_epoch(t,xtrain,ytrain)
            clock1=time.time()
            train_loss,train_acc=self.eval(t,xtrain,ytrain)
            clock2=time.time()
            print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(
                e+1,1000*self.sbatch*(clock1-clock0)/xtrain.size(0),1000*self.sbatch*(clock2-clock1)/xtrain.size(0),train_loss,100*train_acc),end='')
            # Valid
            valid_loss,valid_acc=self.eval(t,xvalid,yvalid)
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss,100*valid_acc),end='')
            # Adapt lr
            if valid_loss<best_loss:
                best_loss=valid_loss
                best_model=utils.get_model(self.model)
                patience=self.lr_patience
                print(' *',end='')
            else:
                patience-=1
                if patience<=0:
                    lr/=self.lr_factor
                    print(' lr={:.1e}'.format(lr),end='')
                    if lr<self.lr_min:
                        print()
                        break
                    patience=self.lr_patience
                    self.optimizer=self._get_optimizer(lr)
            print()

        # Restore best, save model as old
        utils.set_model_(self.model,best_model)
        if t>0:
            model_state = utils.get_model(self.model)
            model_old_state = utils.get_model(self.model_old)
            for name, param in self.model.named_parameters():
                #model_state[name]=(1-self.alpha)*model_old_state[name]+self.alpha*model_state[name]
                model_state[name]=(model_state[name]+model_old_state[name]*t)/(t+1)
            utils.set_model_(self.model,model_state)

        self.model_old=deepcopy(self.model)
        utils.freeze_model(self.model_old)
        self.model_old.eval()


        return

    def train_epoch(self,t,x,y):
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

            # Forward current model
            outputs=self.model.forward(images)
            output=outputs[t]
            loss=self.criterion(output, targets, t)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.model.parameters(),self.clipgrad)
            self.optimizer.step()

        return

    def eval(self,t,x,y):
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
            outputs=self.model.forward(images)
            output=outputs[t]
            loss=self.criterion(output,targets,t)
            _,pred=output.max(1)
            hits=(pred==targets).float()

            # Log
            total_loss+=loss.data.cpu().numpy()[0]*len(b)
            total_acc+=hits.sum().data.cpu().numpy()[0]
            total_num+=len(b)

        return total_loss/total_num,total_acc/total_num

    def criterion(self, output, targets, t):

        # L2 multiplier loss
        loss_reg=0
        if t>0:
            for (name,param),(_,param_old) in zip(self.model.named_parameters(), self.model_old.named_parameters()):
                loss_reg += torch.sum((param_old-param).pow(2))/2

        # Cross entropy loss
        loss_ce=self.ce(output,targets)

        return loss_ce + self.reg*loss_reg
