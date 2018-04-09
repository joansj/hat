import sys,time
import numpy as np
import torch
from copy import deepcopy

import utils

class Appr(object):

    def __init__(self,model,nepochs=100,sbatch=64,lr=0.05,lr_min=1e-4,lr_factor=3,lr_patience=5,clipgrad=10000,args=None):
        self.model=model
        self.initial_model = deepcopy(model)

        self.nepochs=nepochs
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
        return torch.optim.SGD(self.model.parameters(),lr=lr)

    def train(self,tasks,xtrain,ytrain,xvalid,yvalid):
        self.model=deepcopy(self.initial_model) # Restart model

        task_t,task_v=tasks
        best_loss=np.inf
        best_model=utils.get_model(self.model)
        lr=self.lr
        patience=self.lr_patience
        self.optimizer=self._get_optimizer(lr)

        # Loop epochs
        try:
            for e in range(self.nepochs):
                # Train
                clock0=time.time()
                self.train_epoch(task_t,xtrain,ytrain)
                clock1=time.time()
                train_loss=self.eval_validation(task_t,xtrain,ytrain)
                clock2=time.time()
                print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f} |'.format(e+1,1000*self.sbatch*(clock1-clock0)/xtrain.size(0),1000*self.sbatch*(clock2-clock1)/xtrain.size(0),train_loss),end='')
                # Valid
                valid_loss=self.eval_validation(task_v,xvalid,yvalid)
                print(' Valid: loss={:.3f} |'.format(valid_loss),end='')
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
        except KeyboardInterrupt:
            print()

        # Restore best
        utils.set_model_(self.model,best_model)

        return

    def train_epoch(self,t,x,y):
        self.model.train()

        r=np.arange(x.size(0))
        np.random.shuffle(r)
        r=torch.LongTensor(r)

        # Loop batches
        for i in range(0,len(r),self.sbatch):
            if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
            else: b=r[i:]
            images=torch.autograd.Variable(x[b],volatile=False).cuda()
            targets=torch.autograd.Variable(y[b],volatile=False).cuda()
            tasks=torch.autograd.Variable(t[b],volatile=False).cuda()

            # Forward
            outputs=self.model.forward(images)
            loss=self.criterion_train(tasks,outputs,targets)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.model.parameters(),self.clipgrad)
            self.optimizer.step()

        return

    def eval_validation(self,t,x,y):
        total_loss=0
        total_num=0
        self.model.eval()

        r=np.arange(x.size(0))
        r=torch.LongTensor(r)

        # Loop batches
        for i in range(0,len(r),self.sbatch):
            if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
            else: b=r[i:]
            images=torch.autograd.Variable(x[b],volatile=True).cuda()
            targets=torch.autograd.Variable(y[b],volatile=True).cuda()
            tasks=torch.autograd.Variable(t[b],volatile=True).cuda()

            # Forward
            outputs=self.model.forward(images)
            loss=self.criterion_train(tasks,outputs,targets)

            # Log
            total_loss+=loss.data.cpu().numpy()[0]*len(b)
            total_num+=len(b)

        return total_loss/total_num

    def eval(self,t,x,y):
        # This is used for the test. All tasks separately
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
            images=torch.autograd.Variable(x[b],volatile=True).cuda()
            targets=torch.autograd.Variable(y[b],volatile=True).cuda()

            # Forward
            outputs=self.model.forward(images)
            output=outputs[t]
            loss=self.criterion(output,targets)
            _,pred=output.max(1)
            hits=(pred==targets).float()

            # Log
            total_loss+=loss.data.cpu().numpy()[0]*len(b)
            total_acc+=hits.sum().data.cpu().numpy()[0]
            total_num+=len(b)

        return total_loss/total_num,total_acc/total_num

    def criterion_train(self,tasks,outputs,targets):
        loss=0
        for t in np.unique(tasks.data.cpu().numpy()):
            t=int(t)
            output=outputs[t]
            idx=(tasks==t).data.nonzero().view(-1)
            loss+=self.criterion(output[idx,:],targets[idx])*len(idx)
        return loss/targets.size(0)
