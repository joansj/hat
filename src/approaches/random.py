import sys
import numpy as np
import torch

import utils

class Appr(object):

    def __init__(self,model,nepochs=0,sbatch=0,lr=0,lr_min=1e-4,lr_factor=3,lr_patience=5,clipgrad=10000,args=None):
        self.model=model

        self.criterion=None
        self.optimizer=None

        return

    def train(self,t,xtrain,ytrain,xvalid,yvalid):

        # Save labels
        self.labels=list(ytrain.cpu().numpy())

        return

    def eval(self,t,x,y):

        r=[]
        while len(r)<len(y):
            np.random.shuffle(self.labels)
            r+=self.labels
        r=np.array(r[:len(y)],dtype=int)
        np.random.shuffle(r)
        pred=torch.LongTensor(r).cuda()
        hits=(pred==y).float()

        return 0,hits.mean()
