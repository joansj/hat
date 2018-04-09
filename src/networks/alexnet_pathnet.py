import sys
import torch
import numpy as np
import utils

class Net(torch.nn.Module):

    def __init__(self,inputsize,taskcla):
        super(Net,self).__init__()

        ncha,size,_=inputsize
        self.taskcla=taskcla
        self.ntasks = len(self.taskcla)

        """
        # Config of Sec 2.5 in the paper
        expand_factor = 0.231 # to match num params
        self.N = 5
        self.M = 20     # Large M numbers like this, given our architecture, produce no training
        #"""
        """
        # Config of Sec 2.4 in the paper
        expand_factor = 0.325 # match num params
        self.N = 3
        self.M = 10
        #"""
        #"""
        # Better config found by us
        expand_factor = 0.258 # match num params
        self.N = 3
        self.M = 16
        #"""
        self.L = 5      # our architecture has 5 layers

        self.bestPath = -1 * np.ones((self.ntasks,self.L,self.N),dtype=np.int) #we need to remember this between the tasks

        #init modules subnets
        self.conv1=torch.nn.ModuleList()
        self.sizec1 = int(expand_factor*64)

        self.conv2=torch.nn.ModuleList()
        self.sizec2 = int(expand_factor*128)

        self.conv3=torch.nn.ModuleList()
        self.sizec3 = int(expand_factor*256)

        self.fc1=torch.nn.ModuleList()
        self.sizefc1 = int(expand_factor*2048)

        self.fc2=torch.nn.ModuleList()
        self.sizefc2 = int(expand_factor*2048)

        self.last=torch.nn.ModuleList()

        self.maxpool=torch.nn.MaxPool2d(2)
        self.relu=torch.nn.ReLU()

        self.drop1=torch.nn.Dropout(0.2)
        self.drop2=torch.nn.Dropout(0.5)

        #declare task columns subnets
        for j in range(self.M):
            self.conv1.append(torch.nn.Conv2d(ncha,self.sizec1,kernel_size=size//8))
            s=utils.compute_conv_output_size(size,size//8)
            s=s//2
            self.conv2.append(torch.nn.Conv2d(self.sizec1,self.sizec2,kernel_size=size//10))
            s=utils.compute_conv_output_size(s,size//10)
            s=s//2
            self.conv3.append(torch.nn.Conv2d(self.sizec2,self.sizec3,kernel_size=2))
            s=utils.compute_conv_output_size(s,2)
            s=s//2

            self.fc1.append(torch.nn.Linear(self.sizec3*s*s,self.sizefc1))
            self.fc2.append(torch.nn.Linear(self.sizefc1,self.sizefc2))

        for t,n in self.taskcla:
            self.last.append(torch.nn.Linear(self.sizefc2,n))

        return

    def forward(self,x,t,P=None):
        if P is None:
            P = self.bestPath[t]
        # P is the genotype path matrix shaped LxN(no.layers x no.permitted modules)

        h=self.maxpool(self.drop1(self.relu(self.conv1[P[0,0]](x))))
        for j in range(1,self.N):
            h = h + self.maxpool(self.drop1(self.relu(self.conv1[P[0,j]](x)))) #sum activations

        h_pre=self.maxpool(self.drop1(self.relu(self.conv2[P[1,0]](h))))
        for j in range(1,self.N):
            h_pre = h_pre + self.maxpool(self.drop1(self.relu(self.conv2[P[1,j]](h)))) #sum activations
        h = h_pre

        h_pre=self.maxpool(self.drop2(self.relu(self.conv3[P[2,0]](h))))
        for j in range(1,self.N):
            h_pre = h_pre + self.maxpool(self.drop2(self.relu(self.conv3[P[2,j]](h)))) #sum activations
        h=h_pre.view(x.size(0),-1)

        h_pre=self.drop2(self.relu(self.fc1[P[3,0]](h)))
        for j in range(1,self.N):
            h_pre = h_pre + self.drop2(self.relu(self.fc1[P[3,j]](h))) #sum activations
        h = h_pre

        h_pre=self.drop2(self.relu(self.fc2[P[4,0]](h)))
        for j in range(1,self.N):
            h_pre = h_pre + self.drop2(self.relu(self.fc2[P[4,j]](h))) #sum activations
        h = h_pre

        y=[]
        for t,i in self.taskcla:
            y.append(self.last[t](h))
        return y

    def unfreeze_path(self,t,Path):
        #freeze modules not in path P and the ones in bestPath paths for the previous tasks
        for i in range(self.M):
            self.unfreeze_module(self.conv1,i,Path[0,:],self.bestPath[0:t,0,:])
            self.unfreeze_module(self.conv2,i,Path[1,:],self.bestPath[0:t,1,:])
            self.unfreeze_module(self.conv3,i,Path[2,:],self.bestPath[0:t,2,:])
            self.unfreeze_module(self.fc1,i,Path[3,:],self.bestPath[0:t,3,:])
            self.unfreeze_module(self.fc2,i,Path[4,:],self.bestPath[0:t,4,:])
        return

    def unfreeze_module(self,layer,i,Path,bestPath):
        if (i in Path) and (i not in bestPath): #if the current module is in the path and not in the bestPath
            utils.set_req_grad(layer[i],True)
        else:
            utils.set_req_grad(layer[i],False)
        return


