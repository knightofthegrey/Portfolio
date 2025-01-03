#ArsRete_Models
#This is a file containing PyTorch objects for creating the ArsRete GAN

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchgan.layers import MinibatchDiscrimination1d
from time import time
from ArsRete_Datasets import Decode
import json

#__THE MAIN GAN OBJECT__

class GANModel():
    #Main wrapper class containing all the parameters of the neural network, and the training and evaluation code.
    #This is a very complicated class with a lot of tweakable parameters for use in hyperparameter tuning, so it can take a lot of arguments.
    #I have, at present, avoided giving default values for anything, since I don't have a clear understanding of what values work best and am still tuning this program overall.
    #However, some suggestions are given in the input notes to the __init__ function.
    
    def __init__(self,g_net,g_opt,d_net,d_opt,cost_params):
        #Takes inputs describing the parameters of the generator/discriminator network and optimizers, the cost function, and evaluation data.
        '''
        INPUTS:
        g_net,d_net: list of two lists, one (ints) of sizes, one (lists of three strings) of layer parameters. These must be valid inputs to the Network class.
        --Input/output sizes will vary based on how big the data you're planning on running the GANModel on is.
        
        g_opt,d_opt: List containing optimizer type (str) and optimizer parameters (list of 2-3 floats/tuples of floats). The possibilities and the PyTorch defaults are given below:
        --"Adam": learning rate, betas (tuple), l2 regularization. [0.001, (0.9, 0.99), 0]
        --"Adagrad": learning rate, lr decay, l2 regularization. [0.01, 0, 0]
        --"Adamax": learning rate, betas (tuple), l2 regularization. [0.002, (0.9, 0.99), 0]
        --"RMSprop": learning rate, alpha, l2 regularization. [0.01, 0.99, 0]
        --"SGD": learning rate, l2 regularization. [0.01, 0]
        
        cost_params: List of two strings, type and addon. The basic cost functions are "BCE", "BCEL", "MSE", and "KLD", which can be modified with "rs" (relativistic loss).
        This class also supports "WGAN" cost, which can be modified with "GP" (WC, SN not yet implemented)
        '''
        
        #FIRST: Metastring. These are strings describing the network architecture that are written to run log files to aid record-keeping.
        
        g_meta = "\n".join(["Gen:","|".join([str(a) for a in g_net[0]])," || ".join(["|".join(a) for a in g_net[1]]),"Opt: " + g_opt[0] + " " + str(g_opt[1])])
        d_meta = "\n".join(["Disc:","|".join([str(a) for a in d_net[0]])," || ".join(["|".join(a) for a in d_net[1]]),"Opt: " + d_opt[0] + " " + str(d_opt[1])])
        self.net_meta = "{}\n\n{}\n\n{}".format(" ".join(cost_params),g_meta,d_meta)
        
        #SECOND: Define generator
        self.gen = Network(g_net[0],g_net[1],f=False)
        
        if g_opt[0] == "Adam": self.g_opt = optim.Adam(self.gen.parameters(),lr = g_opt[1][0],betas = g_opt[1][1],weight_decay = g_opt[1][2])
        elif g_opt[0] == "Adagrad": self.g_opt = optim.Adagrad(self.gen.parameters(),lr = g_opt[1][0],lr_decay = g_opt[1][1],weight_decay = g_opt[1][2])
        elif g_opt[0] == "Adamax": self.g_opt = optim.Adamax(self.gen.parameters(),lr = g_opt[1][0],betas = g_opt[1][1],weight_decay = g_opt[1][2])
        elif g_opt[0] == "RMSprop": self.g_opt = optim.RMSprop(self.gen.parameters(), lr = g_opt[1][0],alpha = g_opt[1][1],weight_decay = g_opt[1][2])
        elif g_opt[0] == "SGD": self.g_opt = optim.SGD(self.gen.parameters(),lr = g_opt[1][0],weight_decay = g_opt[1][1])
        
        self.n_size = g_net[0][0]       
        
        #THIRD: Define discriminator
        self.disc = Network(d_net[0],d_net[1],f=True)
        
        if d_opt[0] == "Adam": self.d_opt = optim.Adam(self.gen.parameters(),lr = d_opt[1][0],betas = d_opt[1][1],weight_decay = d_opt[1][2])
        elif d_opt[0] == "Adagrad": self.d_opt = optim.Adagrad(self.gen.parameters(),lr = d_opt[1][0],lr_decay = d_opt[1][1],weight_decay = d_opt[1][2])
        elif d_opt[0] == "Adamax": self.d_opt = optim.Adamax(self.gen.parameters(),lr = d_opt[1][0],betas = d_opt[1][1],weight_decay = d_opt[1][2])
        elif d_opt[0] == "RMSprop": self.d_opt = optim.RMSprop(self.gen.parameters(), lr = d_opt[1][0],alpha = d_opt[1][1],weight_decay = d_opt[1][2])
        elif d_opt[0] == "SGD": self.d_opt = optim.SGD(self.gen.parameters(),lr = d_opt[1][0],weight_decay = d_opt[1][1])
        
        self.o_size = d_net[0][-1]
        
        #FOURTH: Set up cost function
        if cost_params[0] == "WGAN": self.cost = ""
        elif cost_params[0] == "BCE": self.cost = nn.BCELoss()
        elif cost_params[0] == "BCEL": self.cost = nn.BCEWithLogitsLoss()
        elif cost_params[0] == "MSE": self.cost = nn.MSELoss()
        elif cost_params[0] == "KLD": self.cost = nn.KLDivLoss()
        
        self.cost_mod = cost_params[1]
        if "GP" in self.cost_mod: self.gp = float(self.cost_mod[2:])
        else: self.gp = 0
        if "WC" in self.cost_mod: self.wc = float(self.cost_mod[2:])
        else: self.wc = 0
        
    def Train(self,in_data,epochs,ratio = (1,1),run_name = "rundump",state_name = ""):
        #Trains the network
        #This is a simple initial version with no visualization or evaluation
        run_list = []
        running_acc = []
        last = time()
        for a in range(epochs):
            errstate = [0,0]
            epoch_acc = 0
            for b,data in enumerate(in_data):
                inputs,labels = data
                for c in range(ratio[0]):
                    errstate[0] += self.DiscTrain(inputs)
                for c in range(ratio[1]):
                    errstate[1] += self.GenTrain(inputs)
                epoch_acc += self.Acc(inputs)
            #self.gen.eval()
            run_list.append([Decode(a) for a in self.Generate(200)])
            running_acc.append(epoch_acc / b)
            
            if ratio[1] == 0:
                print("{}: t {}, d {}, da {}".format(a,round(time()-last,4),round(errstate[0]/(ratio[0] * b),8),running_acc[-1]))
            else:
                print("{}: t {}, d {}, g {}, da {}".format(a,round(time() - last,4),round(errstate[0] / (ratio[0] * b),8),round(errstate[1]/(ratio[1] * b),8),running_acc[-1]))
            if state_name: self.gen.textState(state_name + " G " + str(a) + ".txt")
            last = time()
        
        dump = open("{}.txt".format(run_name).replace(".txt.txt",".txt"),"w")
        dump.write(self.net_meta + "\n\n")
        for a in run_list:
            dump.write("|".join(a) + "\n")
        dump.close()
            
    def GenTrain(self,in_data):
        #Run the generator and discriminator, but update only the generator
        #Zero gradients
        self.gen.zero_grad()
        self.disc.zero_grad()
        #self.disc.eval()
        #self.gen.train()
        
        #Generate size, data, and labels
        batch_size = len(in_data)
        if self.o_size == 1:
            real_labels = torch.full((batch_size,self.o_size),0.9)
        elif self.o_size == 2:
            real_labels = torch.tensor([[0.0,0.9] for a in range(batch_size)])
        fake_outputs = self.disc(self.Generate(batch_size))
        
        #Get the cost
        if self.cost: #If using a simple cost function:
            if self.cost_mod == "rs":
                #Relativistic cost: cost of fake - real against real labels
                real_outputs = self.disc(in_data)
                gen_err = self.cost((fake_outputs - real_outputs),real_labels)
            else:
                #Non-relativistic cost: Cost of fake against real labels
                gen_err = self.cost(fake_outputs,real_labels)
        else: #If using WGAN:
            #For the generator, WGAN cost is just the mean of the outputs
            gen_err = fake_outputs.mean()
        
        #Once we have our cost, backprop and update
        gen_err.backward()
        self.g_opt.step()
        
        #Return generator error in case we want to save that
        return gen_err.item()
    
    def DiscTrain(self,in_data):
        #Run the discriminator on real data and on fake data, updating each time.
        #Zero gradients
        self.gen.zero_grad()
        self.disc.zero_grad()
        #self.disc.train()
        #self.gen.eval()
        
        #Get size and labels
        batch_size = len(in_data)
        if self.o_size == 1:
            real_labels = torch.full((batch_size,self.o_size),0.9)
            fake_labels = torch.full((batch_size,self.o_size),0.0)
        elif self.o_size == 2:
            real_labels = torch.tensor([[0.0,0.9] for a in range(batch_size)])
            fake_labels = torch.tensor([[0.9,0.0] for a in range(batch_size)])
        
        
        #Run and get outputs
        real_output = self.disc(in_data)
        
        #self.disc.zero_grad()
        #self.gen.zero_grad()
        
        fake_input = self.Generate(batch_size)
        fake_output = self.disc(fake_input)
        
        #Get cost
        if self.cost:
            if self.cost_mod == "rs":
                disc_err = self.cost(real_output - fake_output,real_labels)
            else:
                disc_err = self.cost(real_output,real_labels) + self.cost(fake_output,fake_labels)
        else:
            if self.gp:
                disc_err = torch.mean(real_output) - torch.mean(fake_output) + self.GradPen(in_data,fake_input.detach()) * self.gp
            else:
                disc_err = torch.mean(real_output) - torch.mean(fake_output)
        
        
        #Backprop and step
        disc_err.backward()
        self.d_opt.step()
        
        if self.wc:
            #If we're using weight clipping:
            for p in self.disc.parameters():
                p.data.clamp_(-self.wc,self.wc)
        
        #Return discriminator error
        return disc_err.item()

    def Generate(self,samples):
        #Generate some words!
        return self.gen(torch.autograd.Variable(torch.randn((samples,self.n_size))))
    
    def GradPen(self,real,fake):
        #Gradient penalty for WGAN
        #I don't understand this very well, so I don't know exactly what this is doing, beyond "keep WGAN gradient small"
        batch_size = real.size()[0]
        
        #Interpolate
        alpha = torch.rand(batch_size,1).expand_as(real)
        interpolated = torch.autograd.Variable((alpha * real.data) + ((1-alpha) * fake.data),requires_grad = True)
        
        #Probability of interpolated
        p_interp = self.disc(interpolated)
        
        #Gradients of probabilities with respect to examples
        gradients = torch.autograd.grad(outputs = p_interp,inputs = interpolated,grad_outputs = torch.ones(p_interp.size()),create_graph = True,retain_graph = True)[0]
        
        #Flatten gradients
        gradients = gradients.view(batch_size,-1)
        
        #Gradients too close to 0 can cause problems
        gradients_norm = torch.sqrt(torch.sum(gradients**2,dim=1) + 1e-12)
        
        #Return gradient penalty
        return ((gradients_norm - 1) ** 2).mean()
    
    def Acc(self,in_data):
        #Attempt to get an accuracy value for the discriminator for analytical purposes
        #Get size and labels
        self.disc.eval()
        self.gen.eval()
        batch_size = len(in_data)
        #print(batch_size)
        if self.o_size == 1:
            real_labels = torch.full((batch_size,self.o_size),1)
            fake_labels = torch.full((batch_size,self.o_size),0)
        elif self.o_size == 2:
            real_labels = torch.tensor([[0,1] for a in range(batch_size)])
            fake_labels = torch.tensor([[1,0] for a in range(batch_size)])
        
        
        gen_data = self.Generate(batch_size)
        
        real_out = torch.round(self.disc(in_data))
        fake_out = torch.round(self.disc(gen_data))
        
        #print(torch.eq(real_out,real_labels).tolist())
        #print(torch.eq(fake_out,fake_labels).tolist())
        
        real_acc = torch.sum(torch.eq(real_out,real_labels)).item()/batch_size
        fake_acc = torch.sum(torch.eq(fake_out,fake_labels)).item()/batch_size
        
        return (real_acc+fake_acc)/2
        

#__BASIC MODULES__

class TRC(nn.Module):
    #TRC = "Tanh Range Correction"
    #Corrects tanh range (-1,1) to range 0-1
    #This is a PyTorch Module, and can be stuck together with other modules as needed
    #Preserves the relatively greater slope of tanh over sigmoid, but keeps the outputs in the 0-1 range.
    def __init__(self):
        super(TRC,self).__init__()
    def forward(self,x):
        return (1+x)/2

class RO(nn.Module):
    #RO = "Round Outputs"
    #Module to round all outputs from a network to 0 or 1
    #This is a PyTorch Module, and can be stuck together with other modules as needed
    #Used to make generator outputs look more like real-set inputs
    def __init__(self):
        super(RO,self).__init__()
    def forward(self,x):
        return torch.round(x)

class FZI(nn.Module):
    #FZI = "FuzZ Inputs"
    #Module to add small random noise to inputs. This makes neural network training more resilient.
    #This is a PyTorch Module, and can be stuck together with other modules as needed
    def __init__(self,magnitude = 0.001):
        super(FZI,self).__init__()
        self.fuzz = magnitude
    def forward(self,x):
        #Get noise by correcting torch.rand to values from -1 to 1, then multiplying by the fuzz magnitude
        #torch.rand normally generates values from 0 to 1, but we'd like the ability to have negative fuzz
        fuzz_tensor = (2*torch.rand(x.size()) - 1) * self.fuzz
        return torch.add(x,fuzz_tensor)

#__COMPOSITE MODULES__

class NetStep(nn.Module):
    #Element of a network module
    #Each "layer" of a NetSection always contains one Linear element, and may also contain an activation function, dropout, or a normalization layer
    #For ease of quick use we're going to implement a NetStep as an input size, output size, and three-element list of norm, dropout, activation in that order
    def __init__(self,in_sz,out_sz,steptype = [" "," "," "]):
        #A NetStep object has up to four elements: l = linear, n = normalization, d = dropout, a = activation
        super(NetStep,self).__init__()
        #Define the linear layer. Normalization may tweak this slightly.
        if steptype[0][0] == "m": self.l = nn.Linear(in_sz,out_sz - int(steptype[0][1:]))
        else: self.l = nn.Linear(in_sz,out_sz)
        #Norm layer. The Channels question in batchnorm may make this break and require extra steps, but we'll do what we can.
        if steptype[0][0] == "m": self.n = MinibatchDiscrimination1d(out_sz - int(steptype[0][1:]),int(steptype[0][1:]),int(steptype[0][1:]))
        elif steptype[0][0] == "b": self.n = nn.LazyBatchNorm1d()
        elif steptype[0][0] == "l": self.n = nn.LayerNorm(out_sz)
        elif steptype[0][0] == "i": self.n = nn.LazyInstanceNorm1d()
        else: self.n = ""
        #Dropout layer.
        if steptype[1][0] == "d": self.d = nn.Dropout(float(steptype[1][1]) / 10)
        else: self.d = ""
        #Activation layer.
        if steptype[2][0] == "s": self.a = nn.Sigmoid()
        elif steptype[2][0] == "t": self.a = nn.Sequential(nn.Tanh(),TRC())
        elif steptype[2][0] == "r": self.a = nn.ReLU()
        elif steptype[2][0] == "l": self.a = nn.LeakyReLU(float(steptype[2][1]) / 10)
        elif steptype[2][0] == "e": self.a = nn.ELU()
        else: self.a = ""
        '''
        #Initialization
        if steptype[2][0] in ["s","t"," "]:
            nn.init.xavier_normal_(self.l.weight)
            #nn.init.zeros_(self.l.bias)
        else:
            nn.init.kaiming_normal_(self.l.weight)
            #nn.init.zeros_(self.l.bias)'''
            
    def forward(self,x):
        #Run x through the linear layer, then through all other layers if present
        x = self.l(x)
        if self.n: x = self.n(x)
        if self.d: x = self.d(x)
        if self.a: x = self.a(x)
        return x

class Network(nn.Module):
    #General class containing a number of NetStep objects. This is mostly used for the GAN discriminator and generator here, but could be used for other networks
    #if needed.
    #Consists of a number of NetStep objects
    def __init__(self,sizelist,steplist,f = 0):
        #Inputs:
        '''
        sizelist: list of ints, layer sizes. Should be exactly one longer than steplist.
        steplist: list of [str,str,str] lists. Steptype inputs to NetStep, each string has values that mean something, see that class for details.
        '''
        super(Network,self).__init__()
        self.main = nn.Sequential()
        if f: self.main.append(FZI(f))
        for a in range(len(steplist)):
            self.main.append(NetStep(sizelist[a],sizelist[a+1],steplist[a]))
    
    def textState(self,out_file):
        #Writes the state of the module to a human-readable .txt file for visual inspection
        f = open(out_file,"w")
        
        for a in self.state_dict():
            f.write("{}\n".format(a))
            t = self.state_dict()[a].tolist()
            if type(t[0]) == list:
                t1 = "\n".join([" ".join([str(c) for c in b]) for b in t])
            else:
                t1 = " ".join([str(b) for b in t])
            f.write(t1)
            f.write("\n\n")
        
        f.close()
        #print([a for a in self.state_dict()])
        
    def forward(self,x):
        return self.main(x)