#Ars_TorchGen
#Implements various AGN structures using Pytorch
#Allows for a range of architectures and structures for testing purposes

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torchgan.layers import MinibatchDiscrimination1d
from time import time
import math
import json
import matplotlib.pyplot as plt
import numpy as np

#__GLOBALS__

#wlg = 12

#__DATASET__

class unifiedWordset(Dataset):
    #Unifies earlier versions of the wordsdataset
    #This set usually uses labels in the GAN class rather than its own, but can use its own if needed
    def __init__(self,words_paths,voflag = False,dflag = False,dw = 1):
        super(unifiedWordset,self).__init__()
        self.data = []
        self.labels = []
        big_set = []
        for a in words_paths:
            temp_dict = json.load(open(a,"r"))
            for b in temp_dict:
                if b.strip().isalpha():
                    big_set.append(b)
                    try:
                        self.data.append([float(c) for c in encode_v2(b)])
                        if temp_dict[b] in [0,1]: self.labels.append([float(int(temp_dict[b] == 0)),float(int(temp_dict[b] == 1))])
                        else: self.labels.append([float(c) for c in temp_dict[b]])
                    except: pass
        self.datalen = len(self.data[0])
        self.data = torch.tensor(self.data)
        self.labels = torch.tensor(self.labels)
        if voflag: self.distros = [lvector(big_set)]
        elif dflag: self.distros = [newVvector(big_set,self.datalen//5),lgf(big_set,[dw],self.datalen//5)[0]]
        else: self.distros = [lvector(big_set),qvector(big_set),vvector(big_set,self.datalen//5)]
    def __len__(self):
        return self.data.size()[0]
    def __getitem__(self,index):
        return self.data[index],self.labels[index]

#__BASIC MODULES__

class TRC(nn.Module):
    #TRC = "Tanh Range Correction"
    #Corrects tanh range (-1,1) to range 0-1
    #Preserves the relatively greater slope of tanh over sigmoid, but keeps the outputs in the 0-1 range.
    def __init__(self):
        super(TRC,self).__init__()
    def forward(self,x):
        return (1+x)/2

class RO(nn.Module):
    #RO = "Round Outputs"
    #Module to round all outputs from a network to 0 or 1
    #Used to make generator outputs look more like real-set inputs
    def __init__(self):
        super(RO,self).__init__()
    def forward(self,x):
        return torch.round(x)

class FZI(nn.Module):
    #FZI = "FuzZ Inputs"
    #Module to add small random noise to inputs
    #Somehow this seems like it might be useful
    def __init__(self,magnitude = 0.001):
        super(FZI,self).__init__()
        self.fuzz = magnitude
    def forward(self,x):
        #Get noise by correcting torch.rand to values from -1 to 1, then multiplying by the fuzz magnitude
        fuzz_tensor = (2*torch.rand(x.size()) - 1) * self.fuzz
        return torch.add(x,fuzz_tensor)

class NetStep(nn.Module):
    #Element of a network module
    #Each "layer" of a NetSection always contains one Linear element, and may also contain an activation function, dropout, or a normalization layer
    #For ease of quick use we're going to implement a NetStep as an input size, output size, and three-element list of norm, dropout, activation in that order
    def __init__(self,in_sz,out_sz,steptype = [" "," "," "]):
        super(NetStep,self).__init__()
        #Define the linear layer. Normalization may tweak this slightly.
        if steptype[0][0] == "m": self.l = nn.Linear(in_sz,out_sz - int(steptype[0][1:]))
        else: self.l = nn.Linear(in_sz,out_sz)
        #Norm layer. The Channels question in batchnorm may make this break and require extra steps, but we'll do what we can.
        if steptype[0][0] == "m": self.n = MinibatchDiscrimination1d(out_sz - int(steptype[0][1:]),int(steptype[0][1:]),int(steptype[0][1:]))
        elif steptype[0][0] == "b": self.n = nn.LazyBatchNorm1d()
        elif steptype[0][0] == "l": self.n = nn.LayerNorm(int(steptype[0][1:]))
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
    
    def forward(self,x):
        #Run x through the linear layer, then through all other layers if present
        x = self.l(x)
        if self.n: x = self.n(x)
        if self.d: x = self.d(x)
        if self.a: x = self.a(x)
        return x

class Network(nn.Module):
    #One class that can be used for generator or for discriminator
    #Consists of a number of NetStep objects
    def __init__(self,sizelist,steplist,f = False):
        #Sizelist must be one longer than steplist
        super(Network,self).__init__()
        self.main = nn.Sequential()
        if f: self.main.append(FZI(0.001))
        for a in range(len(steplist)):
            self.main.append(NetStep(sizelist[a],sizelist[a+1],steplist[a]))
        
    def forward(self,x):
        return self.main(x)
    
#__MAIN MODEL CLASSES__

class EnsembleModel():
    #Wrapper class containing several GANs to control producing output of varying size
    def __init__(self,g_params,d_params,cost_params,load_wordset = "commonl.json"):
        #When defined load_wordset may be a .json or .txt path, from which the model will create unifiedWordset objects, or a prefix, from which the model will load premade wordsets
        if ".json" in load_wordset:
            word_list = json.load(open(load_wordset,"r"))

class GANModel():
    #Wrapper class containing generator, discriminator, cost, and optimizer
    
    def __init__(self,g_params,d_params,cost_params,rdistros = [],vdistros = []):
        #Takes in lists describing generator parameters, discriminator parameters, cost parameters
        #Generator/discriminator params are: sizelist, steplist, optimizer type, optimizer parameters (three floats or tuples of floats)
        #For speed of use, some defaults:
        '''
        #Adam: [0.001, (0.9,0.99), 0]
        #Adagrad: [0.01, 0, 0]
        #Adamax: [0.002, (0.9,0.99), 0]
        #RMSprop: [0.01, 0.99, 0]
        #SGD: [0.01, 0]
        '''
        
        '''gen_params = [
        [data_2.datalen,200,128,data_2.datalen],
        [[" "," ","l1"],[" "," ","l1"],[" "," ","t"]],
        "Adam",
        [0.001, (0.8,0.99), 0]
        ]'''
        
        g_metadata = "\n".join(["Gen:","|".join([str(a) for a in g_params[0]]),"||".join(["|".join(a) for a in g_params[1]]),
                      "Opt: " + g_params[2] + " " + str(g_params[3])])
        
        d_metadata = "\n".join(["Disc:","|".join([str(a) for a in d_params[0]]),"||".join(["|".join(a) for a in d_params[1]]),
                      "Opt: " + d_params[2] + " " + str(d_params[3])])
        
        self.metadata = "{}\n\n{}\n\n{}".format(cost_params,g_metadata,d_metadata)
        
        #Define generator
        self.gen = Network(g_params[0],g_params[1])
        if g_params[2] == "Adam": self.gen_opt = optim.Adam(self.gen.parameters(),lr = g_params[3][0],betas = g_params[3][1],weight_decay = g_params[3][2])
        elif g_params[2] == "Adagrad": self.gen_opt = optim.Adagrad(self.gen.parameters(),lr = g_params[3][0],lr_decay = g_params[3][1],weight_decay = g_params[3][2])
        elif g_params[2] == "Adamax": self.gen_opt = optim.Adamax(self.gen.parameters(),lr = g_params[3][0],betas = g_params[3][1],weight_decay = g_params[3][2])
        elif g_params[2] == "RMSprop": self.gen_opt = optim.RMSprop(self.gen.parameters(), lr = g_params[3][0],alpha = g_params[3][1],weight_decay = g_params[3][2])
        elif g_params[2] == "SGD": self.gen_opt = optim.SGD(self.gen.parameters(),lr = g_params[3][0],weight_decay = g_params[3][1])
        self.n_size = g_params[0][0]
        
        #Define discriminator
        self.disc = Network(d_params[0],d_params[1],f = True)
        if d_params[2] == "Adam": self.disc_opt = optim.Adam(self.disc.parameters(),lr = d_params[3][0],betas = d_params[3][1],weight_decay = d_params[3][2])
        elif d_params[2] == "Adagrad": self.disc_opt = optim.Adagrad(self.disc.parameters(),lr = d_params[3][0],lr_decay = d_params[3][1],weight_decay = d_params[3][2])
        elif d_params[2] == "Adamax": self.disc_opt = optim.Adamax(self.disc.parameters(),lr = d_params[3][0],betas = d_params[3][1],weight_decay = d_params[3][2])
        elif d_params[2] == "RMSprop": self.disc_opt = optim.RMSprop(self.disc.parameters(), lr = d_params[3][0],alpha = d_params[3][1],weight_decay = d_params[3][2])
        elif d_params[2] == "SGD": self.disc_opt = optim.SGD(self.disc.parameters(),lr = d_params[3][0],weight_decay = d_params[3][1])
        
        #If we're using a cost function
        if "WGN" in cost_params:
            self.cost = ""
            if "GP" in cost_params: self.cost += "G" #Indicates the use of gradient penalty
            if "WC" in cost_params: self.cost += "C" #Indicates the use of weight clipping
            if "SN" in cost_params: self.cost += "S" #Indicates the use of spectral norm
            
        elif cost_params == "BCE": self.cost = nn.BCELoss()
        elif cost_params == "BCEL": self.cost = nn.BCEWithLogitsLoss()
        elif cost_params == "MSE": self.cost = nn.MSELoss()
        elif cost_params == "KLD": self.cost = nn.KLDivLoss()
        
        #If we have reference distributions for grading progress:
        self.rdistros = rdistros
        self.vdistros = vdistros
        
        #Keep a list handy to record run performance
        self.run_list = []
        self.eval_list = []
    
    def GenSamples(self,num_samples):
        noise = torch.autograd.Variable(torch.randn((num_samples,self.n_size)))
        return self.gen(noise)
    
    def GradPen(self,real,fake,gpw):
        #Gradient penalty function for WGAN costs
        b_sz = real.size()[0]
        
        #Interpolate
        alpha = torch.rand(b_sz,1).expand_as(real)
        interpolated = torch.autograd.Variable(alpha * real.data + (1-alpha) * fake,requires_grad = True)
        
        #Probability of interpolated examples
        p_interpolated = self.disc(interpolated)
        
        #Gradients of probabilities with respect to examples
        gradients = torch.autograd.grad(outputs = p_interpolated,inputs = interpolated,grad_outputs = torch.ones(p_interpolated.size()),create_graph = True,retain_graph = True)[0]
        
        #Flatten gradients
        gradients = gradients.view(b_sz,-1)
        
        #Derivatives close to 0 can cause problems?
        gradients_norm = torch.sqrt(torch.sum(gradients**2,dim=1) + 1e-12)
        
        #Return gradient penalty
        return gpw * ((gradients_norm - 1) **2).mean()
        
    
    def GenLearnBasic(self,in_data,cost_type = ""):
        #Runs the generator and discriminator on a sample of size batch_size, then updates only the generator
        #The basic version is for a non-Wasserstein GAN and uses a normal cost function with no shenanigans
        #Zero gradients
        self.gen.zero_grad()
        self.disc.zero_grad()
        batch_size = len(in_data)
        #Generate fake data and real labels
        real_labels = torch.full((batch_size,1),0.9)
        fake_outputs = self.disc(self.GenSamples(batch_size))
        if cost_type == "rs":
            #If using RSGAN: cost is the cost of the fake set minus the real set against 1s
            real_outputs = self.disc(in_data)
            gen_err = self.cost((fake_outputs - real_outputs),real_labels)
        elif cost_type == "wg":
            #If using Wasserstein GAN: cost on the generator is just the mean value of the outputs
            gen_err = -fake_outputs.mean()
        else:
            #If not using RS or WGAN, just check fake results against real labels
            #Get the cost of the fake results against the real labels
            gen_err = self.cost(fake_outputs,real_labels)
        #Backprop and update
        gen_err.backward()
        self.gen_opt.step()
        #Append error to most recent run in run_list
        self.run_list[-1][0].append(gen_err.item())
        
    def DiscLearnBasic(self,in_data,cost_type = "",gpw = 3):
        #Runs the discriminator on generator outputs and on real inputs, then updates the discriminator
        #The basic version is for a non-Wasserstein GAN and uses a normal cost function with no shenanigans
        #Zero gradients
        self.gen.zero_grad()
        self.disc.zero_grad()
        #Generate labels
        batch_size = len(in_data)
        real_labels = torch.full((batch_size,1),0.9)
        fake_labels = torch.full((batch_size,1),0.0)
        
        #Run on real data first
        real_output = self.disc(in_data)
        real_error = self.cost(real_output,real_labels)
        #real_error.backward()
        #self.disc_opt.step()
        
        #Zero gradients again, then run on fake data
        self.gen.zero_grad()
        self.disc.zero_grad()
        fake_samples = self.GenSamples(batch_size).detach()
        fake_output = self.disc(fake_samples)
        fake_error = self.cost(fake_output,fake_labels)
        #fake_error.backward()
        #self.disc_opt.step()
        
        if cost_type == "rs":
            disc_error = self.cost(real_output - fake_output,real_labels)
            
        elif cost_type == "wg":
            #If using WGAN, our error if the difference in the mean of the results minus grad-penalty
            disc_error = fake_output.mean() - real_output.mean() - self.GradPen(in_data,fake_samples,gpw)
        
        else:
            disc_error = real_error + fake_error
        
        disc_error.backward()
        self.disc_opt.step()
        
        #Append average of real and fake error to most recent run_list item
        self.run_list[-1][1].append(disc_error.item())
    
    def FastRun(self,in_data,loops,ratio = (1,2),eval_time = 10,run_name = "rundump",cost_type = "",gpw = 3,wl = "",dw = 1):
        #Rather than slowing ourselves down by running the eval process every batch we'll try running it every few epochs instead, see what happens.
        last_eval = time()
        for a in range(loops):
            self.run_list.append([[],[]])
            for b,data in enumerate(in_data):
                inputs,labels = data
                for c in range(ratio[0]):
                    self.DiscLearnBasic(inputs,cost_type,gpw)
                for c in range(ratio[1]):
                    self.GenLearnBasic(inputs,cost_type)
            
            if time() - last_eval >= eval_time:
                last_eval = time()
                self.eval_list.append([a,[],[]])
                self.gen.eval()
                gen_out = self.GenSamples(1000)
                
                if self.rdistros and len(self.rdistros[0]) == 3:
                    samples = [decode_v2(c.tolist()) for c in gen_out]
                    self.eval_list[-1][1].append(samples[:10])
                    batch_distros = vectors(samples,wl)
                    self.eval_list[-1][2].append([vectorDistance(batch_distros[a],self.rdistros[0][a],self.rdistros[1][a]) for a in range(3)])
                
                elif self.rdistros and len(self.rdistros[0]) == 2:
                    samples = [decode_v2(c.tolist()) for c in gen_out]
                    self.eval_list[-1][1].append(samples[:10])
                    batch_distros = [newVvector(samples,wl),lgf(samples,[dw],wl)[0]]
                    self.eval_list[-1][2].append([dictCompare(batch_distros[a],self.rdistros[0][a],self.rdistros[1][a]) for a in range(2)])
                
                print("Epoch {}: {} Gen, {} Disc".format(a,self.run_list[-1][0][-1],self.run_list[-1][1][-1]))
                if self.rdistros:
                    if len(self.rdistros[0]) == 3:
                        print("Vd: letterv {}, qv {}, vv {}".format(*self.eval_list[-1][2][0]))
                    elif len(self.rdistros[0]) == 2:
                        print("Dd: vdict {}, ldict {}".format(*self.eval_list[-1][2][0]))
                    print("Words: {}".format(self.eval_list[-1][1][:5]))
                self.gen.train()
        self.quickSave(run_name)
    
    def quickSave(self,run_name):
        t1 = open("ArsOut/{}.txt".format(run_name),"w")
        t1.write("{}\n\n{}\n\n".format(run_name,self.metadata))
        evdex = 0
        for a in range(len(self.run_list)):
            t1.write("Epoch {}\n".format(a))
            t1.write("GenE||" + "|".join([str(b) for b in self.run_list[a][0]]) + "\n")
            t1.write("DiscE||" + "|".join([str(b) for b in self.run_list[a][1]]) + "\n")
            try:
                if self.eval_list[evdex][0] == a:
                    t1.write("QualVs||" + "||".join(["|".join([str(b) for b in c]) for c in self.eval_list[evdex][2]]) + "\n")
                    t1.write("Words||" + "||".join(["|".join([b for b in c[:10]]) for c in self.eval_list[evdex][1]]) + "\n")
                    evdex += 1
            except:
                pass
            t1.write("\n")
        t1.close()
        
        
    def TrainBasic(self,in_data,loops,ratio = (1,2),vis = False,print_time = 5,run_name = "rundump",cost_type = "",gpw = 3,wl = "",dw = 1):
        #Basic non-WGAN nn training
        last_print = time()
        for a in range(loops):
            #Append four-element list for discriminator error, generator error, sample outputs, and eval data to run_list for the epoch
            self.run_list.append([[],[],[],[]])
            for b,data in enumerate(in_data):
                #For each mini-batch:
                #Split the batch into inputs and labels
                #While training a GAN we don't use the labels, but they're in the dataset so we need to separate them out
                inputs,labels = data
                
                for c in range(ratio[0]):
                    #Train discriminator first, a number of times according to the ratio
                    self.DiscLearnBasic(inputs,cost_type,gpw)
                for c in range(ratio[1]):
                    #Train generator next, a number of times according to the ratio
                    self.GenLearnBasic(inputs,cost_type)
                
                self.gen.eval() #Set to eval to ignore any dropout layers in the generator
                batch_gen_out = self.GenSamples(1000)
                
                if self.rdistros and len(self.rdistros[0]) == 3:
                    #If we have rdistros we'll have a list of three vectors and three weights, l, q, and v distances, respectively.
                    #Vectors are from the original distribution, weights are the distance between the original distribution and random gibberish
                    #Ideally this gives us some kind of idea whether our outputs are better than gibberish (<1), worse than gibberish (>1)
                    batch_samples = [decode_v2(c.tolist()) for c in batch_gen_out]
                    self.run_list[-1][2].append(batch_samples)                    
                    batch_distros = vectors(batch_samples,wl)
                    self.run_list[-1][3].append([vectorDistance(batch_distros[a],self.rdistros[0][a],self.rdistros[1][a]) for a in range(3)])
                
                elif self.rdistros and len(self.rdistros[0]) == 2:
                    #Version 3 eval metric: scrap the q-metric and l-distance, v-distance and l-follow distances
                    batch_samples = [decode_v2(c.tolist()) for c in batch_gen_out]
                    self.run_list[-1][2].append(batch_samples)
                    batch_distros = [newVvector(batch_samples,wl),lgf(batch_samples,[dw],wl)[0]]
                    self.run_list[-1][3].append([dictCompare(batch_distros[a],self.rdistros[0][a],self.rdistros[1][a]) for a in range(2)])
                
                elif self.rdistros and len(self.rdistros[0]) == 1:
                    #If we're trying the v-only version?
                    batch_samples = [decode_v2(c.tolist()) for c in batch_gen_out]
                    self.run_list[-1][2].append(batch_samples)
                    batch_distros = lvector(batch_samples)
                    self.run_list[-1][3].append([vectorDistance(batch_distros,self.rdistros[0][0],self.rdistros[1][0])])
                    
                elif self.vdistros:
                    #If we're using vdistros we'll have a mean vector of the dataset and a weight
                    run_metric = oDist(batch_gen_out,self.vdistros[0],self.vdistros[1])
                    self.run_list[-1][2].append([decode_v2(c.tolist()) for c in batch_gen_out[:5]])
                    self.run_list[-1][3].append(run_metric)
                self.gen.train() #Set to train to turn dropout layers back on
                    
            if time() - last_print > print_time:
                print("Epoch {}: Gen {}, Disc {}".format(a,self.run_list[-1][0][-1],self.run_list[-1][1][-1]))
                if self.rdistros: 
                    if len(self.rdistros[0]) == 3: print("Vdist: l {:.4f}, q {:.4f}, v {:.4f}".format(*self.run_list[-1][3][-1]))
                    elif len(self.rdistros[0]) == 2: print("Ddist: vp {:.4f}, lf {:.4f}".format(*self.run_list[-1][3][-1]))
                    elif len(self.rdistros[0]) == 1: print("Vdist: {:.4f}".format(self.run_list[-1][3][-1][0]))
                elif self.vdistros: print("Vector metric: {:.4f}".format(self.run_list[-1][3][-1]))
                if self.run_list[-1][2]: print("Words: {}".format(self.run_list[-1][2][-1][:5]))
                last_print = time()
        self.saveRun(run_name)
                
    def saveRun(self,run_name):
        t1 = open("{}.txt".format(run_name),"w")
        t1.write("{}\n\n{}\n\n".format(run_name,self.metadata))
        for a in range(len(self.run_list)):
            t1.write("Epoch {}\n".format(a))
            t1.write("GenE||" + "|".join([str(b) for b in self.run_list[a][0]]) + "\n")
            t1.write("DiscE||" + "|".join([str(b) for b in self.run_list[a][1]]) + "\n")
            if self.rdistros:
                t1.write("QualVs||" + "||".join(["|".join([str(b) for b in c]) for c in self.run_list[a][3]]) + "\n")
            elif self.vdistros:
                t1.write("QualVs||" + "|".join([str(b) for b in self.run_list[a][3]]) + "\n")
            t1.write("Words||" + "||".join(["|".join([b for b in c[:10]]) for c in self.run_list[a][2]]) + "\n\n")
    
    def runQual(self):
        #Returns a convenient list of metrics on how well the run did (total and epoch lists for error, vector metrics, and average vector metrics)
        #Format of run_list: [run], run = [[generr],[discerr],[words],[[letter],[q],[vowel]]]
        #To get epoch lists of generr, discerr:
        epoch_lists = [[sum(a[b]) for a in self.run_list] for b in range(2)] + [[sum(a[3][b]) for a in self.run_list] for b in range(len(self.run_list[0][3]))]
        epoch_lists.append([sum([a[b] for a in epoch_lists[2:]]) / len(self.run_list[0][3]) for b in range(len(epoch_lists))])
        epoch_totals = [sum(a) for a in epoch_lists]
        return epoch_totals,epoch_lists #Return value: total generr, discerr, lv, qv, vv, avgv, list of same
                

#__UTILITY FUNCTIONS__

def encode(word):
    #Quick conversion of word to binary
    maxlen = int(math.log(27**len(word)) / math.log(2)) + 1
    #Make base-27 list
    t1 = [ord(a) - 96 if a != " " else 0 for a in word.lower()]
    #Make decimal
    t2 = sum([27**a * t1[a] for a in range(len(t1))])
    #Make binary
    b = [int(a) for a in str(bin(t2))[2:]]
    if len(b) < maxlen:
        return [0]*(maxlen-len(b)) + b
    else:
        return b
    
def encode_v2(word):
    #v2 encode
    #Instead of converting to base 27 and then to binary convert letter by letter to five binary digits
    encode = []
    for a in word:
        if a == " ": encode += [0.0,0.0,0.0,0.0,0.0]
        else: 
            l = [float(b) for b in str(bin(ord(a) - 96))[2:]]
            encode += [0.0]*(5-len(l)) + l
    return encode

def encode_v3(word):
    #v3 encode
    #Would using a three-digit base 3 value for each letter be an interesting experiment?
    #Round is less helpful, but it might be an interesting thing to try
    ##
    encode = []
    for a in word:
        if a == " ": encode += [0.0,0.0,0.0]
        else:
            l = [float(b)/2 for b in np.base_repr(ord(a) - 96,3)]
            encode += [0.0]*(3-len(l)) + l
    return encode

def decode(bin_list):
    #Decodes a list of binary values to a word so we can see what the generator is actually making
    #Takes as input a list of floats between 0 and 1
    #Can break if given negative values (unprocessed tanh) or if given a tensor instead of a list, for now we use try/except to address those cases
    #If we feed non-tanh values into the tanh version they'll just give answers like they were all 1s (0+1/2 = 0.5, rounds up to 1), but the tanh
    #values in the non-tanh version will give an invalid string literal error, so we run those first.
    #Internal try/except is for data types (tensor of floats, list of tensors of floats, list of floats)
    try:
        try: decimal = int("".join([str(round(a)) for a in bin_list.tolist()[0]]),2)
        except: 
            try: decimal = int("".join([str(round(a.item())) for a in bin_list]),2) 
            except: decimal = int("".join([str(round(a)) for a in bin_list]),2)
    except:
        try: decimal = int("".join([str(round((a+1)/2)) for a in bin_list.tolist()[0]]),2)
        except: 
            try: decimal = int("".join([str(round((a.item()+1)/2)) for a in bin_list]),2)
            except: decimal = int("".join([str(round((a + 1)/ 2)) for a in bin_list]),2)
    #Step 2: Convert decimal to base 27 list
    digits = []
    while decimal:
        digits.append(int(decimal % 27))
        decimal = decimal // 27
    #Step 3: Convert base 27 list back to string
    return "".join([chr(96+a) if a != 0 else " " for a in digits])

def decode_v2(bin_list):
    #Break into five-digit chunks, convert to letters
    try: l_list = bin_list.tolist()
    except:
        try: l_list = [a.item() for a in bin_list]
        except: l_list = bin_list
    
    #Round
    bin_list = [int(round(a)) for a in bin_list]
    #Chunk and word
    worddexes = [int("".join([str(a) for a in bin_list[5*c:5*c+5]]),2) for c in range(len(bin_list) // 5)]
    word = "".join([chr(96+a) if a != 0 else " " for a in worddexes])
    return word

def decode_v3(bin_list):
    #Breaks into three-digit chunks, convert to letters
    bin_list = [int(round(2*a)) for a in bin_list]
    worddexes = [int("".join([str(a) for a in bin_list[3*b:3*b+3]]),3) for b in range(len(bin_list)//3)]
    word = "".join([chr(96+a) if a != 0 else " " for a in worddexes])
    return word
    

def vectorDistance(v1,v2,weight = 1):
    #Compute error between two distributions
    return torch.sum(torch.square(v1-v2)).item() / weight

def oDist(v1,v2,weight = 1):
    #Compute distance between the distributions of two pytorch tensors: reference distribution, and generator outputs
    #Attempted replacement for lqv vectors for speeding up run time
    #Weight variable is used for the distance between the reference distribution and the random distribution
    #When used we get a number where 0 is the reference distribution, 1 is the random distribution, and our value is hopefully between 0 and 1.
    #I *think* this is going to be a reasonable proxy for letter distribution, but I might be wrong.
    return torch.sum(torch.square(torch.mean(v1,0) - torch.mean(v2,0))).item() / weight

def vectors(word_set,word_len):
    #Gets l, q, v vectors
    return [lvector(word_set),qvector(word_set),vvector(word_set,word_len)]

def lvector(word_set):
    #Compute letter frequency in word_set as vector of 27 floats
    lv = [0]*32
    for a in word_set:
        for b in a.lower():
            if b == " ": lv[0] += 1
            else: lv[ord(b) - 96] += 1
    ov = torch.tensor(lv)
    return ov/torch.sum(ov)

def qvector(word_set):
    #Compute the frequency of qu vs. q(any other letter)
    qv = [0,0]
    for a in word_set:
        if len(a) >= 2:
            for b in range(len(a) - 1):
                if a[b] == "q":
                    if a[b+1] == "u": qv[1] += 1
                    else: qv[0] += 1
            if a[-1] == "q": qv[0] += 1
    ov = torch.tensor(qv)
    if torch.sum(ov) != 0: return ov/torch.sum(ov)
    else: return 1

def newVvector(word_set,len_chop):
    #Let's see if we can get back to something that'll do v/c/s/p patterns
    pd = {}
    for a in word_set:
        if len(a) == len_chop:
            vcsp = ""
            for b in range(len_chop):
                if a[b] in "aeiouy": vcsp += "v"
                elif a[b] == " ": vcsp += "s"
                elif not a[b].isalpha(): vcsp += "p"
                else: vcsp += "c"
            if vcsp in pd: pd[vcsp] += 1
            else: pd[vcsp] = 1
    tot = sum([pd[a] for a in pd])
    return {a:pd[a] / tot for a in pd}

def dictCompare(set1,set2,weight = 1):
    #Compare two frequency dicts
    #Almost certainly slower than our old vector-compare method, will think carefully about deployment
    #Problem: We're trying to compare both dict(float:int) and dict(float:dict(float:int))
    total_dist = 0
    num_patterns = 0
    for a in set1:
        if a in set2: total_dist += (set1[a] - set2[a]) ** 2
        else: total_dist += set1[a]**2
        num_patterns += 1
    for a in set2:
        if a not in set1:
            total_dist += set2[a]**2
            num_patterns += 1
    return total_dist / weight

def vvector(word_set,len_chop):
    #Hrm. Not ideal, at the moment it just computes frequency of vowels/consonants/spaces/non-alphas
    incidence = [[0 for a in range(4)] for b in range(len_chop)]
    for a in word_set:
        if len(a) == len_chop:
            for b in range(len_chop):
                if a[b] in "aeiouy": incidence[b][0] += 1
                elif a[b] == " ": incidence[b][2] += 1
                elif not a[b].isalpha(): incidence[b][3] += 1
                else: incidence[b][1] += 1
    return torch.tensor([c for d in [[(b/sum(a)) / len_chop for b in a] for a in incidence] for c in d])

def old_vvector(word_set,len_chop):
    #Compute the frequency of various vc_ patterns
    v_patterns = ["v","c","_","p"]
    for a in range(1,len_chop):
        v_patterns = [d for e in [[b+c for b in v_patterns] for c in "vc_p"] for d in e]
    vv = {a:0 for a in v_patterns}
    for a in word_set:
        if len(a) == len_chop:
            ap = ""
            for b in a:
                if b == " ": ap += "_"
                elif b in "aeiouy": ap += "v"
                elif b in "bcdfghjklmnpqrstvwxz": ap += "c"
                else: ap += "p"
            vv[ap] += 1
    ov = torch.tensor([vv[a] for a in vv])
    return ov/torch.sum(ov)

def newVP(word_set,len_chop):
    v_patterns = [0]*(4**len_chop)
    for a in word_set:
        if len(a) == len_chop:
            wl = []
            for b in a:
                if b in "aeiouy": wl.append(0)
                elif b == " ": wl.append(2)
                elif not b.isalpha(): wl.append(3)
                else: wl.append(1)
            wn = sum([4**b * wl[b] for b in range(len(wl))])
            v_patterns[wn] += 1
    return torch.tensor([a/sum(v_patterns) for a in v_patterns])

def newLF(word_set,window,len_chop):
    w_freq = [0]*(32**window)
    for a in word_set:
        if len(a) == len_chop:
            for b in a:
                for c in range(len(a) - (window - 1)):
                    lv = sum([32**d * (ord(a[c:c+window][d]) - 96) if a[c:c+window][d] != " " else 0 for d in range(window)])
                    w_freq[lv] += 1
    return torch.tensor([a/sum(w_freq) for a in w_freq])

def lgf(in_words,lens = [1],wlen = 5):
    #Letter group frequency so we can have 1-d dicts
    groups = [{} for a in range(len(lens))]
    for a in in_words:
        for b in range(len(lens)):
            for c in range(len(a) - (lens[b] - 1)):
                lg = a[c:c+lens[b]]
                if lg in groups[b]: groups[b][lg] += 1
                else: groups[b][lg] = 1
    for a in range(len(groups)):
        gsum = sum([groups[a][b] for b in groups[a]])
        groups[a] = {b:groups[a][b]/gsum for b in groups[a]}
    return groups

def probChain(in_words,window = 1,len_cutoff = 5):
    #Computes the probability of a given letter following the last n letters
    groups = {}
    for a in in_words:
        if len(a) >= len_cutoff:
            #For each word, start from index (window - 1) and go until the second-last index
            for b in range(window-1,len(a) - 1):
                pl = a[b-(window-1):b+1]
                nl = a[b+1]
                if pl in groups:
                    if nl in groups[pl]: groups[pl][nl] += 1
                    else: groups[pl][nl] = 1
                else: groups[pl] = {nl:1}
    #We want to just get the probability of nl from groups[pl] here, we don't care about the frequency or position of pl
    for a in groups:
        tot = sum([groups[a][c] for c in groups[a]])
        groups[a] = {c:groups[a][c] / tot for c in groups[a]}
    return groups

#__READ AND PROCESS__

def extractQual(filename):
    #Get qualvs out of filename for graphing
    data = open("ArsOut/{}".format(filename),"r").read()
    #batchq = [[],[],[]]
    #epochq = [[],[],[]]
    qual_lines = [[b.split("||")[1:] for b in a.split("\n") if b.split("||")[0] == "QualVs"][0] for a in data.split("Epoch ")[1:]]
    
    batchq = [[] for a in range(len(qual_lines[0][0].split("|")))]
    epochq = [[] for a in range(len(qual_lines[0][0].split("|")))]
    
    for a in qual_lines:
        sub_a = [[float(b.split("|")[c]) for b in a] for c in range(len(a[0].split("|")))]
        #print(sub_a)
        avg_a = [sum(b) / len(b) for b in sub_a]
        for b in range(len(sub_a)):
            batchq[b] += sub_a[b]
            epochq[b].append(avg_a[b])
        
    batchq.append([sum([a[b] for a in batchq])/len(batchq) for b in range(len(batchq[0]))])
    
    epochq.append([sum([a[b] for a in epochq])/len(epochq) for b in range(len(epochq[0]))])   
    return batchq,epochq

def extractError(filename):
    #Get error out of filename for graphing
    data = open("ArsOut/{}".format(filename),"r").read()
    batche = [[],[]]
    epoche = [[],[]]
    gene_l = [[b.split("||")[1].split("|") for b in a.split("\n") if b.split("||")[0] == "GenE"][0] for a in data.split("Epoch ")[1:]]
    disce_l = [[b.split("||")[1].split("|") for b in a.split("\n") if b.split("||")[0] == "DiscE"][0] for a in data.split("Epoch ")[1:]]
    
    for a in gene_l:
        sub_a = [float(b) for b in a]
        avg_a = sum(sub_a) / len(sub_a)
        batche[0] += sub_a
        epoche[0].append(avg_a)
    
    for a in disce_l:
        sub_a = [float(b) for b in a]
        avg_a = sum(sub_a) / len(sub_a)
        batche[1] += sub_a
        epoche[1].append(avg_a)
    
    #batche.append([batche[0][a] + batche[1][a] for a in range(len(batche[1]))])
    #epoche.append([epoche[0][a] + epoche[1][a] for a in range(len(epoche[0]))])
    
    return batche,epoche

def singleRunDisplay(in_file):
    batche,epoche = extractError(in_file)
    batchq,epochq = extractQual(in_file)
    indexnames = [["BE Gen","BE Disc","BE Cum"],["EE Gen","EE Disc","EE Cum"],["BQ LD","BQ QM","BQ VD","BQ Avg"],["EQ LD","EQ QM","EQ VD","EQ Avg"]]
    modedex = {"be":[batche,0],"ee":[epoche,1],"bq":[batchq,2],"eq":[epochq,3]}
    window = 1
    horizontal = False
    while True:
        mode = input("What would you like to see about this run? (b = batch, e = epoch, q = qual, r = err, 0-3 = list index, w = window, exit = quit)")
        if mode == "exit": break
        elif mode[0] == "w":
            try:
                window = int(mode[1:])
                print("Window changed to {}".format(window))
            except: print("I'm sorry, I didn't catch that?")
        elif mode[0] == "h":
            try:
                horizontal = float(mode[1:])
                print("Horizontal print set to {}".format(horizontal))
            except:
                horizontal = False
                print("Horizontal turned off")
        elif "-" in mode:
            ms = mode.split("-")
            ms0a = ms[0][:2]
            ms0b = int(ms[0][2])
            ms1a = ms[1][:2]
            ms1b = int(ms[1][2])
            temp_list = [modedex[ms0a][0][ms0b][a] - modedex[ms1a][0][ms1b][a] for a in range(len(modedex[ms0a][0][ms0b]))]
            linename = indexnames[modedex[ms0a][1]][ms0b] + "-" + indexnames[modedex[ms1a][1]][ms1b]
            plt.plot(trendLine(temp_list,window),label = linename)
            plt.legend()
            plt.show()
        elif mode[:2] in modedex:
            cm = mode[:2]
            indexval = modedex[cm][1]
            refl = modedex[cm][0]
            widest = 0
            for a in range(len(indexnames[indexval])):
                if str(a) in mode:
                    plt.plot(trendLine(refl[a],window),label = indexnames[indexval][a])
                    widest = max(len(refl[a]),widest)
            if widest == 0: print("Nothing to display? Try entering some numbers with that prefix next time.")
            else:
                print("Close the graph to return to the menu.")
                if horizontal: plt.plot([horizontal]*widest)
                plt.legend()
                plt.show()
        else:
            print("I'm sorry, I didn't catch that?")
            

def trendLine(in_list,trend_window):
    out_list = []
    for a in range(len(in_list)):
        if a < trend_window:
            out_list.append(sum(in_list[:a+1]) / (a+1))
        else:
            out_list.append(sum(in_list[a-trend_window:a+1]) / (trend_window + 1))
    return out_list
    
def quickData(data_paths,batch_size = 200, gibberish_path = "gibberish_5.txt",dflag = False,dw = 1):
    data = unifiedWordset(data_paths,dflag = dflag,dw = dw)
    dl = DataLoader(data,batch_size,shuffle = True)
    
    #gibberish = open(gibberish_path,"r").read().split("|")
    #gibberish = [a for a in json.load(open(gibberish_path,"r"))]
    gibberish = unifiedWordset([gibberish_path],dflag = dflag,dw = dw)
    if dflag:
        g_dist = [dictCompare(gibberish.distros[a],data.distros[a]) for a in range(len(gibberish.distros))]
    else:
        g_dist = [vectorDistance(gibberish.distros[a],data.distros[a]) for a in range(len(gibberish.distros))]
    
    return dl,data.datalen,data.distros,g_dist

def batchGroups(prefix,batches,loop_indexes = []):
    #When running groups of parameter searches, this lets us look at the output by parameter
    #If given loop_indexes will group that way
    #Blergh. Explanations.
    #General theory: prefix + batches gets us files we can run extractQual on
    #extractQual gives us per-batch and per-epoch quality measures
    #This function takes that and gets the average quality from all three metrics over batches as one list
    #Then it groups our outputs if given loop_indexes
    #Anything in, e.g., mth loop n  will be given by a list of lists in indexed_vals[m][n]
    out_vals = []
    indexed_vals = [{} for a in range(len(loop_indexes))]
    for a in range(batches):
        batchq,epochq = extractQual("{} {}.txt".format(prefix,a))
        batche,epoche = extractError("{} {}.txt".format(prefix,a))
        total_vals = batchq + epochq + batche + epoche #NOTE: This will be a 1d list of lqva(b),lqva(e),gdt(b),gdt(e)
        out_vals.append(total_vals)
        if loop_indexes:
            
            batch_indexes = batchIndex(loop_indexes)[a]
            for b in range(len(indexed_vals)):
                if batch_indexes[b] in indexed_vals[b]:
                    indexed_vals[b][batch_indexes[b]].append(total_vals)
                else:
                    indexed_vals[b][batch_indexes[b]] = [total_vals]
                
    return out_vals,indexed_vals

def batchIndex(li):
    #This reverse-engineers the values from a nested for loop from an integer index, knowing what the for loop's limits were
    #Example:
    '''
    x = 0
    for a in range(2): for b in range(2): for c in range(2): x++
    Produces:
    0: 0,0,0
    1: 0,0,1
    2: 0,1,0
    3: 0,1,1
    4: 1,0,0
    5: 1,0,1
    6: 1,1,0
    7: 1,1,1
    
    This gives us a loop_indexes of [2,2,2] and qi of [[4,2],[2,2],[1,2]]
    We then reverse-engineer via batch_indexes to:
    0//4%2 = 0, 0//2%2 = 0, 0//1%2 = 0
    1//4%2 = 0, 1//2%2 = 0, 1//1%2 = 1
    2//4%2 = 0, 2//2%2 = 1, 2//1%2 = 0
    3//4%2 = 0, 3//2%2 = 1, 3//1%2 = 1
    4//4%2 = 1, 4//2%2 = 0, 4//1%2 = 0
    5//4%2 = 1, 5//2%2 = 0, 5//1%2 = 1
    6//4%2 = 1, 6//2%2 = 1, 6//1%2 = 0
    7//4%2 = 1, 7//2%2 = 1, 7//1%2 = 1
    
    Overengineered, perhaps, when we could just save the a,b,c index and read it back, but it works.
    '''    
    qi = [[int(np.prod(li[b+1:])),li[b]] for b in range(len(li))]
    return [[(a//qi[b][0]) % qi[b][1] for b in range(len(qi))] for a in range(np.prod(li))]

def listAvg(lists):
    return [sum([a[b] for a in lists]) / len(lists) for b in range(len(lists[0]))]

def overfitUtility(run_dump,ref_dict):
    dump_data = open(run_dump,"r").read().splitlines()
    word_lines = [a for a in dump_data if a.split("||")[0] == "Words"]
    practical_split = [[b for b in a.split("|") if len(b) == 5] for a in word_lines]
    real_dict = json.load(open(ref_dict,"r"))
    ofdex = []
    for a in practical_split:
        in_val = 0
        total = 0
        for b in a:
            if b in real_dict: in_val += 1
            total += 1
        ofdex.append(in_val / total)
        
    return ofdex

def numPerf(in_list):
    #Given a list of average values over time for a number of values can we compute which has the lowest total, and how long they spent in that order?
    order = [[a,sum(in_list[a]) / len(in_list[a])] for a in range(len(in_list))]
    order.sort(key = lambda a:a[1])
    od = [a[0] for a in order]
    
    half_order = [[a,sum(in_list[a][(len(in_list[a]) // 2):]) / (len(in_list[a]) // 2)] for a in range(len(in_list))]
    half_order.sort(key = lambda a:a[1])
    hod = [a[0] for a in half_order]
    
    opf = 0
    oph = 0
    
    for a in range(len(in_list[0])):
        inst_ord = [[b,in_list[b][a]] for b in range(len(in_list))]
        inst_ord.sort(key = lambda a:a[1])
        io = [b[0] for b in inst_ord]
        if io == od: opf += 1
        if io == hod: oph += 1
    
    opf = opf / len(in_list[0])
    oph = oph / (len(in_list[0]) // 2)
        
    
    return order,half_order,oph,oph
    


#__MAIN__

def eval_main():
    batchName = "022724 Ratio"
    loopSizes = [4,4]
    paramNames = ["Disc_R","Gen_R"]
    totalSize = 1
    for a in loopSizes:
        totalSize = totalSize * a
    
    
    db,di = batchGroups(batchName,totalSize,loopSizes)
    #Current run: g 1 2 3 4, d 1 2 3 4, [100, 100:200, 200:100, 200:200]
    
    num_perf = [[]] + [[] for a in range(len(loopSizes))]
    
    #Show parameter groups
    for a in range(len(loopSizes)):
        for b in di[a]:
            avg = listAvg([c[3] for c in di[a][b]])
            num_perf[a].append(avg)
            plt.plot(trendLine(avg,5),label = "{} {}".format(paramNames[a],b))
        plt.legend()
        print("Now showing graph of {}".format(paramNames[a]))
        plt.show()
        plt.close()
        print("Numerical performance at this step:")
        od,hod,opf,oph = numPerf(num_perf[a])
        print("Full run order: {}".format(", ".join(["{}: {}".format(b[0],b[1]) for b in od])))
        print("Full order is accurate {}% of the time".format(opf*100))
        print("Last half run order: {}".format(", ".join(["{}: {}".format(b[0],b[1]) for b in hod])))
        print("Last half order is accurate {}% of the time".format(oph*100))        
        input("Press enter to continue:")
        
    for a in range(totalSize):
        num_perf[-1].append(db[a][3])
        a_data = trendLine(db[a][3],100)
        a_labels = batchIndex(loopSizes)[a]
        label = ".".join(["{}_{}".format(paramNames[b],a_labels[b]) for b in range(len(paramNames))])
        plt.plot(a_data,label = label)
    
    print("Full graph:")
    
    plt.legend()
    plt.show()
    
    od,hod,opf,oph = numPerf(num_perf[-1])
    print("Full run order: {}".format(", ".join(["{}: {}".format(b[0],b[1]) for b in od])))
    print("Full order is accurate {}% of the time".format(opf*100))
    print("Last half run order: {}".format(", ".join(["{}: {}".format(b[0],b[1]) for b in hod])))
    print("Last half order is accurate {}% of the time".format(oph*100))     

    
def parameterSearch():
    #Main function to iterate over several sets of parameters and look for the best results over a short run
    #List of parameters that could be tried:
    #Model size: datalen input/datalen or 1 output (gen/disc) is fixed, but how many and how big are the hidden layers?
    #Layer parameters: hidden + out, regularize, dropout, activation
    #Model optimizer: type, parameters
    #Loss type. WGAN not coded at present.
    #Problem: Exhaustive parameter search might take a long-ass time, what works best with what?
    #Upside: While running we could get network.run_list and use to evaluate instead of needing to go back and read the dump files
    #GANModel.runQual() is the call
    #For the moment let's try varying one thing and see what happens
    #Earlier tests appear to stabilize on error within a few epochs, I think
    
    #Initial test: 1 hidden layer gen/disc, varying generator hidden layer size
    #GENERAL TRENDS FROM TEST 1 (gen hidden layer 100-200 increment 25): Smaller slightly better, minimal variance in 75 runs
    #Test 2: Disc hidden layer 100-200 increment 25): Smaller still slightly better, minimal variance in 100 runs. Not *much* better, changes by ~1% per +25, but still better.
    #Test 3: Two-layer gen, 100-200/150, one-layer disc. Stronger performance towards the middle. Does this indicate stronger performance from
    #150 specifically, or from more uniform values? Let's try the same run, only with a 2nd layer of 200.
    #Test 4: With a 2nd layer of 200 the 175-200 outperformed the smaller 1st layers, so it looks like similar-size hidden layers works better for gen
    #Does this hold for disc? For magnifying the effect we'll try with 2nd layer 100 and 200
    #Test 5: No conclusive data for #5. Hrm. We'll see. Next test hypothesis: depth. Given layers of size 100 for runspeed 1, 2, 3 hidden layers for both?
    #Test 6: Shallower generator/deeper discriminator did better. Not sure what to conclude from that.
    #Let's try varying depth for size 150
    #Test 7: At layer size 150 0,1,2 (shallow generator) still did really well, but 3 (2,1) actually did pretty good as well. Hrm. Try for 200?
    #Test 8: At 200 0,1,2 remained solid and reliable, but 7 (3,2) performed incredibly well towards the end. Hrm. Try 300?
    #Test 9: At 300 got a couple of NAN errors, and the averages spread themselves out a bit, 2 (1,3) at 646, 8 (3,3) at 800.
    #Does this effect carry forward with larger discriminators? Return to 100 for speed.
    #Test 10: Back to 100 but with gen 1,2,3 and disc 3,4,5. There may be diminishing returns, 1,3 was the best performer at this stage,
    #though 2,4 did very well, too, so perhaps "two more discriminator layers" rather than max disc/min gen.
    #Theory: Do we have other levers to pull that might address one going too much faster than the other? Ratio?
    #Let's try 1,2,3 at 100 with a 1:1 ratio, see what happens.
    #Test 11: With the 1:1 ratio the best performer was 1,1. Hrm. Fascinating. Let's increase width and see what happens.
    #Test 12: 1:1 ratio and 200, 1:1 and 1:2 remain the best overall, but by the end of the run 2,2 was quite strong
    #Let's consider deeper nets (3,5) with higher widths
    #Test 13: Some kind of number overflow problem. No great performers and no distinctions, really. Hard to pick this out as better.
    #Let's try depths of 1-3 with size 500 and M12 for the overnight run, see if that works.
    #Test 14: Moderately strong, 1,1 was the best performer overall, but 3,2 did better in the 400+ timescale somehow
    #Relative depth may be a dead end, given how similar lots of these get, and how the numbers are never that strong.
    #Test 15: Literature suggests one-sided label smoothing (real label = 0.9), we were doing two-sided label smoothing (real 0.9, fake 0.1), let's try one-sided
    #With label smoothing and 500-size we're looking at 1,3 as the best performer, but wasn't that much better than 2,1
    #Considering where to go next: Longer runs, change other variables? Slowed learning rate from 0.002 to 0.001/0.003, and added a 0.0001 L2 reg
    #Test 16: 2,1 2,2 and 3,2 performed best, but the margins are quite low (peak: 267/300). Let's try iterating over L2 reg with 2,2 networks and see what happens.
    #L2 reg terms: 0, 0.001, 0.002 for a and b
    #Test 17: No particular pattern emerging, some outliers (1,0 particularly strong, 0,1 particularly weak). Should probably try bumping the range up a ways.
    #Trying 0,0.003,0.006, see what that changes.
    #Test 18: More regularization doesn't necessarily help, 0,0 was quite good here and 2,2 bad. Peak 262/300
    #Next: Adamax betas 0.5/0.7/0.9.
    #Test 19: Adamax betas don't seem to matter a whole lot, low variance, peak 274/300 at 2,2.
    #Next: What else is a variable we could fiddle with? LR! Also trying relatavistic loss. LR .001, .002, .003
    #Test 20: Not super conclusive RE learning rates, peak 264/300, so RL *works*, just doesn't present a huge improvement. #TYPO: a controlled both LR
    #With the typo that gives us three trials of learning rates and suggests more is better, but let's try this again properly.
    #Test 21: Peak 270/300. Not conclusive. Let's try weighting one more heavily, gen 0.002/4/6, disc .001/2/3
    #Going back to nets with 100 internal width for speed
    #Test 22: .002/4/6 gen, .001/2/3 disc, no conclusions from this run. Let's invert, see if that makes any difference when the discriminator can be much better than the generator.
    #Test 23: .001/2/3 gen, .002/4/6 disc, no clear outliers, 0,0 did very well for a while but then spiked up a ways. 272/300 peak, 1,0, which is equal learning rates. Hrm.
    #Also the question of ratio now. And possibly multiple runs on the same batch. Hrm. Let's try equivalent with greater magnitude (2,4,6), 2:2 ratio.
    #Thoughts to date: Relativistic loss isn't really helping, not a lot of data suggesting any particular layout is better than any other layout, really.
    #Probably a good idea to try doing a setup to run a range of architectures with a fixed seed.
    #Things that might have some effect that we can vary: depth, breadth, lr, rs, betas
    #Test 24: .2/4/6 lrs for both, 2:2 ratio. Peak 269/300 at .2,.4, but very little variation overall.
    #Test 25: No conclusions of use. Mistakenly put the random seed outside the loop, and results were not controlled at all.
    #Test 26: Resetting for a more controlled test. Not varying lr this time (pegged to 0.002 each), only network sizes, whether relativistic loss used.
    #Sizes are 30 or 60, hoping that'll make it run faster through 36 runs. If not that's...108 minutes. Ish. Well, it'll be done by...12:10!
    #Problem: Metrics. Precalculating saves us on storing large sets of words, but slows down training loop. Can we do a distribution distance metric just with
    #a torch comparison without having to compute word properties?
    #Stopped run 27 short, due to duplication would have been a nightmare to work out what actually meant anything, which really borks test 25 further.
    #Conclusions from run 27: Shitty. Bigger is *slightly* worse, but everything is so close and so bad no meaningful conclusions to draw.
    #Run 28: Trying just 100,100 and 200,200.
    #Looking at graphs: Rel looks slightly better than trad and narrower networks slightly better than wider here.
    #Major adjustment: v2 encoding (5-digit binary per letter, 25 binary values for a 5-letter words) seems to work better than either v3 encoding
    #(3-digit trinary per letter) or v1 encoding (base 27-base 10-base 2, 24 binary values for 5-letter words)
    #General comparison: Should be sufficient to reach equilibrium, the equilibrium of a v2 encoding net is way better
    #Run 29: 100,100 vs 200,200 and rel v. not with new data. On average rel slightly better, bigger slightly better, but not convincingly.
    #Hypothesis: WGAN version?
    #Hrm. WGAN-GP is implemented now, but will it work?
    #Run 30: WGAN-GP characterized by poor performance and discontinuity. Let's not.
    
    #For loops for varying depth and breadth at the same time may not be easily doable
    #Trying to produce: [1],[2],[1,1],[1,2],[2,1],[2,2] from the same for loop?
    #Run 31: Some discontinuity, have adjusted to avoid in future. General conclusions: More is better, early value slightly more so than later value.
    #Good news: Code will generate samples of fixed length well
    #Bad news: Code won't generate samples of arbitrary length, even given padded inputs
    #Short fix is just to parallel-train several models, are we happy with that?
    #Let's try it and see.
    #Facepalm moment: Nope, I was stripping space-padded words out of the input dataset, it actually does work on varying-length inputs
    #Now that that works let's see if some parameter searching does things
    #Testing the 5_7 data for different sizes
    #Run 32: Bigger initial seemed to work better for generator, but smaller initial seemed to work better for discriminator, which seems...odd.
    #Let's try just running the big/small ones without the even ones and see what happens. Also adding a c term to give us duplicate runs to average.
    #Also going to 100,300 instead of 100,200 to see if that magnifies the effect until we can see it.
    #...I feel extremely silly, we're doing fixed seed, the c term doesn't do anything. Gr.
    #Run 33: Effect is minute, but bigger initial seemed to work slightly better for generator, and bigger final slightly better for discriminator.
    #No idea what this means.
    #...Now I kind of want to try adding dropout and seeing what happens.
    #Overnight single run with some dropout: Letterdistro and vdistro seem correlated. Let's try modifying stuff to track only ldist and see if that lets us run bigger data.
    #Single test with ldistros only: Doesn't seem to be running any faster? Hrm. We'll try running the bigger set at some point but is anything still calling vvector?
    #We're not actually calling vvector anywhere. So why is this still slowing down?
    #Not sure what changed, normal version is slow now too (1/5 sec instead of 2)
    #Anywaystimes: Let's try a good ol'-fashioned parameter search! Looking to compare d0,d2,d4 for g/dis
    #Possible false alarm, parameter search is running at 2/5s or thereabouts, might just have been depth 3 throwing the numbers out
    #Run 34: Fascinating, 2/0, 2/1, and 2/2 are all strong and reliable contenders, and 0/0 did badly, but 0,1 was unexpectedly strong.
    #Run 35: No 0/1 outlier this time, just 2/0, 2/1, and 2/2 clustered at the bottom, impressive performance given 90% dropout towards the end.
    #Run 36: Doesn't repeat 34-35, 2/x were the worst overall, with no strong separation at the bottom (0,0 and 1,2 seemed strong)
    #Let's do another trial of the same parameters.
    #Run 37: Repeats run 36, 2/x poor, 1/2 had strong peak performance.
    #In light of the poor performance of 60% garn dropout and strong 30/10 let's try resetting back to same scale on a,b
    #Run 38: Back to 0/.2/.4 not really conclusive. Let's make the layers uniform, remove any unnecessary variables that way, and then see if adding dropout to more layers does much.
    #Run 39: Undermined by really bad mode collapse. Not sure why, any dropout in the generator seems to have made an utter mess of the output
    #Run 40: Nothing really to set much of this apart, annoyingly. Let's peg to one layer at 0.4 and then try different activation functions, comparing relu/elu/leaky
    #Run 41: Leaky is a bit better than regular, but elu is better and much smoother? Hrm.
    #Dramatically so. Let's mess with the dropout as well for the overnight run.
    #Run 42: Big overnight run comparing activation functions and dropout rates. It seems with this data that more dropout is better most of the time,
    #and elu is the best activation function.
    #Bumping back up to two hidden layers with dropout on both and elu activations for the next test run, also bumping to m16 to help stem mode collapse
    #like back on run 39.
    #Run 43: Runs going for around 7m, rather than 5m for earlier configurations. Worth keeping an eye on. Ideally for parameter search we want shorter runs.
    #Inconclusive RE dropout, seems to not work super well. Earlier tests have shown two dropout in generator is worse, maybe change to one?
    #Trying with one gen dropout/two disc dropout
    #Run 44: Nothing able to distinguish itself, really. Let's try going back down in size for faster runs, and then trying to add d0 back into the mix.
    #Run 45: More dropout seems better, but not linearly better. We might need other metrics beyond looking at the graph.
    #With numerical data more is sometimes better. I don't know if we're going to get a good, consistent answer to the dropout question the way we did with the activation question.
    #Let's try and rescale again, test for 2,4,6,8 instead of 0,2,4,6.
    #Run 46: Still not really giving us much. What other parameters can we mess around with?
    #Last half top three: .8/.6 .614, .8/.2 .623, .6/.6 .649
    #Run 47: Smaller size seemed to be better on this run, which doesn't line up super well with earlier estimates. Hrm.
    #Last half top three: 50g/100d .545, 100g/150d .551, 100g/100d .564
    #From runs 46 and 47, then, 50g/100d with .5/.5 dropout was significantly better than any 200-size run.
    #Hypothesis: If we use sizes 50/100 on a dropout test we'll see better similar top values and better peak performance.
    #50/100 with .5/.6/.7/.8:
    #Run 48: Lower gen dropout is better, disc dropout doesn't make much difference.
    #Last half top three: .5/.8 .534, .5/.5 .545, .5/.6 .556,
    #The .545 for .5/.5 gives me increased confidence the reliability of the fixed seed is working.
    #What kind of stuff does that .534 last average represent? Barqi. Rqmrmb. Ajeene. Diraw. Famresd. Chilaa. Parund. S mni. Ramafel. Mmgvmee.
    #Not bad at all! No non-letter characters, 7/10 pronouncable!
    #With disc drop making little difference let's try a long run with a low gen drop.
    #Run of 300 on the big six-layer gave me a last-100 average of 0.550 at the end
    #Hrm. Let's try another depth setup.
    #Run 49: Depth 2/3/4/5, lower G depth performed pretty consistently better at width 75/75, peak last-half performance was .651 at depth 2/2
    #Let's try two variations on width for two variations on depth, see what happens.
    #Run 50: Depth 3/4, width 50/100
    #Lower depth, higher width seems generally preferable
    #Best last-half: 1 (d3/d3/w0/w1) .707, 7 (d3/d4/w1/w1) .739, 3 (d3/d3/w1/w1) .767
    #Let's spread depth/width further apart and see if this still holds
    #Let's also bump batch size up a bit and see if that helps.
    #Run 51: Depth 2/4, width 100/200
    #Confirm improved performance of wider/shallower on average, but peak individual performance was .581 from 5 (d2/d4/w100/w200).
    #Let's do a long wide shallow individual run
    #Overnight single run: Very wide (1700/1900), shallow (2,2), some dropout (d3 g/d5 d), average reliably under 0.4 by the end.
    #Problem: Still producing some special characters/unpronouncable gibberish
    #Considering a new metric.
    #New metrics (letter distro/letter group distro, actual vcsp patterns) do work, but may be a bit slow.
    #Long-run setup for running for a while between evals is set up and moving, with that new eval may be no slower than old, which helps.
    #Run 52: Depth search: Quite poor. 2-8 hidden layers showed poor performance for deeper networks.
    #Returning to slightly wider, shallower networks: let's take a poke at lr and see if that does much
    #Run 53: Betas .5, .6, .7. Not hugely separable. Lower seems better early on, but explodes a bit further out. Let's do a short run of 0.3 and 0.4, see what happens.
    #Run 54: Betas .3, .4. Lower slightly better, but hard to separate.
    #Run 55: LR .002, .003. Higher better.
    #Run 56: LR .004, .005. Lower better.
    #At this point prepping for long overnight run. Let's try .002-.005, see if the sweet spot remains at the middle over a larger number of runs.
    #Run 57: .002-.005 for 500 runs. .004 seems to have been the standout, but the best performer was .004/.003 with a last half performance of .454.
    #If we have time today let's try a long, wide, shallow run with the .5 beta and .004 lr
    #It worked pretty well, got down below .33 by the end. Let's do a detailed version so we can run the graph.
    #Status at the moment: Running with moderately wide (500/750) sets with 2 hidden layers, lr 0.4 and beta 0.5, dropout .2 in one layer on the generator and
    #.4 in one layer on the discriminator, seems to be able to produce all right outputs from 1,000 words in ~60 epochs and from 500 words in ~100 epochs
    #Next things to do: Try on more datasets. Pokemon v. English dataset comparison experiment is running, but there are other experiments to do,
    #e.g. English words of German origin vs. French origin, if we can figure out a usable breakdown. Accents remain a limiting factor on bringing in non-English words.
    #Considering German, cyrillic to English, possibly see if the net can capture the Japanese syllabary.
    #Pokemon v. English report: Poor. Guesses by observers couldn't separate the datasets.
    #Run 58: Ratio test, 1-4,1-4. Major problem here is length of run, 6-7min starting drifted to 15-17m by the end.
    #Run 59: Mildly surprising result, I've been running with ratios 1:2, but 2:1 seems to be better here.
    #Hypothesis: What would happen to the human guesses if we got better results?
    #Major revision to the program produced a set of NN structures different enough that I couldn't figure out how to get them to work.
    #Not sure if I give up and return to this one, or keep trying to make that one work.
    
    
    
    data,data_l,data_v,g_dist = quickData(["ArsData/5_7_la.json"],gibberish_path = "ArsData/5_7_l_g.json")
    #lgl = [[[" "," ","e"],[" ","d2","e"]] + [[" "," ","e"]] * a + [[" "," ","t"]] for a in range(4)]
    #ldl = [[["m12"," ","e"]] + [[" ","d4","e"]] * (a+1) + [[" "," "," "]] for a in range(4)]
    #hidgl = [[data_l] + [75] * (a+2) + [data_l] for a in range(4)]
    #hiddl = [[data_l] + [75] * (a+2) + [1] for a in range(4)]

    
    rindex = 0
    for a in range(4):
        for b in range(4):
            lg = [[" "," ","e"],[" ","d2","e"],[" "," ","t"]]
            #lg = [[" "," ","e"],[" ","d2","e"]] + [[" "," ","e"]] * ((2*a)) + [[" "," ","t"]]
            ld = [["m12"," ","e"],[" ","d4","e"],[" "," "," "]]
            #ld = [["m12"," ","e"]] + [[" ","d4","e"]] * ((2*b)+1) + [[" "," "," "]]
            torch.manual_seed(0)
            hidg = [data_l,150,150,data_l]
            #hidg = [data_l] + [100] * ((2*a)+2) + [data_l]
            hidd = [data_l,250,250,1]
            #hidd = [data_l] + [150] * ((2*b)+2) + [1]
            gen_params = [hidg,lg,"Adamax",[0.004,(0.5,0.99),0.001]]
            disc_params = [hidd,ld,"Adamax",[0.004,(0.5,0.99),0.001]]
            test_net = GANModel(gen_params,disc_params,"BCEL",[data_v,g_dist])
            test_net.TrainBasic(data,100,ratio = (a+1,b+1),run_name = "022724 Ratio {}".format(rindex),cost_type = "rs",wl = data_l//5)
            #print(hidg,hidd)
            rindex += 1


def run_main():
    print("Starting main")
    
    vflag = False
    dflag = False
    dw = 2
    #Parameters: sizes, parameters, optimizer type, optimizer parameters
    
    #data_2 = unifiedWordset(["ArsData/5_5_comm_{}.json".format(a) for a in "abcdef"])
    #data_1 = unifiedWordset(["ArsData/5_5_full_{}.json".format(a) for a in "abcdef"])
    #fake_d = unifiedWordset(["gibberish_5.json"])
    
    #data = unifiedWordset(["ArsData/3_12_cl.json"])
    #data = unifiedWordset(["ArsData/3_12_pokel.json"])
    #data = unifiedWordset(["ArsData/5_7_pokel.json"],voflag = vflag,dflag = dflag,dw = dw)
    data = unifiedWordset(["ArsData/5_7_lb.json"],voflag = vflag,dflag = dflag,dw = dw)
    #data = unifiedWordset(["ArsData/5_7_ra.json"])
    #data = unifiedWordset(["ArsData/5_7_ca.json"])
    #data = unifiedWordset(["ArsData/5_7_aa.json"])
    
    #gibberish = unifiedWordset(["ArsData/3_12_g.json"])
    gibberish = unifiedWordset(["ArsData/5_7_l_g.json"],voflag = vflag,dflag = dflag,dw = dw)
    #gibberish = unifiedWordset(["ArsData/5_7_r_g.json"])
    #gibberish = unifiedWordset(["ArsData/5_7_c_g.json"])
    #gibberish = unifiedWordset(["ArsData/5_7_a_g.json"])
    
    #data_2 = unifiedWordset(["commonl5.json"])
    dl = DataLoader(data,200,shuffle = True)
    wlg = data.datalen//5
    if dflag:
        g_dist = [dictCompare(gibberish.distros[a],data.distros[a]) for a in range(len(data.distros))]
    else:
        g_dist = [vectorDistance(gibberish.distros[a],data.distros[a]) for a in range(len(data.distros))]
    print(data.data.size())
    
    print("Vectors loaded")
    
    #Problem: Can we use wordsets for oDist?
    
    '''
    t1 = oDist(data.data,fake_d.data)
    vdist = [data.data,t1]'''
    
    #SET UP MODEL
    gen_params = [
        [data.datalen,750,750,data.datalen],
        [[" "," ","e"],[" ","d3","e"],[" "," ","t"]],
        "Adamax",
        [0.005, (0.6,0.99), 0]
        ]
    
    disc_params = [
        [data.datalen,1500,1500,1],
        [["m12"," ","e"],[" ","d5","e"],[" "," "," "]],
        "Adamax",
        [0.005, (0.6,0.99), 0]
        ]
    
    testNet = GANModel(gen_params,disc_params,"BCEL",rdistros = [data.distros,g_dist])
    testNet.TrainBasic(dl,100,ratio = (3,1),run_name = "030724 t1",cost_type = "rs",wl = wlg,dw = dw)
    #testNet.FastRun(dl,500,run_name = "022224 Poketest 2",cost_type = "rs",wl = wlg,dw = dw)


#parameterSearch()
#eval_main()
#run_main()

#singleRunDisplay("022724 Ratio Test.txt")