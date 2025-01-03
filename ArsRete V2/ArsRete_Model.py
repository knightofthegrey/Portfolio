#ArsRete_Model take 2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchgan.layers import MinibatchDiscrimination1d
from time import time
import json
from ArsRete_Datasets import Decode,Compare
from ArsRete_Postprocess import stdDev

#SUBSTANTIVE PART

def GanFromFile(prefix):
    tg = open("ARModels/{} Meta.txt".format(prefix),"r").read().splitlines()[0]
    tGAN = GanFromStr(tg)
    tGAN.loadGAN(prefix)
    return tGAN

def GanFromStr(in_str):
    #Makes a GANModel object using the callstr of a saved one
    temp = in_str.split("|||")
    gsz = [int(a) for a in temp[0].split("|")]
    glp = [a.split("|") for a in temp[1].split("||")]
    got = temp[2]
    if len(temp[3].split("|")) == 2: gop = [float(a) for a in temp[3].split("|")]
    else: gop = [float(temp[3].split("|")[0]),(float(temp[3].split("|")[1]),float(temp[3].split("|")[2])),float(temp[3].split("|")[3])]
    
    dsz = [int(a) for a in temp[4].split("|")]
    dlp = [a.split("|") for a in temp[5].split("||")]
    dot = temp[6]
    if len(temp[7].split("|")) == 2: dop = [float(a) for a in temp[7].split("|")]
    else: dop = [float(temp[7].split("|")[0]),(float(temp[7].split("|")[1]),float(temp[7].split("|")[2])),float(temp[7].split("|")[3])]
    
    cpr = temp[8].split("|")
    
    return GANModel([gsz,glp,got,gop],[dsz,dlp,dot,dop],cpr)

class GANModel():
    #Wrapper class containing generator, discriminator, cost, and optimizer
    
    #COMPARING MULTIPLE VERSIONS OF THE CLASS
    
    def __init__(self,g_params,d_params,cost_params):
        #When initializing: give parameters for generator, discriminator, and cost
        #This is sufficiently complex that I'm hesitant to do defaults for the moment
        
        #Metadata: String describing the network to write to log files
        
        g_meta = "\n".join(["Gen:","|".join([str(a) for a in g_params[0]])," || ".join(["|".join(a) for a in g_params[1]]),"Opt: " + g_params[2] + " " + str(g_params[3])])
        d_meta = "\n".join(["Disc:","|".join([str(a) for a in d_params[0]])," || ".join(["|".join(a) for a in d_params[1]]),"Opt: " + d_params[2] + " " + str(d_params[3])])
        self.metadata = "{}\n\n{}\n\n{}".format(" ".join(cost_params),g_meta,d_meta)
        
        #Call data: Used to reconstruct a GANModel when loading from file
        
        gsz = "|".join([str(a) for a in g_params[0]])
        glp = "||".join(["|".join(a) for a in g_params[1]])
        got = g_params[2]
        try: gop = "|".join([str(g_params[3][0]),str(g_params[3][1][0]),str(g_params[3][1][1]),str(g_params[3][2])])
        except: gop = "|".join([str(a) for a in g_params[3]])
        
        dsz = "|".join([str(a) for a in d_params[0]])
        dlp = "||".join(["|".join(a) for a in d_params[1]])
        dot = d_params[2]
        try: dop = "|".join([str(d_params[3][0]),str(d_params[3][1][0]),str(d_params[3][1][1]),str(d_params[3][2])])
        except: dop = "|".join([str(a) for a in d_params[3]])
        
        cpr = "|".join(cost_params)
        
        self.callstr = "|||".join([gsz,glp,got,gop,dsz,dlp,dot,dop,cpr])
        
        #Next: Define discriminator and generator
        #These are essentilly the same between versions.
        
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
        self.o_size = d_params[0][-1]
        
        self.wt = ""
        self.gp = ""
        
        #Next: Cost function.
        if cost_params[0] == "WGAN":
            self.cost = ""
            if "GP" in cost_params[1]:
                self.wt = "G"
                self.gp = float(cost_params[1][2:])
            elif "WC" in cost_params[1]:
                self.wt = "C"
                '''
            elif "SN" in cost_params[1]:
                self.wt = "S"''' #WGAN-SN not implemented
        
        elif cost_params[0] == "BCE": self.cost = nn.BCELoss()
        elif cost_params[0] == "BCEL": self.cost = nn.BCEWithLogitsLoss()
        elif cost_params[0] == "MSE": self.cost = nn.MSELoss()
        elif cost_params[0] == "KLD": self.cost = nn.KLDivLoss()
        elif cost_params[0] == "CEL": self.cost = nn.CrossEntropyLoss()
        
        if cost_params[1] == "rs": self.rel = True
        else: self.rel = False
        
        #Skipping over the class-level distros and run/eval lists, we'll handle those in the train function
    
    #The train functions are probably where the error happens
    
        
    def Train(self,in_data,epochs,comp_sets,ratio = (1,1),run_name = "rundump",state_name = "",eval_mode = False):
        #Takes as input lots of things and trains the network
        
        run_list = []
        last = time()
        last_report = time()
        
        for a in range(epochs):
            #If evaluating, we're storing err_g, err_d, acc, words, lc, m1c, m2c
            #Otherwise just err_g,err_d,acc,words
            #Update: More?
            if eval_mode: run_list.append([[],[],[],[],[],[],[],[]])
            else: run_list.append([[],[],[],[],[]])
            
            for b,data in enumerate(in_data):
                #For each batch:
                inputs,labels = data
                batch_err = [0,0]
                for c in range(ratio[0]):
                    batch_err[0] += self.DiscTrain(inputs)
                for c in range(ratio[1]):
                    batch_err[1] += self.GenTrain(inputs)
                
                
                self.gen.eval()
                
                #Add average batch error to run_list
                if ratio[0]: run_list[-1][0].append(batch_err[0] / ratio[0])
                else: run_list[-1][0].append(0)
                
                if ratio[1]: run_list[-1][1].append(batch_err[1] / ratio[1])
                else: run_list[-1][1].append(0)
                
                #Add batch accuracy to run_list
                run_list[-1][2].append(self.Acc(inputs))
                
                #Add words to run_list
                gen_words = self.Generate(200,False)
                run_list[-1][3].append(gen_words)
                
                #If eval_mode, add eval numbers to run_list
                if eval_mode:
                    temp = Compare(gen_words,comp_sets[0],comp_sets[1])
                    for c in range(3):
                        run_list[-1][c+4].append(temp[c])
                run_list[-1][-1].append(time() - last_report)
                last_report = time()
            if time() - last > 10:
                print("Epoch {} ({:.4f}s): DE {:.6f}, GE {:.6f}, Acc {:.6f}".format(a,time() - last,sum(run_list[-1][0]) / len(run_list[-1][0]),
                                                                                    sum(run_list[-1][1])/len(run_list[-1][1]),sum(run_list[-1][2]) / len(run_list[-1][2])))
                if eval_mode: print("V {:.4f}, M1 {:.4f}, M2 {:.4f}".format(sum(run_list[-1][4]) / len(run_list[-1][4]),sum(run_list[-1][5]) / len(run_list[-1][5]),sum(run_list[-1][6]) / len(run_list[-1][6])))
                #print("Quick state check: Gen {}, Disc {}".format(*self.avgWB()))
                print("Some words:",self.Generate(10,False))
                last = time()
            
        self.TrainDump(run_list,run_name)

    def GenTrain(self,in_data):
        #Run generator and discriminator, but update generator only
        
        self.gen.zero_grad()
        self.disc.zero_grad()
        
        self.gen.train()
        
        batch_size = len(in_data)
        rl,fl = self.GetLabels(batch_size,0.9,0)
        
        fake_outputs = self.disc(self.Generate(batch_size))
        
        #Possible versions: RSGAN, Wasserstein, simple
        if self.rel:
            real_outputs = self.disc(in_data)
            gen_err = self.cost((fake_outputs - real_outputs),rl)
        elif self.cost == "":
            #Wasserstein loss: For generator, simple:
            gen_err = -fake_outputs.mean()
        else:
            #If not using RS or WGAN, just check fake results against real labels
            gen_err = self.cost(fake_outputs,rl)
        
        #Once we have error, backprop and step
        gen_err.backward()
        self.gen_opt.step()
        
        return gen_err.item()
    
    def DiscTrain(self,in_data):
        #Run the generator and discriminator, and update discriminator only
        #self.gen.zero_grad()
        self.disc.zero_grad()
        
        self.disc.train()
        
        batch_size = len(in_data)
        rl,fl = self.GetLabels(batch_size,0.9,0)
        
        if self.rel:
            real_outputs = self.disc(in_data)
            fake_outputs = self.disc(self.Generate(batch_size))
            d_loss = self.cost(real_outputs - fake_outputs,rl)
            
        
        elif self.cost == "":
            #Wasserstein version
            fake_data = self.Generate(batch_size)
            
            real_output = self.disc(in_data)
            fake_output = self.disc(fake_data)
            
            d_loss = torch.mean(fake_output) - torch.mean(real_output)
            if self.wt == "G":
                gradient_penalty = self.GradPen(in_data.data,fake_data.data)
                d_loss += self.gp * gradient_penalty
        
        else:
            real_loss = self.cost(self.disc(in_data),rl)
            fake_loss = self.cost(self.disc(self.Generate(batch_size).detach()),fl)
            d_loss = (real_loss + fake_loss) / 2
        
        d_loss.backward()
        self.disc_opt.step()
        
        if self.wt == "C":
            for p in self.disc.parameters():
                p.data.clamp_(-1,1)
            
        return d_loss.item()
    
    def Generate(self,samples,raw = True):
        #Same in both, upgraded a bit to avoid needing too much list comprehension outside this function
        self.gen.eval()
        if raw: return self.gen(torch.autograd.Variable(torch.randn((samples,self.n_size))))
        else: return [Decode(a) for a in self.gen(torch.autograd.Variable(torch.randn((samples,self.n_size))))]
    
    def GetLabels(self,batch_size,rsm = 1,fsm = 0):
        #Returns appropriate real/fake labels, smoothed to real-smooth/fake-smooth values (typically 1,0, 0.9,0, or 0.9,0.1)
        if self.o_size == 1:
            return [torch.full((batch_size,1),rsm),torch.full((batch_size,1),fsm)]
        elif self.o_size == 2:
            return [torch.tensor([[fsm,rsm] for a in range(batch_size)]),torch.tensor([[rsm,fsm] for a in range(batch_size)])]
    
    def GradPen(self,real,fake):
        #Gradient penalty function
        #I don't really know how this works, so it's completely possible this is part of what's breaking.
        
        batch_size = real.size()[0]
        
        alpha = torch.rand(batch_size,1).expand_as(real)
        interp = torch.autograd.Variable((alpha * real.data) + ((1-alpha) * fake.data),requires_grad = True)
        
        p_interp = self.disc(interp)
        '''
        gradients = torch.autograd.grad(outputs = p_interp,inputs = interp,grad_outputs = torch.ones(p_interp.size()),create_graph = True,retain_graph = True)[0]
        gradients = gradients.view(batch_size,-1)
        
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2,dim = 1) + 1e-12)
        
        return ((gradients_norm - 1)**2).mean()'''
        
        #Version copied from different code than the above version, will this work?
        
        gradients = torch.autograd.grad(outputs = p_interp,inputs = interp,grad_outputs = torch.ones(p_interp.size()),create_graph = True,retain_graph = True,only_inputs = True)[0]
        gradients = gradients.view(gradients.size(0),-1)
        gradient_penalty = ((gradients.norm(2,dim = 1) - 1) ** 2).mean()
        return gradient_penalty
    
    def Acc(self,in_data):
        #Accuracy rate for the discriminator for analytical purposes
        self.disc.eval()
        self.gen.eval()
        batch_size = len(in_data)
        
        rl,fl = self.GetLabels(batch_size)
        
        gen_data = self.Generate(batch_size)
        
        real_out = torch.round(self.disc(in_data))
        fake_out = torch.round(self.disc(gen_data))
        
        real_acc = torch.sum(torch.eq(real_out,rl)).item()/batch_size
        fake_acc = torch.sum(torch.eq(fake_out,fl)).item()/batch_size
        
        return (real_acc + fake_acc) / 2
    
    def avgWB(self):
        gen_p = self.gen.StateCheck()
        disc_p = self.disc.StateCheck()
        gen_avg = sum([abs(a[0]) for a in gen_p])/len(gen_p)
        disc_avg = sum([abs(a[0]) for a in disc_p])/len(gen_p)
        return [gen_avg,disc_avg]
    
    def TrainDump(self,in_data,out_file):
        #Dump run to text file
        t1 = open("{}.txt".format(out_file).replace(".txt.txt",".txt"),"w")
        t1.write("{}\n\n".format(self.metadata))
        
        #fieldnames = ["DiscE","GenE","DiscAcc","Words","LFreqD","Markov1","Markov2"]
        if len(in_data[0]) == 5: fieldnames = ["DiscE","GenE","DiscAcc","Words","Time"]
        else: fieldnames = ["DiscE","GenE","DiscAcc","Words","LFreqD","Markov1","Markov2","Time"]
        for a in range(len(in_data)):
            t1.write("Epoch {}\n".format(a))
            for b in [0,1,2,4,5,6,7]:
                if b in range(len(in_data[a])):
                    t1.write(fieldnames[b] + ": " + "|".join([str(c) for c in in_data[a][b]]) + "\n")
            t1.write("Words: " + "||".join(["|".join(c) for c in in_data[a][3]]) + "\n\n")
        
        t1.close()
    
    def saveGAN(self,out_prefix):
        t1 = open("ARModels/{} Meta.txt".format(out_prefix),"w")
        t1.write(self.callstr)
        t1.close()
        torch.save(self.gen.state_dict(),"ARModels/{} Gen.pt".format(out_prefix))
        torch.save(self.disc.state_dict(),"ARModels/{} Disc.pt".format(out_prefix))
    
    def loadGAN(self,in_prefix):
        self.gen.load_state_dict(torch.load("ARModels/{} Gen.pt".format(in_prefix)))
        self.disc.load_state_dict(torch.load("ARModels/{} Disc.pt".format(in_prefix)))
                
class Classifier:
    #Simple classifier NN framework
    def __init__(self,params,cost_params):
        #When initializing: give parameters for generator, discriminator, and cost
        #This is sufficiently complex that I'm hesitant to do defaults for the moment
        
        #Metadata: String describing the network to write to log files
        
        n_meta = "\n".join(["Net:","|".join([str(a) for a in params[0]])," || ".join(["|".join(a) for a in params[1]]),"Opt: " + params[2] + " " + str(params[3])])
        self.metadata = "{}\n\n{}".format(" ".join(cost_params),n_meta)
        
        #Next: Define discriminator and generator
        #These are essentilly the same between versions.
        
        self.net = Network(params[0],params[1],f=True)
        if params[2] == "Adam": self.opt = optim.Adam(self.net.parameters(),lr = params[3][0],betas = params[3][1],weight_decay = params[3][2])
        elif params[2] == "Adagrad": self.opt = optim.Adagrad(self.net.parameters(),lr = params[3][0],lr_decay = params[3][1],weight_decay = params[3][2])
        elif params[2] == "Adamax": self.opt = optim.Adamax(self.net.parameters(),lr = params[3][0],betas = params[3][1],weight_decay = params[3][2])
        elif params[2] == "RMSprop": self.opt = optim.RMSprop(self.net.parameters(), lr = params[3][0],alpha = params[3][1],weight_decay = params[3][2])
        elif params[2] == "SGD": self.opt = optim.SGD(self.net.parameters(),lr = params[3][0],weight_decay = params[3][1])
        self.o_size = params[0][-1]

        if cost_params[0] == "BCE": self.cost = nn.BCELoss()
        elif cost_params[0] == "BCEL": self.cost = nn.BCEWithLogitsLoss()
        elif cost_params[0] == "MSE": self.cost = nn.MSELoss()
        elif cost_params[0] == "KLD": self.cost = nn.KLDivLoss()
        elif cost_params[0] == "CEL": self.cost = nn.CrossEntropyLoss()
    
    def Classify(self,in_data):
        if self.o_size != 1:
            return torch.max(self.net(in_data),1).indices.tolist()
        else:
            return torch.round(self.net(in_data)).tolist()
        
    def BatchTrain(self,in_data,in_labels):
        self.net.zero_grad()
        self.net.train()
        
        loss = self.cost(self.net(in_data),in_labels)
        loss.backward()
        self.opt.step()
            
        return loss.item()
    
    def Acc(self,in_data,in_labels):
        #Accuracy rate on indicated data
        self.net.eval()
        if self.o_size != 1:
            test_acc = torch.sum(torch.eq(torch.max(self.net(in_data),1).indices,torch.max(in_labels,1).indices)).item() / len(in_data)
        else:
            test_acc = torch.sum(torch.eq(torch.round(self.net(in_data)),torch.round(in_labels))).item() / len(in_data)
        return test_acc
    
    def Train(self, in_data, epochs, run_name = "rundump",eval_data = "",eval_mode = False):
        #Takes as input lots of things and trains the network
        
        run_list = []
        last = time()
        last_report = time()
        
        for a in range(epochs):
            #If evaluating, we're storing err_g, err_d, acc, words, lc, m1c, m2c
            #Otherwise just err_g,err_d,acc,words
            run_list.append([[],[],[],[]])
            
            for b,data in enumerate(in_data):
                #For each batch:
                inputs,labels = data
                batch_err = self.BatchTrain(inputs,labels)
                t_acc = self.Acc(inputs,labels)
                if eval_mode:
                    e_d = eval_data.data
                    e_l = eval_data.labels
                    batch_acc = self.Acc(e_d,e_l)
                else:
                    batch_acc = self.Acc(inputs,labels)
                
                run_list[-1][0].append(batch_err)
                run_list[-1][1].append(t_acc)
                run_list[-1][2].append(batch_acc)
                run_list[-1][3].append(time() - last_report)
                last_report = time()
    
            if time() - last > 10:
                print("Epoch {}: total time {}, err {}, t_acc {}, v_acc {}".format(a,sum(run_list[-1][3]),sum(run_list[-1][0])/len(run_list[-1][0]),sum(run_list[-1][1])/len(run_list[-1][1]),sum(run_list[-1][2])/len(run_list[-1][2])))
                last = time()
            
        self.TrainDump(run_list,run_name)
    
    def TrainDump(self,run_list,run_name):
        t1 = open("{}.txt".format(run_name).replace(".txt.txt",".txt"),"w")
        t1.write("{}\n\n".format(self.metadata))
        fieldnames = ["Err","T_Acc","V_Acc","Time"]
        for a in range(len(run_list)):
            t1.write("Epoch {}\n".format(a))
            for b in range(3):
                t1.write(fieldnames[b] + ": " + "|".join([str(c) for c in run_list[a][b]]) + "\n")
            t1.write("\n")
        
        t1.close()
        


#TRC, RO, FZI, NetStep, Network are essentially identical between versions
#This version adds the option to use initialization to NetStep

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

class NetStep(nn.Module):
    #Element of a network module
    #Each "layer" of a NetSection always contains one Linear element, and may also contain an activation function, dropout, or a normalization layer
    #For ease of quick use we're going to implement a NetStep as an input size, output size, and three-element list of norm, dropout, activation in that order
    def __init__(self,in_sz,out_sz,steptype = [" "," "," "," "]):
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
        if steptype[1][0] == "d": self.d = nn.Dropout(float(steptype[1][1]))
        else: self.d = ""
        #Activation layer.
        if steptype[2][0] == "s": self.a = nn.Sigmoid()
        elif steptype[2][0] == "t": self.a = nn.Sequential(nn.Tanh(),TRC())
        elif steptype[2][0] == "u": self.a = nn.Tanh()
        elif steptype[2][0] == "r": self.a = nn.ReLU()
        elif steptype[2][0] == "l": self.a = nn.LeakyReLU(float(steptype[2][1:]))
        elif steptype[2][0] == "e": self.a = nn.ELU()
        else: self.a = ""
        
        #Initialize, if we're using that.
        #Default initialization for a nn.linear layer is kaiming_uniform with a=math.sqrt(5)
        
        if steptype[3] != " ":
            nn.init.zeros_(self.l.bias)
            if steptype[3] == "xn":
                nn.init.xavier_normal_(self.l.weight)
            elif steptype[3] == "xu":
                nn.init.xavier_uniform_(self.l.weight)
            elif steptype[3] == "kn":
                nn.init.kaiming_normal_(self.l.weight)
            else:
                nn.init.normal_(self.l.weight)
            
    def forward(self,x):
        #Run x through the linear layer, then through all other layers if present
        x = self.l(x)
        if self.n: x = self.n(x)
        if self.a: x = self.a(x)
        if self.d: x = self.d(x)
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
    
    def StateCheck(self):
        #Gets some info about the current state of the Network
        out_info = []
        for a in self.state_dict():
            t = self.state_dict()[a].tolist()
            if type(t[0]) == list:
                t = [b for c in t for b in c]
            out_info.append([sum(t)/len(t),stdDev(t)])
        return out_info
            