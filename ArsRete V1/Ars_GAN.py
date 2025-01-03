#Ars_GAN
#Make generative network to make fake words that sound like real words!

import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
from Ars_Data import loadRealset
from time import time
import itertools

class RealDataset(Dataset):
    #torch dataset class for loading real words for use training the GAN
    def __init__(self,file_path,mode = 0):
        self.data,self.labels = loadRealset(file_path)
        #print(len(self.data[0]))
    def __len__(self):
        return len(self.data)
    def __getitem__(self,index):
        return self.data[index], self.labels[index]

class QuickNetwork(nn.Module):
    #torch module class for making quick network stacks of arbitrary parameters
    def __init__(self,p_list):
        #Initializes a new QuickNetwork with the given parameters
        #Parameter input formatting is designed for changing the parameters up quickly
        #Format is a list of at least three lists, each containing an int (layer size), a string (activation), and a float (dropout)
        #E.g. [[6,"",1],[100,"r",.5],[20,"s",.5]]
        #The activation and dropout for the first parameter are ignored
        super().__init__()
        temp_layer_list = [nn.Linear(p_list[0][0],p_list[1][0])]
        for a in range(1,len(p_list)):
            #For each parameter, apply activation function, then dropout, then next linear layer
            if p_list[a][1] == "s": temp_layer_list.append(nn.Sigmoid())
            elif p_list[a][1] == "t": temp_layer_list.append(nn.Tanh())
            elif p_list[a][1] == "e": temp_layer_list.append(nn.ELU())
            elif p_list[a][1] == "h": temp_layer_list.append(nn.Hardswish())
            else: temp_layer_list.append(nn.ReLU())
            if p_list[a][2] > 0 and p_list[a][2] < 1: temp_layer_list.append(nn.Dropout(p_list[a][2]))
            else: temp_layer_list.append(nn.Dropout(0))
            temp_layer_list.append(nn.Linear(p_list[a-1][0],p_list[a][0]))
        
        self.model = nn.ModuleList(temp_layer_list)
        
    def forward(self,in_data):
        #Runs the network forward
        temp = in_data
        for a in self.model:
            temp = a(temp)
        return temp

class GenerativeModel():
    #Class to wrap the parameters and training of a generative adversarial network
    def __init__(self,gen_params = [[30,"",""],[500,"r",0.5],[500,"r",0.5],[34,"r",0]],disc_params = [[34,"",""],[500,"r",0.5],[500,"r",0.5],[1,"s",0]]):
        #Create the model
        #Initialize networks and optimizers for generator and discriminator
        #self.generator = QuickNetwork(gen_params)
        self.generator = nn.Sequential(
            nn.Linear(30,250),
            nn.LeakyReLU(),
            nn.Linear(250,500),
            nn.LeakyReLU(),
            nn.Linear(500,250),
            nn.LeakyReLU(),
            nn.Linear(250,34),
            nn.Tanh())
        self.gen_opt = torch.optim.Adam(self.generator.parameters(),lr = 0.0001)
        #self.discriminator = QuickNetwork(disc_params)
        self.discriminator = nn.Sequential(
            nn.Linear(34,250),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(250,500),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(500,250),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(250,1))            
        self.disc_opt = torch.optim.Adam(self.discriminator.parameters(),lr = 0.00001)
        
        #Initialize other parameters of the network
        self.batchsize = 1000
        self.cost = nn.BCEWithLogitsLoss() #Cost function
        self.fixed_noise = torch.div(torch.randn(self.batchsize,1,30),2) #Constant noise across loops to watch the evolution of the network
        #Real and fake labels for training the network 
        self.real_batch = torch.full((self.batchsize,1,1),1.0)
        self.fake_batch = torch.full((self.batchsize,1,1),0.0)
    
    def trainGenerator(self,in_data,loops,vis = True):
        #Train the generative model
        generated_words = []
        g_losses = []
        d_losses = []
        
        print("Beginning training loop")
        last_print = time()
        for loop in range(loops):
            #For each run through loops:
            epoch_loss = [0,0]
            for i,data in enumerate(in_data):
                
                #__STEP 1: UPDATE DISCRIMINATOR__
                inputs,labels = data
                
                self.discriminator.zero_grad() #Zero the discriminator gradient
                output_1 = self.discriminator(inputs) #Run forward
                error_1 = self.cost(output_1,labels) #Compute error
                error_1.backward(retain_graph = True) #Run backward
                
                noise = torch.div(torch.randn(self.batchsize,1,30),2) #Generate random noise for the generator
                fake_intermediates = self.generator(noise) #Generate inputs to the discriminator
                output_2 = self.discriminator(fake_intermediates) #Run forward
                error_2 = self.cost(output_2,self.fake_batch) #Compute error on fake samples
                error_2.backward(retain_graph = True) #Run backward
                
                #Save discriminator error, then step
                epoch_loss[0] += (error_1+error_2)
                self.disc_opt.step()
                
                #__STEP 2: UPDATE GENERATOR__
                self.generator.zero_grad() #Zero the generator gradient
                output_3 = self.discriminator(fake_intermediates) #Run forward
                error_3 = self.cost(output_3,self.real_batch) #Compute error on fake samples as if they were real
                error_3.backward() #Run backward
                
                #Save generator error, then step
                epoch_loss[1] += error_3
                self.gen_opt.step()
                
            #Add average loss per batch to the list for graphing 
            d_losses.append(epoch_loss[0].item() / i)
            g_losses.append(epoch_loss[1].item() / i)
            
            #Print status update
            if time() - last_print > 10:
                print("Loop {} status:".format(loop))
                print("Disc error: {}, Gen error: {}".format(epoch_loss[0],epoch_loss[1]))
                print("Some random words at last loop:")
                print([decode(a) for a in fake_intermediates[:5]])
                print("Status of fixed noise:")
                print([decode(a) for a in self.generator(self.fixed_noise)[:5]])
                last_print = time()
        if vis:
            plt.plot(d_losses)
            plt.plot(g_losses)
            plt.show()

def decode(in_binary):
    #Converts binary number into a word via converting to base-27
    decimal = int("".join([str(round((a + 1) / 2)) for a in in_binary.tolist()[0]]),2)
    digits = []
    while decimal:
        digits.append(decimal % 27)
        decimal = decimal // 27
    return "".join([chr(96+a) if a != 0 else " " for a in digits])


def main():
    realsets = [DataLoader(RealDataset("5_6_quickset_{}.json".format(a)),1000,shuffle = True) for a in "abcdefghij"]
    
    TestNet = GenerativeModel()
    
    TestNet.trainGenerator(realsets[0],100)
                
                
main()