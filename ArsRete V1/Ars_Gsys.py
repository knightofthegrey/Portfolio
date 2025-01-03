#ARS GENERATOR
#Version 1, 11/23
#This was an attempt to create a generative adversarial network structure using my Ars_Network v3 code

from Ars_Network import Network
import numpy as np
import time
import random
import math

#Concept: Implement adversarial generative network
#Goal: Create a generator that makes words that look like real words from the control set but are not
#Alph is used to convert things from lists of floats to strings or vice versa, to avoid needing to fiddle with the ord of non-sequential characters
#At the moment this is set up primarily for English, but it shouldn't take a huge amount of work to adapt it to other character sets
alph = " abcdefghijklmnopqrstuvwxyz"
#Status: Runs! Doesn't *work*, but runs!


class Generator_System:
    #In an adversarial generator system we have two networks: the Discriminator trains to distinguish generated samples
    #from the control, and the Generator trains to make samples that the Discriminator has a hard time telling apart from the control.
    
    #Generator System Pseudocode
    #Alternate steps.
    #1: Generator runs and produces output based on random input seed.
    #2: Discriminator runs on generator output and input from real set.
    #3: Generator runs backprop through discriminator to generator, but only updates w/b in generator
    #4: Discriminator runs on generator output and input from real set.
    #5: Discriminator runs backprop on self only
    #6: Repeat
    
    def __init__(self,disc_sizes,gen_sizes,real_source = "padwords.txt",fake_source = "padfakes.txt",constraint = ""):
        #This initializes a new adversarial generator system
        #We have two Network objects with sizes given by the integer lists in the input.
        #real_source and fake_source are the relative path to text files giving a set of real/fake inputs to train against
        #The generator's output must be the same size as the discriminator's input, but otherwise these are unconstrained.
        #Parameters assumed for testing purposes:
        #-Outputs from the generator and inputs to the discriminator are words of no more than 14 letters in length.
        #-Letters in inputs/outputs are represented by floats between 0 and 1, and converted back to letters by multiplying by 27 (letters or a space for padding)
        #-Words are padded front and back with spaces to a total length of 14, preferring back padding
        self.Discriminator = Network(disc_sizes)
        self.Discriminator.costmode = 1
        self.Generator = Network(gen_sizes)
        self.Generator.costmode = 1
        self.real_data = self.readData(real_source,1,constraint)
        self.fake_data = self.readData(fake_source,0,constraint)
        #cross_weights should be an identity matrix of size equal to gen outputs/disc inputs
    
    def readData(self,in_path,e_out,constraint):
        #This reads data from the input path and uses it to make input,output pairs.
        #in_path points to a .txt file containing words pre-padded to 14 characters
        #Return is a list of input,output,string tuples, as np arrays of floats
        out_data = []
        for a in open(in_path,"r").read().splitlines():
            if constraint:
                alist = np.array([alph.index(b) / 27 for b in a if len(a) == constraint])
            else:
                alist = np.array([alph.index(b) / 27 for b in a ])
            out_data.append((np.array([alist]).T,np.array([[1] if a == e_out else [0] for a in range(2)])))
        random.shuffle(out_data)
        return out_data
    
    def getDataSet(self,set_size,real = True):
        #Gets a random selection of data from self.real_data or self.fake_data, controlled by the real bool
        #Returns a list of input,output pairs
        if real:
            return random.choices(self.real_data,k = set_size)
        else:
            return random.choices(self.fake_data,k = set_size)
    
    def trainNetwork(self,gen_size,loops = 1000,check_interval = 100,gen_rate = 0.03,disc_rate = 0.03):
        #This is the main training loop.
        #Inputs:
        '''
        gen_size: int, number of samples to generate
        loops: int, number of times to loop
        check_interval: int, print output every time you pass this many loops
        gen_rate: generator base learning rate
        disc_rate: discriminator base learning rate
        '''
        #Outputs: None
        for j in range(loops):
            #Step 1: Train the generator
            #This is a separate function rather than simply calling the Network.learn function,
            #because it has to run backprop through both discriminator and generator
            for n in range(1):
                self.genLearn(gen_size,gen_rate)
            gen_out = self.generate(gen_size)
            #Step 2: Get the generator output and some real data, then mix them
            temp_data = self.getDataSet(gen_size)
            blended_data = gen_out + temp_data
            random.shuffle(blended_data)
            #Step 3: Train the discriminator on blended_data
            self.Discriminator.learn(blended_data,1,disc_rate)
            #Step 4: If we're at a check interval, print some output
            if (j+1) % check_interval == 0:
                #Give the state of the discriminator based on the Network.evaluate function
                normal_mix = self.getDataSet(gen_size) + self.getDataSet(gen_size,False)
                random.shuffle(normal_mix)
                d_acc, d_err = self.Discriminator.evaluate(blended_data)
                print("Discriminator at step {}: {:.3f}% accuracy, {:.6f} error".format(j,d_acc*100,d_err))
                #Print the latest outputs of the generator so we can see if they do look like real words
                print("Generator outputs: {}".format([self.outputToString(a[0]) for a in gen_out]))
    
    def outputToString(self,in_list):
        #Turns an array of floats from 0 to 1 into strings
        #We use math.floor here because the output should be from 0 to 26, except in the vanishingly rare case where it's actually exactly 1,
        #in which case we use min to peg it back to 26 so we don't get an index error
        letter_list = [alph[min(math.floor(a[0] * 27),26)] for a in in_list]
        return "".join(letter_list)
    
    def generate(self,gen_size):
        #Gets output from the generator, formatted as input for the discriminator
        #Input is an integer, output is a list of input,output pairs. Sizes are controlled by the generator Network object.
        gen_in = [np.random.randn(1,self.Generator.sizes[0]) for a in range(gen_size)]
        gen_out = [(self.Generator.feedForward(a.T),np.array([[1],[0]])) for a in gen_in]
        return gen_out
    
    def discLearn(self,gen_size,learn_rate,iterations,eval_):
        #Runs backprop for discriminator only to give the discriminator a training headstart
        #gen_size is the number of samples to generate (int), learn_rate is the learning rate (float)
        #This function only calls self.Discriminator.learn() and does not return
        real_set = self.getDataSet(gen_size // 2)
        fake_set = self.getDataSet(gen_size // 2,False)
        blended_set = real_set + fake_set
        random.shuffle(blended_set)
        self.Discriminator.learn(blended_set,iterations,learn_rate,evaluate = eval_)
    
    def genLoop(self,gen_size,learn_rate,iterations):
        #Trains the generator only to see what happens with a generator given varying qualities of discriminator
        #gen_size is the number of samples to generate (int), learn_rate is the learning rate (float)
        #This function only calls self.genLearn() and does not return
        for a in range(iterations):
            self.genLearn(gen_size,learn_rate)
            if a % 20 == 0:
                temp_gen = self.generate(gen_size)
                temp_eval = self.getDataSet(gen_size) + temp_gen
                random.shuffle(temp_eval)
                eval_data = self.Discriminator.evaluate(temp_eval)
                print("Generator outputs at step {}:".format(a))
                print([self.outputToString(a[0]) for a in random.choices(temp_gen,k = 10)])
                print("Discriminator effectiveness: {:.5f}%, error {:.5f}".format(eval_data[0] * 100,eval_data[1]))
    
    def genLearn(self,gen_size,learn_rate):
        #Runs backprop through both generator and discriminator, but only updates generator
        #gen_size is the number of samples to generate, learn_rate is the learning rate
        #This function updates self.Generator's parameters and does not produce outputs
        #The generator's input is a set of random variables, and has the expected output value [[1],[0]] because if the discriminator were correct it would recognize them as fake.
        gen_in = [(np.random.randn(1,self.Generator.sizes[0]),np.array([[1],[0]])) for a in range(gen_size)]
        cum_nabla_b = [np.zeros(b.shape) for b in self.Generator.biases]
        cum_nabla_w = [np.zeros(w.shape) for w in self.Generator.weights]
        #Backpropagation:
        for a in gen_in:
            nabla_b = [np.zeros(b.shape) for b in self.Generator.biases]
            nabla_w = [np.zeros(w.shape) for w in self.Generator.weights]            
            #For each input:
            act = a[0].T
            act_list = [act]
            z_list = []
            for n in range(self.Generator.depth - 1):
                w = self.Generator.weights[n]
                b = self.Generator.biases[n]
                z = np.dot(w,act) + b
                z_list.append(z)
                act = self.Generator.activation(z,n == self.Generator.depth - 2)
                act_list.append(act)
            #Now, run through the discriminator.
            for n in range(self.Discriminator.depth - 1):
                w = self.Discriminator.weights[n]
                b = self.Discriminator.biases[n]
                z = np.dot(w,act) + b
                z_list.append(z)
                act = self.Generator.activation(z,n == self.Discriminator.depth - 2)
                act_list.append(act)
            #At the end run the cost function
            #delta here is negative, because the generator seeks to maximize the discriminator's errro
            delta = -self.Discriminator.costDeriv(act_list[-1],a[1]) * self.Discriminator.actDeriv(z_list[-1],True)
            for n in range(2,self.Discriminator.depth):
                #Running backwards through the discriminator using negative Python indexes.
                #For this stage we don't do any nablas/update any weights or biases, as we only update the generator at this step
                z = z_list[-n]
                sp = self.Discriminator.actDeriv(z)
                delta = np.dot(self.Discriminator.weights[-n+1].T,delta) * sp
                #print(sp.size,delta.size,z.size)
            #Now, we run through the generator and do this all again, this time updating weights and biases
            #crossover_weights is an extra array of 1s in the correct dimensions to allow us to run the backprop transform.
            #Normally a network has weights in between its own layers but none externally, creating a fencepost problem for running through two concatenated networks.
            crossover_weights = np.zeros((self.Discriminator.sizes[1],self.Discriminator.sizes[0])) + 1
            delta = np.dot(crossover_weights.T,delta)
            nabla_b[-1] = delta
            nabla_w[-1] = np.dot(delta,act_list[-(self.Discriminator.depth +1)].T)
            for n in range(2,self.Generator.depth):
                z = z_list[-(self.Discriminator.depth - 1 + n)]
                sp = self.Generator.actDeriv(z)
                delta = np.dot(self.Generator.weights[-n+1].T,delta) * sp
                nabla_b[-n] = delta
                nabla_w[-n] = np.dot(delta,act_list[-(self.Discriminator.depth - 1) - n - 1].T)
            cum_nabla_w = [w+nw for w,nw in zip(cum_nabla_w,nabla_w)]
            cum_nabla_b = [b+nb for b,nb in zip(cum_nabla_b,nabla_b)]
        self.Generator.weights = [w + (learn_rate / gen_size) * nw for w,nw in zip(self.Generator.weights,cum_nabla_w)]
        self.Generator.biases = [b + (learn_rate / gen_size) * nb for b,nb in zip(self.Generator.biases,cum_nabla_b)]
            

def main():
    testGen = Generator_System([14,65,47,2],[12,73,65,93,14])
    #testGen.headstartTraining()
    #Testing status: Generator system produces output.
    testGen.trainNetwork(100,loops = 1000,check_interval = 20,gen_rate = 0.007,disc_rate = 0.0002)
    #print(testGen.Discriminator.evaluate(testGen.getDataSet(100,True)))
    

main()
                