#ARS NETWORK
#Version 3, 11/23
#Third attempt at creating a neural network from scratch using numpy arrays

import numpy as np
import matplotlib.pyplot as plt
import time
import math

#__GLOBAL VARIABLES__
#Relative paths to save and load locations
global_save_path = "ArsRete"

class Network:
    #This class implements a simple MLP neural network
    
    #__FUNCTIONS TO INITIATE AND SAVE THE NETWORK__
    #Networks are initialized with random values, but can be saved to file and read from file as well,
    #thereby preserving themselves across multiple runs of the program
    def __init__(self,sizes,modes = [0,1,0],in_save_path = global_save_path):
        #Initializes a new Network object with random weights and biases
        #Sizes is a list of integers representing the number of nodes at each layer.
        #Modes is a list of integers representing which activation and cost functions a Network object is using
        #The first value in sizes is the expected size of inputs, and the last is the size of outputs.
        #__1: Basic information about the network__
        self.depth = len(sizes) #Depth of the network for looping purposes
        self.sizes = sizes
        #__2: Initializes the biases of the network as numpy arrays of random numbers__
        #Each array is of size y,1 where y is the size of the associated layer
        #We run from sizes[1:] because the input layer has no biases
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        #__3: Initializes the weights of the network as numpy arrays of random numbers__
        #Each array connects two nodes, and its dimensions are the size of the second and the size of the first
        #For this reason we have one fewer weight array than depth
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]
        #__4: Initializes the cost and activation modes of the network
        #These may be changed depending on what we're doing
        #Default mode right now is tanh internal, sigmoid final, mse cost.
        if modes[0] in [0,1]: self.costmode = modes[0]
        else: self.costmode = 0
        if modes[1] in [0,1]: self.actmode = modes[1]
        else: self.actmode = 0
        if modes[2] in [0,1]: self.finalmode = modes[2]
        else: self.finalmode = 0
        #__5: Save path__
        self.save_path = in_save_path

    def saveToFile(self,filename):
        #Save the network variables to file for future use.
        #The network will be saved as three separate files, one for super-parameters, one for weights, and one for biases,
        #but they are referenced by one "filename" string
        outFile = open("{}/{}.txt".format(self.save_path,filename),"w")
        outFile.write("|".join(self.sizes() + [self.num_layers,self.costmode,self.actmode,self.finalmode]))
        outFile.close()
        np.save("{}/{} biasarray.npy".format(self.save_path,filename),self.biases)
        np.save("{}/{} weightarray.npy".format(self.save_path,filename),self.weights)

    def readFromFile(self,filename):
        #Read a new network in from a file
        smallData = open("{}/{}.txt".format(self.save_path,filename),"r").read()
        self.num_layers = int(smallData.split("|")[-4].strip())
        self.costmode = int(smallData.split("|")[-3].strip())
        self.actmode = int(smallData.split("|")[-2].strip())
        self.finalmode = int(smallData.split("|")[-1].strip())
        self.sizes = [int(x) for x in smallData.split("|")[:-4]]
        self.biases = np.load("{}/{} biasarray.npy".format(self.save_path,filename))
        self.weights = np.load("{}/{} weightarray.npy".format(self.save_path,filename))
    
    #__ACTIVATION AND COST FUNCTIONS__
    #The theory here is to allow us to vary these programmatically for quick tests of different kinds
    def activation(self,data,final = False):
        #Uses self.actmode or self.finalmode (controlled by final) to decide which activation function to use, then applies it to the input data.
        #data is a 2d numpy array.
        #Implemented activation functions are: 0 sigmoid, 1 tanh, 2 relu, and 3 swish. At the moment modes 0 and 1 have been well-tested, modes 2 and 3 have not.
        if final: mode = self.finalmode
        else: mode = self.actmode
        if mode == 0:
            #Sigmoid: f(x) = 1/(1+e**-x)
            return 1/(1+np.exp(-data))
        elif mode == 1:
            #Tanh: (e**x - e**-x)/(e**x + e**-x), but numpy has a built-in here
            return np.tanh(data)
        elif mode == 2:
            #ReLu: f(x) = max(0,x)
            return np.maximum(0,data)
        elif mode == 3:
            #Swish: f(x) = x*sigmoid(x)
            return data/(1+np.exp(-data))

    def actDeriv(self,data,final = False):
        #We need to be able to use the derivative of the activation function for backpropagation
        #This is used on arrays of activations during backprop
        #Final is used because some neural nets might use different activation functions for the output from the inputs
        #Implemented activation functions are: 0 sigmoid, 1 tanh, 2 relu, and 3 swish. At the moment modes 0 and 1 have been well-tested, modes 2 and 3 have not.
        if final: mode = self.finalmode
        else: mode = self.actmode
        if mode == 0:
            #The derivative of sigmoid: f'(x) = f(x) * (1-f(x))
            #We know that the mode is the same for this and for activation since it's a class variable,
            #so calling activation here gets us the sigmoid version
            return self.activation(data,final) * (1-self.activation(data,final))
        elif mode == 1:
            #The derivative of tanh: f'(x) = 1-f(x)**2
            return 1-(self.activation(data,final)**2)
        elif mode == 2:
            #ReLu derivative is parametric, 1 for x>0, 0 for x<= 0
            return (data > 0) * 1
        elif mode == 3:
            #Derivative of x*(1/(1+e**-x)) takes some doing
            #Can be simplified to (xe**-x) / (1+e**-x) **2 + 1/(1+e**-x), which isn't...great, but it's what we've got
            return ((data * np.exp(-data)) / ((1+np.exp(-data))**2)) + (1/(1+np.exp(-data)))

    def cost(self,expected,actual):
        #Cost functions for the network
        #We use the cost derivative more than the cost, so these are sometimes chosen for simple derivatives
        #expected is a np array of the expected outputs of the network, actual is the actual outputs
        #Modes are: 0 quadratic/MSE, 1 cross-entropy, 2 Hellinger distance, 3 K-L divergence
        #For this purpose we need the average cost, so we divide by the number of samples.
        #At the moment modes 1 and 2 have been well-tested, modes 2 and 3 have not
        m = actual.shape[1]
        if self.costmode == 0:
            #Squared error: (expected - actual)**2/2
            return np.squeeze(((expected - actual) ** 2) / (2*m))
        elif self.costmode == 1:
            #Cross-entropy: -(expected*ln(actual) + (1-expected) * len(1-actual))
            return np.squeeze(((-1 * (expected*np.log(actual))) + ((1-expected) * np.log(1-actual))) / m)
        elif self.costmode == 2:
            #Hellinger distance: (sqrt(actual) - sqrt(expected))**2 * 1/sqrt(2)
            #Note that doing sqrts means we need actual and expected outputs above 0
            return np.squeeze(((np.sqrt(actual) - np.sqrt(expected))**2) / (m * np.sqrt(2)))
        elif self.costmode == 3:
            #Kullback-Leibler divergence: expected * ln(expected / actual)
            return np.squeeze((expected * np.log(expected / actual)) / m)
    
    def costDeriv(self,actual,expected):
        #The derivative of the cost function, used for backpropagating
        #Modes are the same as for cost. Note that modes 0 and 1 have been well-tested, modes 2 and 3 have not.
        m = actual.shape[1]
        if self.costmode == 0:
            #Squared error
            return (actual - expected) / m
        elif self.costmode == 1:
            #Cross-entropy
            return -(((actual - expected) / ((1-actual) * actual))/m)
        elif self.costmode == 2:
            #Hellinger distance
            return ((np.sqrt(actual) - np.sqrt(expected)) / (np.sqrt(2) * np.sqrt(actual)))/m
        elif self.costmode == 3:
            #Kullback-Leibler divergence
            return (-expected / actual)/m
    
    #__FUNCTIONS FOR RUNNING AND TRAINING THE NETWORK__
    def feedForward(self,a):
        #Runs a forward pass through the network and produces an output
        #a is the input activations, as a numpy list with dimensions input size * number of elements
        for n in range(self.depth - 1):
            #The n == self.depth - 2 term is used to check if we're on the final layer, so we can use the final layer activation if it's different
            a = self.activation(np.dot(self.weights[n],a) + self.biases[n],n == self.depth - 2)
        return a
    
    def backprop(self,inputs,expected):
        #Runs through the network, and returns lists of NP arrays representing the change in weights and biases
        #Inputs is an inputsize x numsamples np array, expected is outputsize x numsamples
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        #Forward pass through the network
        #At each step save input activations A and output activations Z from a given layer
        act = inputs
        actList = [act]
        zList = []
        for n in range(self.depth - 1):
            w = self.weights[n]
            b = self.biases[n]
            z = np.dot(w,act) + b
            zList.append(z)
            act = self.activation(z,n == self.depth - 2)
            actList.append(act)
        #Then, we need to do the backwards pass through the network, updating nablas as we go
        delta = self.costDeriv(actList[-1],expected) * self.actDeriv(zList[-1],True)
        #print(delta.shape)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta,actList[-2].T)
        for n in range(2,self.depth):
            #Running backwards here via Python negative list indexes
            z = zList[-n]
            sp = self.actDeriv(z)
            delta = np.dot(self.weights[-n+1].T,delta) * sp
            nabla_b[-n] = delta
            nabla_w[-n] = np.dot(delta,actList[-n - 1].T)
        return nabla_b,nabla_w
    
    def update(self,batch,learnRate):
        #Updates weights and biases based on the output from the batch
        #Inputs: batch is a list of inputs,expected outputs pairs. learnRate is a float.
        #This function does not return anything, it modifies the properties of the network in place.
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x,y in batch:
            #For each sample, find suggested deltas for that sample, then add them to the running nablas
            delta_b, delta_w = self.backprop(x,y)
            nabla_b = [nb + dnb for nb,dnb in zip(nabla_b,delta_b)]
            nabla_w = [nw + dnw for nw,dnw in zip(nabla_w,delta_w)]
        #Update the weights, biases with the average update
        self.weights = [w + (learnRate / len(batch)) * nw for w,nw in zip(self.weights,nabla_w)]
        self.biases = [b + (learnRate / len(batch)) * nb for b,nb in zip(self.biases,nabla_b)]
        
    def parameterizedUpdate(self,batch,learn_rate,gamma):
        #Allows for the use of dynamic learning rates adapted for each parameter
        #Loosely based on the RMSProp algorithm
        #Inputs:
        '''
        batch: List of tuples of np arrays of input/output pairs, 2d
        learn_rate: float base learning rate
        gamma: float between 0 and 1, the higher the number the more the last delta influences the current delta.
        '''
        nabla_b = [np.zeroes(b.shape) for b in self.biases]
        nabla_w = [np.zeroes(w.shape) for w in self.weights]
        #Save past deltas
        last_b = []
        last_w = []
        for x,y in batch:
            #For each input, run the backprop algorithm
            delta_b,delta_w = self.backprop(x,y)
            if last_b:
                #If this isn't the first loop, modify delta_b,delta_w based on last_b,last_w
                #Formula: delta = gamma * last delta + (1-gamma) * current delta for weight/bias
                delta_b = [(gamma * lb) + ((1-gamma) * (db ** 2)) for lb,db in zip(last_b,delta_b)]
                delta_w = [(gamma * lw) + ((1-gamma) * (dw ** 2)) for lw,dw in zip(last_w,delta_w)]
            #Modify running totals based on delta_b, delta_w
            nabla_b = [nb + dnb for nb,dnb in zip(nabla_b,delta_b)]
            nabla_w = [nw + dnw for nw,dnw in zip(nabla_w,delta_w)]
            #Update last_b, last_w
            last_b = delta_b
            last_w = delta_w
        #Modify weights, biases based on cumulative nabla lists
        self.weights = [w - (learnRate / len(batch)) * nw for w,nw in zip(self.weights,nabla_w)]
        self.biases = [b - (learnRate / len(batch)) * nb for b,nb in zip(self.biases,nabla_b)]        
            
    def learn(self,input_data,iterations,learn_rate,lr_mode = 0,batch_size = "",evaluate = False,vis = False,eval_input = [],eval_time = 30):
        #Train the neural network using gradient descent
        #Can use stochastic batches, but doesn't have to
        #This is the main training loop that will be called from outside the Network object
        #Inputs:
        '''
        input_data: list of input/expected output pairs (np arrays)
        iterations: int, number of times to run through the input data
        learn_rate: float, base learning rate
        lr_mode: int, controls dynamic learning rate mode. -1 uses RMSProp, 0 is static, 1 and 2 are different fixed decay rates (see dynamicLRate for details)
        batch_size: int, number of samples to run as a batch. Warning: If the batch size isn't an even divisor of the size of input_data some data will be cut off the end of your run.
        evaluate: bool, controls if the program does an evaluation run after each training run
        vis: bool, controls if the program displays a graph of the accuracy and error after running. Requires evaluate to be enabled or it won't work.
        eval_input: list of input/expected output pairs (np arrays). If given evaluate runs on these instead of on the training samples.
        eval_time: int or float, if evaluate is True this is how often the program prints the evaluate results while running in seconds.
        '''
        #Returns list of three lists of floats (evaluate error, evaluate accuracy, learning rate) for use in comparing different runs.
        n = len(input_data)
        eval_list = [[],[],[]] #Empty list for storing evaluate data
        last_eval = time.time()
        for j in range(iterations):
            if batch_size:
                #If we have a batch_size variable, shuffle the input data and divide the inputs into that many batches.
                random.shuffle(input_data)
                batches = [input_data[a:a+batch_size] for a in range(1,n,batch_size)]
            else:
                #Otherwise, there's just one batch, which is the whole input_data
                batches = [input_data]
            if lr_mode >= 0:
                #For a fixed learning rate schedule, use the basic update function and append learning rate to eval_list
                for a in batches:
                    self.update(a,self.dynamicLRate(learn_rate,j,lr_mode))
                    eval_list[2].append(self.dynamicLRate(learn_rate,j,lr_mode))
            elif lr_mode == -1:
                #For RMSProp, use the parameterized update function. In this case there's no single learning rate, so we don't store it in eval_list
                for a in batches:
                    self.parameterizedUpdate(a,learn_rate,0.2)
            if evaluate:
                #If the function call says to evaluate the output:
                if eval_input:
                    #If we have test inputs, run evaluate on the test inputs
                    eval_data = self.evaluate(eval_input)
                else:
                    #Otherwise, run on the learning inputs
                    eval_data = self.evaluate(input_data)
                eval_list[0].append(eval_data[0])
                eval_list[1].append(eval_data[1])
                if time.time() - last_eval > eval_time:
                    #If it's been more than eval_time since the last time we printed an output, print the output and update the last_eval time.
                    print(j,eval_data)
                    last_eval = time.time()
        if evaluate and vis:
            #After running the loop use matplotlib to show a graph of error and accuracy of evaluate steps.
            plt.plot(eval_list[0])
            plt.plot(eval_list[1])
            plt.show()
        return eval_list
    
    def dynamicLRate(self,initial,loop,mode,k=0.2):
        #Sets the learning rate dynamically in one of several modes.
        #Inputs:
        '''
        initial: float, initial learning rate
        loop: int, the current loop we're on when running this
        mode: int, 0 for constant, 1 for 1/1+(k*j), 2 for 1/(e^(k*j))
        k: float: coefficient. Typically set to <1 to slow down the decay rate.
        '''
        #Outputs the current learning rate for the given mode, loop, and k
        if mode == 0:
            #Constant rate
            return initial
        elif mode == 1:
            #Time-based decay
            return initial * (1/(1+k*loop))
        elif mode == 2:
            #Exponential time-based decay
            return initial * (1/(math.e**(k*loop)))

    def evaluate(self,input_data):
        #Runs test data, then sees how good the result is
        #Inputs: input_data: list of input,expected pairs as np arrays of floats
        #Outputs: two floats, average accuracy and average error across the whole input list
        #Accuracy is currently set up for checking whether the highest value in the input is the same as the highest value in the output
        run_data = [self.feedForward(x[0]) for x in input_data]
        AccList = [int(np.argmax(run_data[a].T) == np.argmax(input_data[a][1])) for a in range(len(input_data))]
        ErrorList = [np.sum(self.cost(input_data[a][1],run_data[a])) / 2 for a in range(len(input_data))]
        return sum(AccList) / len(input_data),sum(ErrorList) / len(input_data)

#__TESTING FUNCTIONS FOR COMPARING SUPERPARAMETERS__

def compareFunctions(inData,trials = 3):
    #In order to compare the performance of different functions we need a network that runs quickly, and a number of trials sufficient to see the differences
    #At time of writing most of our functions give div/0 errors when actually tried, but sigmoid, tanh, and mse are working, so we can at least try those.
    #We need to make a new network on each trial, which means we need to run several trials per combination to account for random variation in start conditions.
    for a in [[0,0],[0,1],[1,0],[1,1]]:
        #For each combination of activation functions:
        evalData = []
        for b in range(trials):
            #For each trial, initialize a new network, run it 500 times, and append the evaluation to evalData
            testNet = Network([14,20,25,2])
            testNet.actmode = a[0]
            testNet.finalmode = a[1]
            evalData.append(testNet.learn(inData,100,0.1,evaluate = True))
        #We'd like to be able to visualize the average performance of a network using this function, so let's average all the evalData
        #evalData is a [[[float],[float]]] where the first is the accuracy over the training set and the second is the error
        #To get the averages:
        #For visualization purposes I'd like to save these plots for viewing later
        savenamedict = {0:"Sigmoid",1:"TanH"}
        sn = "Main {}, Final {}.png".format(savenamedict[a[0]],savenamedict[a[1]])
        sn1 = "Accuracy {}".format(sn)
        sn2 = "Error {}".format(sn)
        accList = [a[0] for a in evalData]
        avgAcc = [sum([a[b] for a in accList]) / trials for b in range(len(accList[0]))]
        plt.plot(avgAcc)
        errList = [a[1] for a in evalData]
        avgErr = [sum([a[b] for a in errList]) / trials for b in range(len(errList[0]))]
        plt.plot(avgErr)
        plt.savefig(sn2)
        
def tuneParameters(inData,inEval):
    #Given roughly 40,000 w/b spread across n layers does the distribution matter?
    #Factoring 1430
    #One internal layer has size about 1430 for this, let's consider some other possibilities
    #Our triples: 
    triples = [[5,11,25],[5,13,21],[5,15,19],[7,9,23],[7,13,15]]
    #Each of these are all different, all odd, and have a product between 1350 and 1430
    for a in triples[:1]:
        permutations = [[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,1,0],[2,0,1]]
        for b in permutations:
            netsize = [14] + [a[z] for z in b] + [2]
            evalData = []
            for c in range(10):
                testNet = Network(netsize)
                evalData.append(testNet.learn(inData,100,0.001,"",True,False,inEval))
                accList = [a[0] for a in evalData]
            avgAcc = [sum([a[b] for a in accList]) / 10 for b in range(len(accList[0]))]
            plt.plot(avgAcc)
            errList = [a[1] for a in evalData]
            avgErr = [sum([a[b] for a in errList]) / 10 for b in range(len(errList[0]))]
            plt.plot(avgErr)
            plt.savefig("{}-{}-{}.png".format(a[b[0]],a[b[1]],a[b[2]]))


    
            
    
        