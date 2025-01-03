#ArsNet
#Version 2 - 10/23
#Second attempt at building a neural network from scratch using numpy arrays.

import random
import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt


class Network:
    #The Network class creates and runs a simple neural network that can learn via SGD
    def __init__(self,sizes):
        #Initializes a new network with random weights and biases
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]
    def saveToFile(self,filename):
        #Save the network variables to a file for future use
        outPath = "ArsRete"
        outFile = open("{}/{}.txt".format(outPath,filename),"w")
        outFile.write("|".join(self.sizes() + [self.num_layers]))
        outFile.close()
        np.save("{}/{} biasarray.npy".format(outPath,filename),self.biases)
        np.save("{}/{} weightarray.npy".format(outPath,filename),self.weights)
    def readFromFile(self,filename):
        #Read a new network in from a file
        inPath = "ArsRete"
        smallData = open("{}/{}.txt".format(inPath,filename),"r").read()
        self.num_layers = int(smallData.split("|")[-1].strip())
        self.sizes = [int(x) for x in smallData.split("|")[:-1]]
        self.biases = np.load("{}/{} biasarray.npy".format(inPath,filename))
        self.weights = np.load("{}/{} weightarray.npy".format(inPath,filename))
    
    def feedForward(self,a):
        #Runs the network on an input and produces an output
        #'Kay. We can now feedForward and get a sensible output. Next problem: Doing so while maintaining the backprop data.
        for b,w in zip(self.biases,self.weights):
            a = sigmoid(np.dot(w,a) + b)
        return sigmoid(np.sum(a,axis = 1,keepdims = True))
    def forwardSave(self,a):
        #Runs the network on an input and produces an output, saving the activations for later
        Zlist = []
        A = a
        Alist = [A]
        for b,w in zip(self.biases,self.weights):
            Z = np.dot(w,A) + b
            Zlist.append(Z)
            A = np.tanh(Z) #Activations! Can change this if needed?
            Alist.append(A)
        return Zlist,Alist,sigmoid(Zlist[-1])
    
    def backprop(self,inAct,expVals):
        #Uses backpropagation to find the changes to various spots in the network
        Zlist,Alist,y = self.forwardSave(inAct)
        m = inAct.shape[1]
        dy = (y-expVals)**2
        dBlist = [np.zeroes(b.shape) for b in self.biases]
        dWlist = [np.zeroes(w.shape) for w in self.weights]
        dBlist[-1] = (Alist[-1] - y)
        
            
    def backpropSingle(self,a,expected):
        #Runs the network forward on inputs, and keeps the activations of the network
        #a should be a 2d numpy array of dimensions inputsize x numinputs, expected should be a 2d numpy array of dimensions outputsize x numinputs
        #But we're not getting that, which I assume is some kind of weird NP convenience
        #Input activation
        act = a
        actList = [a]
        zList = []
        #Then we run through the network forward
        for b,w in zip(self.biases,self.weights):
            z = np.dot(w,act) + b #Raw, unsigmoided vector
            zList.append(z)
            act = sigmoid(z) #Sigmoided activations at that layer
            actList.append(act)
        delta = self.cost(actList[-1],expected)
        #With delta, we should then be able to run through the network backwards and find out deltas for weights, biases
        deltaWeightList = []
        deltaBiasList = []
        print(a.size)
        #for a in range(self.sizes - 1,0,-1):
            #Going backwards through the network
            #deltaWeightN = 
        #Works, now! Next: 
        return actList,zList,delta
    def cost(self,actual,expected):
        #For now just difference error here? Is that why we can't converge? Hrm. Currently squared error.
        return (actual - expected)**2

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

def loadFromFile(fileName):
    #Files we can pull from: 100testA-E,200testA-E, 500testA-E, 1000testA-E
    test1 = np.loadtxt(fileName + "0.csv")
    test2 = np.loadtxt(fileName + "1.csv")
    return (np.loadtxt(fileName + "0.csv").T,np.loadtxt(fileName + "1.csv").T,
            [[a.split("|")[0],a.split("|")[1]] for a in open(fileName + "2.txt","r").read().splitlines()],
            [[test1[a].T,test2[a]] for a in range(len(test1))])

#Minor problem with kaggle code just not working now, unsure why
#Unmodified kaggle section
#Source: https://www.kaggle.com/code/ihalil95/building-two-layer-neural-networks-from-scratch
def setParameters(X, Y, hidden_size):
    #Initializes the parameters of a new network
    #np.random.seed(3)
    input_size = X.shape[0] # number of neurons in input layer
    output_size = Y.shape[0] # number of neurons in output layer.
    W1 = np.random.randn(hidden_size, input_size)*np.sqrt(1/input_size)
    b1 = np.zeros((hidden_size, 1))
    W2 = np.random.randn(output_size, hidden_size)*np.sqrt(1/hidden_size)
    b2 = np.zeros((output_size, 1))
    return {'W1': W1, 'W2': W2, 'b1': b1, 'b2': b2}

def forwardPropagation(X, params):
    Z1 = np.dot(params['W1'], X)+params['b1']
    A1 = np.tanh(Z1)
    Z2 = np.dot(params['W2'], A1)+params['b2']
    y = sigmoid(Z2)  
    return y, {'Z1': Z1, 'Z2': Z2, 'A1': A1, 'y': y}

def cost(predict, actual):
    m = actual.shape[1]
    cost__ = -np.sum(np.multiply(np.log(predict), actual) + np.multiply((1 - actual), np.log(1 - predict)))/m
    return np.squeeze(cost__)

def backPropagation(X, Y, params, cache):
    m = X.shape[1]
    dy = cache['y'] - Y
    dW2 = (1 / m) * np.dot(dy, np.transpose(cache['A1']))
    db2 = (1 / m) * np.sum(dy, axis=1, keepdims=True)
    dZ1 = np.dot(np.transpose(params['W2']), dy) * (1-np.power(cache['A1'], 2))
    dW1 = (1 / m) * np.dot(dZ1, np.transpose(X))
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

def updateParameters(gradients, params, learning_rate = 1.2):
    W1 = params['W1'] - learning_rate * gradients['dW1']
    b1 = params['b1'] - learning_rate * gradients['db1']
    W2 = params['W2'] - learning_rate * gradients['dW2']
    b2 = params['b2'] - learning_rate * gradients['db2']
    return {'W1': W1, 'W2': W2, 'b1': b1, 'b2': b2}

def fit(X, Y, learning_rate, hidden_size, number_of_iterations = 5000):
    params = setParameters(X, Y, hidden_size)
    cost_ = []
    for j in range(number_of_iterations):
        y, cache = forwardPropagation(X, params)
        costit = cost(y, Y)
        gradients = backPropagation(X, Y, params, cache)
        params = updateParameters(gradients, params, learning_rate)
        cost_.append(costit)
    return params, cost_

#End unmodified Kaggle code

def MSEcost(predict, actual):
    print(predict.shape)
    print(actual.shape)
    m = actual.shape[1]
    #cost__ = -np.sum(np.multiply(np.log(predict), actual) + np.multiply((1 - actual), np.log(1 - predict)))/m
    cost__ = (np.sum(actual - predict))**2 / m
    return np.squeeze(cost__)

def kaggleTest(samples = 500):
    X,Y = sklearn.datasets.make_moons(n_samples = samples,noise = 0.2)
    X,Y = X.T,Y.reshape(1,Y.shape[0])
    params,cost_ = fit(X,Y,0.3,5,5000)
    plt.plot(cost_)
    plt.show()

def kaggleWordTest(filename):
    testData1 = loadFromFile(filename)
    X,Y = testData1[0],testData1[1]
    params,cost_ = fit(X,Y,0.3,7,10000)
    plt.plot(cost_)
    plt.show()

def alteredFit(X, Y, Z, learning_rate, hidden_size, number_of_iterations = 5000):
    params = setParameters(X, Y, hidden_size)
    cost_ = []
    for j in range(number_of_iterations):
        y, cache = forwardPropagation(X, params)
        costit = cost(y, Y)
        gradients = backPropagation(X, Y, params, cache)
        params = updateParameters(gradients, params, learning_rate)
        cost_.append(costit)
    return params, cost_

def stochasticFit(X,Y,LR,hidden,batches,iterations = 5000,batchDown = [500,1000]):
    #Alters Kaggle fit loop to run stochastically, see if that changes much
    #Problem: The Kaggle code is set up to take the inputs and expected outputs as separate arrays, but we need to be able to shuffle them
    params = setParameters(X,Y,hidden)
    cost_ = []
    cost_2 = []
    runBatches = batches
    elementDelta = []
    for j in range(iterations):
        #First: Shuffle the training set randomly. Zip into a list of input,output pairs, shuffle, unzip into network inputs.
        zippedList = [[X.T[a],Y.T[a]] for a in range(X.shape[1])]
        random.shuffle(zippedList)
        unzippedList = [np.array([a[0] for a in zippedList]),np.array([a[1] for a in zippedList])]
        #Next: Subsets of the list?
        #Assume for now that batches evenly divides the total sample size
        if j in batchDown:
            runBatches = int(runBatches / 2)
        batchsize = int(X.shape[1] / runBatches)
        batchBounds = [a * batchsize for a in range(runBatches + 1)]
        batchList = [[unzippedList[0][batchBounds[a]:batchBounds[a+1]].T,unzippedList[1][batchBounds[a]:batchBounds[a+1]].T] for a in range(runBatches)]
        #print(batchList)
        for x,batch in enumerate(batchList):
            #For each batch, do backprop
            #print(batch[0].shape)
            #print(batch[1].shape)
            y,cache = forwardPropagation(batch[0],params)
            costit = cost(y,batch[1])
            cost2 = MSEcost(batch[1],y)
            gradients = backPropagation(batch[0],batch[1],params,cache)
            params = updateParameters(gradients,params,LR)
            cost_.append(costit)
            cost_2.append(MSEcost(batch[1],y))
        if j%1000 == 0:
            print(j,cost_[-1],cost_2[-1],eval(Y,y),runBatches)
            
    return params,cost_,cost_2,elementDelta

def eval(expected,actual):
    correct = 0
    visN(expected,actual,10)
    for x in range(actual.shape[1]):
        eX = expected.T[x]
        aX = actual.T[x]
        exA = list(eX).index(max(list(eX)))
        acA = list(aX).index(max(list(aX)))
        if exA == acA: correct += 1
    return correct / actual.shape[1]

def visN(expected,actual,n):
    print((expected.T[:n]).T)
    print((actual.T[:n]).T)

#Would hard-coding a three-layer version of the Kaggle be useful? Or should I look at some raw outputs?
#Outputs seem to jump back and forth randomly, even though the bound on the abs seems to decrease, which suggests to me the network is not powerful enough
#to handle the problem of language
#Which probably means layers.
#Which could be hard-coded, but I'd much rather get the class form working.
    
def visMulti(sfitargs,suffix):
    params,cost,cost_2,elementDelta = stochasticFit(sfitargs[0],sfitargs[1],sfitargs[2],sfitargs[3],sfitargs[4],sfitargs[5],sfitargs[6])
    figure1 = plt.figure()
    plt.plot(cost)
    plt.savefig("cost{}.png".format(suffix))
        

def main():
    #The main function, things that run
    #testData = loadFromFile("100testA")[3]
    #testNet = Network([14,8,2])
    #print(testNet.backpropSingle(testData[0][0],testData[0][1].T))
    #kaggleTest()
    #With the code copy-pasted straight it works fine, so something I did screwed it up.
    #Hrm.
    #If we try it on our word data what happens?
    testData = loadFromFile("1000testA")
    #visMulti([testData[0],testData[1],0.0005,50,1,1000,[]],10)
    testNet = Network([14,20,18,2])
    print(testNet.backpropSet(testData[0].T[0:1].T,testData[1].T[0:1].T))
    #params,cost = fit(testData[0],testData[1],0.01,6,100000)
    #params,cost,cost_2,elementDelta = stochasticFit(testData[0],testData[1],0.0005,50,4,10000,[])
    #absdelta = [abs(a) for a in elementDelta]
    #deltadelta = [elementDelta[a] - elementDelta[a-1] for a in range(1,len(elementDelta))]
    #plt.plot(elementDelta)
    #plt.figure(1)
    #plt.plot(cost_2)
    #plt.plot(deltadelta)
    #stochasticFit(testData[0],testData[1],0.3,5,4,1)
    
main()