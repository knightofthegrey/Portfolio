#Network Program
#Version 1 - 10/23
#First attempt at building a basic neural network, from scratch using numpy arrays.

import numpy as np
import random
import sklearn.datasets
import ArsNames
import gzip
import matplotlib.pyplot as plt

class Network:
    #INITIALIZE NEW OBJECT
    def __init__(self,sizes):
        #Initialize a new Network object with random weights and biases
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]
        
    #FUNCTIONS FOR SAVING AND LOADING NETWORKS
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
    
    #OPERATING THE NETWORK
    def feedForward(self,a):
        #Runs a forward pass through the network, without preserving values for backprop
        #a is a single 1*n numpy array representing an input, and returns a 1*n numpy array representing a network result
        for b,w in zip(self.biases,self.weights):
            a = sigmoid(np.dot(w,a) + b)
        return a
    def SGD(self,trainingData,epochs,batchsize,eta,test_data = None):
        #Runs a stochastic gradient descent algorithm
        #Inputs: trainingData is input,output pairs of 1*n nparrays, epochs (int) is the number of times to run through the training data
        #Batchsize (int) is the number of training runs to do at once, I can't remember what eta is right now, and test_data is extra input,output pairs
        #This function updates the network's internal properties, but doesn't return anything
        if test_data: n_test = len(test_data)
        n = len(trainingData)
        for j in range(epochs):
            #In each epoch we shuffle the training data randomly so that we don't run the same batches over and over again
            random.shuffle(trainingData)
            mini_batches = [trainingData[k:k+batchsize] for k in range(0,n,batchsize)]
            for batch in mini_batches:
                self.update_mini_batch(batch,eta)
            if test_data: print("Epoch {}: {} / {}".format(j,self.evaluate(test_data),n_test))
            else: print("Epoch {} Complete".format(j))
    def update_mini_batch(self,mini_batch,eta):
        #Update weights/biases by applying backpropagation across the results of a single mini_batch
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x,y in mini_batch:
            #For each run, compute the estimated adjustment to the network and add it to the cumulative adjustment
            delta_nabla_b, delta_nabla_w = self.backprop(x,y)
            nabla_b = [nb + dnb for nb,dnb in zip(nabla_b,delta_nabla_b)]
            nabla_w = [nw + dnw for nw,dnw in zip(nabla_w,delta_nabla_w)]
        #Multiply the cumulative adjustment by the learning rate divided by the batch size, then subtract it from the weights/biases
        self.weights = [w-(eta / len(mini_batch)) * nw for w,nw in zip(self.weights,nabla_w)]
        self.biases = [b-(eta / len(mini_batch)) * nb for b,nb in zip(self.biases,nabla_b)]
    def backprop(self,x,y):
        #Given an input,output pair, run backprop
        #These are the suggested changes to the network on the basis of a single run
        #This should be run only from update_mini_batch, x,y are both 1xn numpy arrays
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        #Pass forward through the network, saving activations at each step
        activation = x
        activations = [x]
        zs = []
        for b,w in zip(self.biases,self.weights):
            z = np.dot(w,activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        #Once we have all of our activations for the whole network, pass backwards through the network
        delta = self.cost_derivative(activations[-1],y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta,activations[-2].transpose())
        for l in range(2,self.num_layers):
            z = zs[-1]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(),delta)
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta,activations[(-l - 1)].transpose())
        return (nabla_b,nabla_w)
    def evaluate(self,test_data):
        test_results = [(np.argmax(self.feedForward(x)),y) for x,y in testData]
    def cost_derivative(self,output_activations,y):
        return output_activations - y

def sigmoid_prime(x):
    return sigmoid(x) * (1-sigmoid(x))
            
    

class oldNetwork:
    def __init__(self,sizes):
        #Initializes a new neural net with random values and layer sizes defined by the user input
        self.weights = [np.random.randn(sizes[x],sizes[x-1]) for x in range(1,len(sizes))]
        self.biases = [np.zeros((sizes[x],1)) for x in range(1,len(sizes))]
        self.layers = len(sizes)
    
    def feedForward(self,X):
        #Runs input X forward through the network
        Z = np.dot(self.weights[0],X) + self.biases[0]
        Zlist = [Z]
        A = sigmoid(Z)
        Alist = [A]
        for x in range(1,self.layers - 1):
            Z = np.dot(self.weights[x],A) + self.biases[x]
            Zlist.append(Z)
            if x == (self.layers - 2):
                y = sigmoid(Z).T
            else:
                A = sigmoid(Z)
                Alist.append(A)
        return Zlist,Alist,y
    
    def cost(self,predicted,actual):
        #Cost function used for training the network
        m = actual.shape[0]
        #print(np.multiply(np.log(predicted),actual).shape)
        #print(np.multiply((1-actual),np.log(1-predicted)).shape)
        #cost__ = -np.sum(np.multiply(predicted,np.log(actual)) + np.multiply((1 - predicted),np.log(1-actual))) / m
        #ecost = np.multiply(predicted,np.log(actual))
        #cost__ = -np.sum(ecost) / m
        cost__ = -np.sum(np.multiply(predicted,np.log(actual)) + np.multiply(np.log((1 - actual)), 1 - predicted))/m
        #print(cost__)
        #cost__ = -(np.sum(predicted - actual)**2 / m)
        return np.squeeze(cost__)
    
    def backPropagation(self,X,Y,Z,A,y):
        #Compute deltas for weights and biases based on the output of forwardPropagation
        m = X.shape[1]
        dy = y - Y
        dW = (1/m) * np.dot(dy.T,np.transpose(A[-1]))
        dWList = list([dW])
        dB = (1/m) * np.sum(dy.T,axis = 1,keepdims = True)
        dBList = list([dB])
        #print(len(A))
        #print(self.layers - 2)
        for x in range(self.layers - 2,-1,-1):
            if x == 0: dW = (1/m) * np.dot(Z[x],np.transpose(X))
            else: dW = (1/m) * np.dot(Z[x],np.transpose(A[x-1]))
            dB = (1/m) * np.sum(Z[x],axis = 1,keepdims = True)
            dWList = list([dW]) + dWList
            dBList = list([dB]) + dBList
        return dWList,dBList
    
    def update(self,dW,dB,LR):
        #Update weights and biases of the network based on learning rate
        for w in range(len(self.weights)):
            self.weights[w] = self.weights[w] - LR * dW[w]
            self.biases[w] = self.biases[w] - LR * dB[w]
            
    def SGD(self,X,Y,expected,LR,iterations,jRate = 100):
        #Run a gradient descent on the noted stuff over the noted range
        costlist = []
        for j in range(iterations):
            Zlist,Alist,y = self.feedForward(X)
            costs = self.cost(Y,y)
            #print(costs)
            dWList,dBList = self.backPropagation(X,Y,Zlist,Alist,y)
            self.update(dWList,dBList,LR)
            costlist.append(costs)
            #print(y.shape)
            #print(expected.shape)
            accuracy = self.eval(y,expected)
            accList = []
            if j%jRate == 0:
                print("Run {}: Accuracy {:.3f}%".format(j,accuracy))
                accList.append([j,accuracy])
                print(costs)
        return costlist
                
    def eval(self,actR,expR):
        #Find the percentage correct of the result
        corr = 0
        #print(actR)
        #print(expR)
        for x in range(actR.shape[0]):
            #xr = list(actR[x])
            #dr = xr.index(max(xr))
            #cr = list(expR[x]).index(max(list(expR[x])))
            dr = round(float(actR[x]))
            cr = expR[x]
            if dr == cr: corr += 1
        return 100 * corr / actR.shape[0]
        
                
            
        

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

def setParameters(X, Y, hidden_size):
    input_size = X.shape[0] # number of neurons in input layer
    output_size = Y.shape[1] # number of neurons in output layer.
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
    return y, {'Z1': Z1, 'Z2': Z2, 'A1': A1, 'y': y.T}

def cost(predict, actual):
    m = actual.shape[1]
    cost__ = -np.sum(np.multiply(predict.T,np.log(actual)) + np.multiply(np.log((1 - actual)), 1 - predict.T))/m
    return np.squeeze(cost__)

def backPropagation(X, Y, params, cache):
    m = X.shape[1]
    dy = (cache['y'] - Y)**2
    dW2 = (1 / m) * np.dot(dy.T, np.transpose(cache['A1']))
    db2 = (1 / m) * np.sum(dy.T, axis=1, keepdims=True)
    dZ1 = np.dot(np.transpose(params['W2']), dy.T) * (1-np.power(cache['A1'], 2))
    dW1 = (1 / m) * np.dot(dZ1, np.transpose(X))
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

def updateParameters(gradients, params, learning_rate = 1.2):
    W1 = params['W1'] - learning_rate * gradients['dW1']
    b1 = params['b1'] - learning_rate * gradients['db1']
    W2 = params['W2'] - learning_rate * gradients['dW2']
    b2 = params['b2'] - learning_rate * gradients['db2']
    return {'W1': W1, 'W2': W2, 'b1': b1, 'b2': b2}

def fit(X, Y, Z, learning_rate, hidden_size, wordlist = [], number_of_iterations = 5000,endTest = []):
    params = setParameters(X, Y, hidden_size)
    cost_ = []
    if wordlist: ratelist = {a[0]:[0,0] for a in wordlist}
    else: ratelist = {}
    for j in range(number_of_iterations):
        print(j)
        y, cache = forwardPropagation(X, params)
        costit = cost(Y,y)
        gradients = backPropagation(X, Y, params, cache)
        params = updateParameters(gradients, params, learning_rate)
        cost_.append(costit)
        #vis([params["W1"],params["W2"]])
        #accdisplay,latestC,latestTotal,successList,failList = acc(y,Z,j+1,wordlist = wordlist)
        if wordlist:
            pass
            #for a in successList:
                #ratelist[a[0]][0] += 1
                #ratelist[a[0]][1] += 1
            #for a in failList:
                #ratelist[a[0]][1] += 1
        #print(costit)
        if j%100 == 99:
            #print(accdisplay)
            #failList.sort()
            #print(failList)
            pass
    if endTest:
        y,cache = forwardPropagation(endTest[0],params)
        accDisplay,latestC,latestTotal,successList,failList = acc(y,endTest[1],0,wordlist = endTest[2])
        print("End test on new parameters:")
        print(accDisplay)
        print("Failed on: {}".format(failList))
    return cost_,ratelist

def acc(actR,expR,iteration,wordlist = []):
    #Find the accuracy of the result
    corr = 0
    sl = []
    fl = []
    for x in range(actR.T.shape[0]):
        xr = list(actR.T[x])
        #dr = xr.index(max(xr)) For numerical answers
        #dr = xr.index(max(xr))
        #cr = list(expR[x]).index(max(list(expR[x])))
        #if dr == cr:
            #corr += 1
            #if wordlist: sl.append(wordlist[x])
        #else:
            #if wordlist: fl.append(wordlist[x])
    return "{} Found {} of {} ({:.3f}%)".format(iteration,corr,actR.T.shape[0],100 * corr / actR.T.shape[0]),corr,actR.T.shape[0],sl,fl

def vis(warray):
    strlists = [["[ {} ]".format(", ".join(["{:.3f}".format(b) for b in row])) for row in a] for a in warray]
    strdims = [[len(a[0]),len(a)] for a in strlists]
    maxh = max([a[1] for a in strdims])
    for a in range(maxh):
        row = []
        for b in range(len(strdims)):
            if a in range(len(strlists[b])): row.append(strlists[b][a])
            else: row.append(" " * strdims[b][0])
        print(" | ".join(row))

def gendata(lendata):
    binthing = [["0","0","0"],["0","0","1"],["0","1","0"],["0","1","1"],["1","0","0"],["1","0","1"],["1","1","0"],["1","1","1"]]
    decthing = [[0 if a != b else 1 for a in range(8)] for b in range(8)]
    outdata = []
    for a in range(lendata):
        pick = random.randint(0,7)
        outdata.append([np.array(binthing[pick],dtype = float),np.array(decthing[pick],dtype = float)])
    return [np.array([a[0] for a in outdata]),np.array([a[1] for a in outdata]),np.array([list(a[1]).index(1.) for a in outdata])]

def genMoreData(lendata,bindig = 3):
    #Generates an np array of binary numbers, a np array of 1 at the number/0 elsewhere for network outputs, and the decimal representation for checking
    #Used to generate training data for a test neural network
    outData = []
    for a in range(lendata):
        pick = random.randint(0,2**bindig - 1)
        #print(padBin(pick,bindig))
        outData.append([np.array(padBin(pick,bindig),dtype = float),np.array([0 if a != pick else 1 for a in range(2**bindig)],dtype = float),pick])
    #for a in outData: print(a[0])
    return [np.array([a[0] for a in outData]),np.array([a[1] for a in outData]),[a[1] for a in outData],[[np.array(a[0]),np.array(a[1])] for a in outData]]

def padBin(indig,padsize):
    #Returns a binary digit padded to the correct length
    outdig = list("{:0>{}}".format(str(bin(indig))[2:],padsize))
    return outdig

'''
realDict = [a[0].lower() for a in ArsNames.openDictionary()]
realDict.sort()
realDict = [realDict[0]] + [realDict[a] for a in range(1,len(realDict)) if realDict[a] != realDict[a-1]]
saveset = open("qwords.txt","w")
for line in realDict: saveset.write("{}\n".format(line))
saveset.close()
'''

#Problem: How do we turn words into inputs?

def wordToInput(inWord,padlen = 14):
    padWord = "{: ^{}}".format(inWord,padlen)
    templist = [(ArsNames.alph.index(a) + 1) if a in ArsNames.alph else 0 for a in padWord]
    return np.array(templist)

def genWordData(samples):
    #Generate real and fake words from our real/fake word databases for network testing
    fakelist = [[random.choice([b for b in open("fakeWords.txt","r").read().splitlines() if len(b) <= 14]),0] for a in range(samples)]
    reallist = [[random.choice([b for b in open("qwords.txt","r").read().splitlines() if len(b) <= 14]),1] for a in range(samples)]
    wordlist = fakelist+reallist
    random.shuffle(wordlist)
    return (np.array([wordToInput(a[0]) for a in wordlist]),np.array([np.array([1.0 if b == a[1] else 0.0 for b in range(2)]) for a in wordlist]),wordlist)

def saveData(samples,fileName):
    testData = genWordData(int(samples/2))
    inputs = testData[0]
    outputs = testData[1]
    raw = testData[2]
    np.savetxt(fileName + "0.csv",inputs)
    np.savetxt(fileName + "1.csv",outputs)
    rawfile = open(fileName + "2.txt","w")
    rawfile.write("\n".join(["{}|{}".format(a[0],a[1]) for a in raw]))
    rawfile.close()

def loadFromFile(fileName):
    #Files we can pull from: 100testA-J,200testA-J, 500testA-J, 1000testA-J, 2000testA-J
    test1 = np.loadtxt(fileName + "0.csv")
    test2 = np.loadtxt(fileName + "1.csv")
    return (np.loadtxt(fileName + "0.csv"),np.loadtxt(fileName + "1.csv"),
            [[a.split("|")[0],a.split("|")[1]] for a in open(fileName + "2.txt","r").read().splitlines()],
            [[test1[a],test2[a]] for a in range(len(test1))])

def visRates(rates):
    keylist = [[a,rates[a]] for a in rates]
    keylist.sort(key = lambda a:a[1][0] / a[1][1])
    for entry in keylist: print(entry)
    
def makeFakeWords(n):
    outData = open("fakeWords.txt","a")
    for a in range(n):
        wordlen = random.choice(range(4,14))
        word = "".join([random.choice(ArsNames.alph) for b in range(wordlen)])
        outData.write(word + "\n")
    outData.close()

#makeFakeWords(20000)

#Thought: Problem: Random word sets every time take a long time to start and make the data not very controlled


#testSet = loadFromFile("100testB")
#X = testSet[0].T
#Y = testSet[1]
#Raw = testSet[2]
'''
testSet = sklearn.datasets.make_moons(n_samples = 100,noise = 0.2)
X = testSet[0]
X1 = np.split(X,2)[0].T
X2 = np.split(X,2)[1].T
Y = testSet[1]
Y1 = np.split(Y,2)[0]
Y1 = Y1.reshape(1,Y1.shape[0])
Y2 = np.split(Y,2)[1]
Y2 = Y2.reshape(1,Y2.shape[0])
'''
#testNet = oldNetwork([2,5,1])
#costs = testNet.SGD(X1,Y1,Y1,0.3,5000,jRate = 100)
#costs,rates = fit(X1,Y1,Y1,0.03,5,number_of_iterations = 500)
#plt.plot(costs)
#plt.show()




#cost_,rates = fit(X,Y,Y,0.5,13,Raw,15000,[X1,Y1,Raw1])
#visRates(rates)


#testNet = Network([3,3,8])
#testNet.SGD(X,Y,Z,0.1,10000,jRate = 100)

#testNet = oldNetwork([14,5,2])
#testNet.SGD(X,Y,Y,0.3,10000,jRate = 100)




