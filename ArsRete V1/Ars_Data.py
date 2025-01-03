#ARS DATA
#This file generates and manages datasets to train and run Ars_Network on

import numpy as np
import matplotlib as plt
from torch.utils.data import Dataset
import random
import json
import torch

#The main datasets we use here are:
#-Words: Determining if a word is a member of a set of words or not
#-Points: Determining which, if any, n-dimensional spheres an n-dimensional point is inside
#-Images: Classifying images of typed characters

#__1: WORD DATA__

def genFakeWords(fake_path = "fakewords.txt"):
    alphabet = [chr(a) for a in range(97,123)]
    lendistro = [0,26,121,1135,4347,8497,15066,20552,26434,28833,27924,23773,18837,13877,9151,5585,3223,1738,815,417,194,81,40,16,5]
    with open(fake_path,"w") as file:
        for a in range(210000):
            word_len = random.choices(range(4,16),weights = lendistro[4:16])[0]
            word = ""
            for b in range(word_len):
                word += random.choice(alphabet)
            file.write(word + "\n")
        file.close()

def readAllWords(real_path = "basewords.txt",fake_path = "fakewords.txt"):
    #Reads all words from basewords.txt
    with open(real_path,"r") as file:
        temp_data = file.read().splitlines()
        real_data = [a for a in temp_data if a.islower()]
        file.close()
    with open(fake_path,"r") as file:
        temp_data = file.read().splitlines()
        fake_data = [a for a in temp_data if a.islower()]
        file.close()
    return real_data,fake_data

def wordsToJson(real_data,fake_data):
    words_dict = {}
    for a in real_data:
        words_dict[a] = [0,1]
    for a in fake_data:
        if a not in words_dict.keys():
            words_dict[a] = [1,0]
    json.dump(words_dict,open("wordsdict.json","w"))

def scrambledWords(path = "wordsdict.json"):
    with open(path,"r") as file:
        temp_data = json.load(file)
    words_list = [[a,temp_data[a]] for a in temp_data.keys()]
    random.shuffle(words_list)
    return words_list

def padWord(in_word,pad_len):
    #Pads word to a given length in all possible permutations for data augmentation purposes
    spaces = pad_len - len(in_word)
    if spaces <= 0:
        return [in_word]
    else:
        return [" "*a + in_word + " "*(spaces - a) for a in range(spaces + 1)]

def makeWordset(out_path,min_len,max_len,pad_len,set_size):
    temp_data = scrambledWords()
    sub_data = [a for a in temp_data if len(a[0]) in range(min_len,max_len+1)]
    word_data = [[[b,a[1]] for b in padWord(a[0],pad_len)] for a in sub_data]
    flattened_words = [a for b in word_data for a in b]
    random.shuffle(flattened_words)
    sub_dict = {a[0]:a[1] for a in flattened_words[:set_size]}
    with open(out_path,"w") as file:
        json.dump(sub_dict,file)
    file.close()

def makeReals(out_path,min_len,max_len,pad_len,set_size):
    temp_data = scrambledWords()
    sub_data = [a for a in temp_data if len(a[0]) in range(min_len,max_len+1) and a[1] == [0,1]]
    word_data = [[[b,1] for b in padWord(a[0],pad_len)] for a in sub_data]
    flattened_words = [a for b in word_data for a in b]
    random.shuffle(flattened_words)
    sub_dict = {a[0]:a[1] for a in flattened_words[:set_size]}
    with open(out_path,"w") as file:
        json.dump(sub_dict,file)
    file.close()

def fastWordset(in_path,out_path):
    temp_data = json.load(open(in_path,"r"))
    max_len = len("{:b}".format(27**(len(list(temp_data.keys())[0]) + 1)))
    out_data = {}
    for a in temp_data:
        decimal = sum([27**b * (ord(a[b]) - 96) if a[b] != " " else 0 for b in range(len(a))])
        bin_rep = "{:b}".format(decimal)
        bin_rep = "0" * (max_len - len(bin_rep)) + bin_rep
        out_data[bin_rep] = temp_data[a]
    with open(out_path,"w") as file:
        json.dump(out_data,file)
    file.close()

def loadWordset(in_path,mode = 0):
    with open(in_path,"r") as file:
        words = json.load(file)
    file.close()
    datalist = []
    labellist = []
    for a in words:
        if mode == 0: data = torch.Tensor([(ord(b)-96)/27 if b != " " else 0 for b in a])
        elif mode == 1:
            max_len = len("{:b}".format(27**(len(a) + 1)))
            base_27 = [ord(b) - 96 if b != " " else 0 for b in a]
            decimal_word = sum([27**b * base_27[b] for b in range(len(base_27))])
            binary_word = [float(b) for b in "{:b}".format(decimal_word)]
            pad = [0.0] * (max_len - len(binary_word))
            data = torch.Tensor(pad + binary_word)
        datalist.append(data)
        label = torch.Tensor(words[a])
        labellist.append(label)
    return datalist,labellist

def loadRealset(in_path):
    #Loads dataset prepped for GAN with fastWordset
    #Input format is a binary string and the number 1
    with open(in_path,"r") as file:
        words = json.load(file)
    file.close()
    datalist = []
    labellist = []
    for a in words:
        datalist.append(torch.Tensor([float(b) for b in a]))
        labellist.append(torch.Tensor([float(1)]))
    return datalist,labellist

def getWordData(fileName):
    #This function reads data from lists of real and fake words, given a string representing the file prefix.
    #Each word set is three files: prefix0.csv is the inputs, prefix1.csv is the outputs, and prefix2.txt is the actual word.
    #At time of writing the filename prefixes we can call on are 100testA-E, 500testA-E, and 1000testA-E, each containing half real and half fake words in a random order.
    #It returns: a list of input,output pairs structured to be taken as inputs by the Network object, and a list of words just in case we need them.
    data0 = np.loadtxt(fileName + "0.csv")
    data1 = np.loadtxt(fileName + "1.csv")
    data2 = [a.split("|") for a in open(fileName + "2.txt","r").read().splitlines()]
    return [(np.array([a[0]]).T,np.array([a[1]]).T) for a in zip(data0,data1)],[a[0] for a in data2]

#__2: IMAGE DATA__

#NOTHING HERE AT PRESENT, CONTROLLED BY ARS_IMAGE FILE

#__3: CIRCLE DATA__

def circleData(numPoints,numDims,numCircles = 1):
    #This function creates a set of random n-dimensional circles, then defines points as inside or outside each circle
    #NOTE: The expected outputs from this function are whether the point is inside of each circle, with a 1 at index 0 if it's outside all circles,
    #which the Network's accuracy code is not set up to check for at present. Everything else should work, though.
    #Inputs:
    '''
    num_points: int, number of points to create
    num_dims: int, number of dimensions
    num_circles: int, number of n-dimensional circles
    '''
    #Outputs: List of np array pairs for inputs and outputs. Input size is the number of dimensions, output size is the number of circles.
    #__1: Generate the center point of each circle, anywhere from -10 to 10 in each dimension, and the radius, anywhere from 2 to 5.__
    #The specific numbers here are arbitrary and could really be anything.
    circleCenters = [[np.random.uniform(-10.0,10.0) for a in range(numDims)] for b in range(numCircles)]
    circleR = [np.random.uniform(2.0,5.0) for b in range(numCircles)]
    #__2: Generate points in the graph and check if they're in each circle.__
    pointsList = []
    for a in range(numPoints):
        #Generate random point, find distance, determine if that makes it inside or outside of circle
        #For these purposes on the circle is inside the circle
        newPoint = [np.random.uniform(-15.0,15.0) for a in range(numDims)]
        outputList = [0] #We're presently reserving term 0 to be 0 if the point is in one or more circles, and 1 if it isn't
        for b in range(numCircles):
            #This expression gives us 1 if the point is in circle b and 0 if it isn't.
            outputList.append(int(np.sqrt(sum([(circleCenters[b][n] - newPoint[n]) ** 2 for n in range(numDims)])) <= circleR[b]))
        if sum(outputList) == 0: outputList[0] = 1
        pointsList.append((np.array(newPoint),np.array(outputList)))
    #__3: Return the input/output pairs, and the list of circle parameters.
    return pointsList,(circleCenters,circleR)

