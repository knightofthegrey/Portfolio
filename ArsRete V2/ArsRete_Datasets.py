#ArsRete Datasets

#This module contains the Wordset class, and the required functions to process word datasets for the ArsRete machine learning program.

'''
Version History
2/28/24: Initial version. Assembled from ad-hoc functions written elsewhere.
'''

#IMPORTS#

#Some objects in this module are pytorch tensors to help them interface better with the networks later.
#Additionally Worset inherits from pytorch's Dataset object.
import torch
#The actual word datasets for this program are stored as json files.
import json
#Used ceil once
import math
#For getting random subsets from a dataset
import random
import re
from metaphone import doublemetaphone as dm

from time import time

#__GLOBAL VARIABLES__
#A letter is represented by a five-bit number in our encoding scheme. This is hard-coded to assume five-bit encoding, but could theoretically be changed.
#since the nature of the neural network means a denser encoding (more meaningful values) is easier to learn than a sparser encoding.
#For the encoding scheme we give an encode dictionary and a decode dictionary.
legal_chars = " abcdefghijklmnopqrstuvwxyz-\'012"
encodings = [[float(b) for b in format(a,"05b")] for a in range(len(legal_chars))] #Binary numbers 0-31 as lists of five floats
en_d = {legal_chars[a]:encodings[a] for a in range(len(legal_chars))} #Character->binary encoding
dec_d = {a:legal_chars[a] for a in range(len(legal_chars))} #Decimal index->character (the Decode function will get from binary encoding back to decimal index on its own)

#Vowel-consonant definitions:
v_g = "aeiouy" #Vowels. For simplicity "y" is just sorted here for now.
s_g = " -\'" #Spacers.
c_g = "bcdfghjklmnpqrstvwxz" #Consonants
p_g = "012" #Extraneous values


#__WORDSET CLASS__

class Wordset(torch.utils.data.Dataset):
    #Pytorch dataset for use with ArsRete_Network
    #Additionally Wordset builds reference items to help compare datasets when initialized.
    #Major changes 
    def __init__(self,in_paths,size_1 = False,smooth = False):
        #Input: 
        '''
        in_paths is a list of string paths to .json files containing a dictionary of word:label
        Input words should be of uniform width when given to Wordset, but there are functions later on to fix that.
        '''
        #Initializer
        super(Wordset,self).__init__()
        #These two are built up as a list of floats
        self.data = []
        self.labels = []
        #This is a list of strings for comparison to other functions
        self.big_set = []
        
        #Iterate through in_paths, and build up the three lists from the values in the json file
        for path in in_paths:
            temp_dict = json.load(open(path,"r")) #Open the file.
            for word in temp_dict:
                #Check to see if the characters in the word are legal words
                if wC(word):
                    #If so, append to big_set, encode in data and labels.
                    self.big_set.append(word)
                    self.data.append(Encode(word))
                    if size_1: self.labels.append([float(int(temp_dict[word] in [[0,1],1]))]) #Convert two-int labels back to single-float labels, just in case.
                    elif temp_dict[word] in [0,1]: self.labels.append([float(int(temp_dict[word] == 0)),float(int(temp_dict[word] == 1))]) #Convert single-number labels to two-float labels
                    else: self.labels.append([float(c) for c in temp_dict[word]]) #Otherwise, just make sure the labels are floats
        
        #Once we have our datasets:
        self.dl = len(self.data[0])
        self.data = torch.tensor(self.data)
        self.labels = torch.tensor(self.labels)
        if smooth: self.smooth()
        self.comp = [LVector(self.big_set),CVVector(self.big_set),Markov(self.big_set,1),Markov(self.big_set,2)]
    
    def weights(self,comp_v):
        #Compares LVector, CVVector values of self to a gibberish dataset, and run Markov score on own data
        #This gives us weights that let us scale our network outputs' results to a range of 0-1
        LW = C_Vector(self.comp[0],comp_v[0])
        CVW = C_Vector(self.comp[1],comp_v[1])
        MS1 = MarkovScore(self.big_set,self.comp[2],1)
        MS2 = MarkovScore(self.big_set,self.comp[3],2)
        return [LW,CVW,MS1,MS2]
    
    def smooth(self):
        #Double-sided label smoothing (sets labels of 0.0 to 0.1 and 1.0 to 0.9)
        self.labels = (0.8*self.labels) + 0.1
        
    def __len__(self):
        #Necessary component for torch.Dataset
        return self.data.size()[0]
    
    def __getitem__(self,index):
        #Necessary component for torch.Dataset
        return self.data[index],self.labels[index]

#__Single Word Operations__
            
def wC(w):
    #wordCheck
    #Quick check to see if all characters in w are in the correct character set space for the program
    return sum([a in legal_chars for a in w]) == len(w)

def Encode(w):
    #Encode a word, character-by-character, using the en_d dictionary
    encoding = []
    for a in w:
        encoding += en_d[a]
    return encoding

def Decode(w):
    #Decode binary values outputted by a network back to a word
    #The inputted word can be either a list of floats or a tensor. If it's a tensor we need to convert it to a list before proceeding.
    if type(w) == torch.tensor: w = w.tolist()
    #Int and round
    try: w = [int(round(a)) for a in w]
    except: w = [int(round(a.item())) for a in w]
    #Chunk in 5-character segments and convert to decimal
    w = [int("".join([str(a) for a in w[5*b:5*(b+1)]]),2) for b in range(len(w) // 5)]
    #Get character at index and return
    return "".join([dec_d[a] for a in w])

def Pad(word,sz,al = "<"):
    #Pads a word with spaces to the indicated size
    #al is for left, right, center alignment within the output string. It will interpret l,r,c as left, right, center, or
    #the literal <>^, but if given anything other than those six characters it'll default to < (left)
    if al in "lrc": al = {"l":"<","r":">","c":"^"}
    if al not in "<>^": al = "<"
    return "{: {}{}}".format(word,sz,al)

def Augment(word,sz):
    #Returns a list of a word padded to all possible positions
    #e.g. Augment("hot",5) would return ["hot  "," hot ","  hot"]
    #In theory, though this is under-tested, this could behave similarly to data augmentation on an image
    if len(word) >= sz: return [word]
    else: return [" "*a + word + " "*(sz-len(word)-a) for a in range(sz-len(word) + 1)]

#__Wordset Comparative Data Functions__

def Compare(in_set,ref_set,gib_set):
    #Get quick comparison numbers for in_set v. ref_set
    weights = ref_set.weights(gib_set.comp)
    lc = C_Vector(LVector(in_set),ref_set.comp[0],weights[0])
    #cvc = C_Vector(CVVector(in_set),ref_set.comp[1],weights[1])
    m1c = MarkovScore(in_set,ref_set.comp[2],1,weights[2])
    m2c = MarkovScore(in_set,ref_set.comp[3],2,weights[3])
    return [lc,m1c,m2c]

def LVector(in_set):
    #Gets a tensor containing the float values of the relative frequencies of all letters in the sample
    #Input: List of strings
    #Output: 1x32 tensor of floats
    #Used for comparison, but the frequency of the last three (extraneous punctuation) should approach 0.
    temp_v = [1 for a in range(len(legal_chars))] #Initialize the values of all letters to 0
    for a in in_set: #For each word, tick up each letter found
        for b in a.strip(): #We strip a here, so we count internal spaces but not external spaces.
            temp_v[legal_chars.find(b)] += 1
    
    return torch.tensor([a/sum(temp_v) for a in temp_v]) #Turn into a float of instances/total, and return

def CVVector(in_set):
    #Gets a tensor containing the float values of the relative frequencies of consonant-vowel patterns in the sample
    #This will be of length 2**2n 
    v_patterns = [0]*(4**len(in_set[0]))
    for a in in_set:
        wl = []
        for b in a:
            if b in v_g: wl.append(0)
            elif b in c_g: wl.append(1)
            elif b in s_g: wl.append(2)
            elif b in p_g: wl.append(3)
        wn = sum([4**b * wl[b] for b in range(len(wl))]) #Interpret the consonant-vowel pattern as a quaternary number
        v_patterns[wn] += 1
    return torch.tensor([a/sum(v_patterns) for a in v_patterns])

def C_Vector(v_1,v_2,weight = 1):
    #Gets the mean squared distance between two LVectors
    #Raw values vary somewhat; typically we divide the LVector of our reference set by the LVector of random gibberish (using the Weight value),
    #which gives a value from 0 (close to the reference set) to 1 (close to gibberish), or >1 if it's worse than gibberish, but that doesn't
    #usually happen outside the beginning of a run
    #This line subtracts v2 from v1 element-wise, squares each element, sums them, and divides by the weight.
    if weight != 0: return torch.sum(torch.square(v_1 - v_2)).item()/weight
    else: return 0

def Markov(in_set,window = 1):
    #Gets a 2-d dictionary {a:{b:1}} containing the float values of the frequency of finding character b after string a
    #Input: List of strings
    #Output: dict of str:dict(chr:float)
    #If we're looking at a window wider than 1 we need to assemble a list of all possible groups of that many characters
    groups = legal_chars
    for a in range(window - 1):
        newgroups = []
        for b in groups:
            for c in legal_chars:
                newgroups.append(b+c)
        groups = newgroups
    statechain = {a:{b:0 for b in legal_chars} for a in groups}
    for a in in_set:
        t = a.strip() #Strip a, so we count internal spaces but not external spaces
        for b in range(len(t) - window):
            #Since we compare letters to the following letter we stop one short of the end.
            statechain[t[b:b+window]][t[b+window]] += 1
    #Once we have our statechain we need to convert from integer quantities to frequencies
    for a in groups:
        tot_a = sum([statechain[a][b] for b in legal_chars])
        for b in legal_chars:
            if tot_a != 0: #On the off chance we have no instances of a letter we don't want to div/0 by accident
                statechain[a][b] = statechain[a][b] / tot_a
    
    return statechain

def MarkovScore(in_set,m_pattern,window = 1,weight = 1):
    #Get an average probability score for a group of words using a markov pattern produced by the Markov function
    #This scales to 0-1 by weighting it with a set's m-score against itself at that window, but it's backwards from the v-metric (0 bad, 1 good)
    tot_score = 0
    count = 0
    for a in in_set:
        t_score = 0
        t = a.strip()
        if len(t) > window:
            for b in range(len(t) - window):
                t_score += m_pattern[t[b:b+window]][t[b+window]]
            tot_score += (t_score / (len(t) - window)) #We want an average probability, so long words don't end up with higher numbers
            count += 1
    try:
        return (tot_score / count) / weight
    except:
        return 0

def WMScore(in_set,ref_set,window):
    #Wrapper to call Markov and MarkovScore for a weighted score
    p = Markov(ref_set,window)
    weight = MarkovScore(ref_set,p,window)
    return MarkovScore(in_set,p,window,weight)

def soundex(word):
    #Phonetic pronunciation encoding
    #Experiment with comparing encoding to encodings in the reference set
    #Soundex code rules: first letter unchanged, replace all other letters with their number code, remove yhw (code 8), remove duplicates, remove vowels (code 7), remove first number if it's the duplicate of the first letter's number
    sxn = {"b": "1", "f": "1", "p": "1", "v": "1", "c": "2", "g": "2", "j": "2", "k": "2", "q": "2", "s": "2", "x": "2", "z": "2", "d": "3", "t": "3", "l": "4", "m": "5", "n": "5", "r": "6", "a": "7", "e": "7", "i": "7", "o": "7", "u": "7", "y": "8", "h": "8", "w": "8"}
    intword = [word.strip()[0]] + [sxn[a] for a in word.strip()[1:] if a in sxn]
    intword = "".join(intword)
    intword = intword.replace("8","")
    for a in range(7):
        duptern = str(a) + str(a) + "+"
        while re.search(duptern,intword):
            intword = re.sub(duptern,str(a),intword)
    intword = intword.replace("7","")
    if len(intword) > 1:
        if intword[1] == sxn[intword[0]]: intword = intword[0] + intword[2:]
    return intword

def uniqueSoundex(wordset):
    return list(set([soundex(a) for a in wordset]))

def uniqueM(wordset):
    return list(set([a for b in [dm(c) for c in wordset] for a in b]))
    
def vcp(word):
    v = "aeiou"
    c = "bcdfghjklmnpqrstvwxyz"
    out = ""
    for a in word.strip():
        if a in v: out += "v"
        elif a in c: out += "c"
        elif a == " ": out += "s"
        else: out += "p"
    return out

def vcpPrev(words):
    vcl = {}
    for a in words:
        if vcp(a) in vcl: vcl[vcp(a)] += 1
        else: vcl[vcp(a)] = 1
    return {a:vcl[a]/len(words) for a in vcl}

def proscore(word,mpattern,vpattern,vpthresh = 0.005):
    word_m = dm(word)
    score = 0
    if word_m[0] in mpattern and word_m[0] != "": score += 1
    if word_m[1] in mpattern and word_m[1] != "": score += 1
    word_v = vcp(word)
    if word_v in vpattern:
        if vpattern[word_v] > vpthresh: score += 2
        else: score += 1
    
    return score

def setpro(test_words,ref_words):
    test_m = uniqueM(ref_words)
    test_v = vcpPrev(ref_words)
    return sum([proscore(a,test_m,test_v) for a in test_words]) / len(test_words)

def prodistro(test_words,ref_words):
    test_m = uniqueM(ref_words)
    test_v = vcpPrev(ref_words)
    out_list = [0,0,0,0,0]
    tot = 0
    for a in test_words:
        out_list[proscore(a,test_m,test_v)] += 1
        tot += 1
    #Returns are abs-freq and cum-freq
    absf = [a/tot for a in out_list]
    cumf = [sum(out_list[a:len(out_list)])/tot for a in range(len(out_list))]
    return absf,cumf

#__MAKING NEW WORD DATASET FILES FROM TXT FILES__

def MakeSet(out_filename,sizes,in_file,label = 1,pad_mode = "<"):
    #First: Read a file
    if in_file.split(".")[-1] == "txt": words = [a.strip() for a in open(in_file,"r").read().splitlines()]
    elif in_file.split(".")[-1] == "json": words = [a.strip() for a in json.load(open(in_file,"r"))]
    #Here we turn either a .txt file of line-separated words or a .json file of word:label pairs into a list of words
    else: 
        print("Can't read that file")
        return #If we don't have a file we can read, end here.
    
    #Then:
    out_dict = {}
    for a in words:
        if len(a) in range(sizes[0],sizes[1]+1):
            #If it's the correct size:
            if pad_mode == "a":
                #If we're augmenting words rather than just padding:
                for b in Augment(a,sizes[2]):
                    out_dict[b] = label
            else:
                out_dict[Pad(a,sizes[2],pad_mode)] = label
    
    #Finally, dump dict to a .json
    json.dump(out_dict,open(out_filename,"w"))

def MakeSubset(out_filename,in_file,size):
    #Gets a random subset of a word set
    #The word set should already be a .json and padded to uniform length using MakeSet
    word_dict = json.load(open(in_file,"r"))
    
    word_list = [a for a in word_dict]
    random.shuffle(word_list)
    
    out_dict = {}
    
    for a in word_list[:size]:
        out_dict[a] = word_dict[a]
    
    json.dump(out_dict,open(out_filename,"w"))