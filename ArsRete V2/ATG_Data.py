#ATG_Data
#Ars_Torch_GAN project implementing a generative adversarial network to generate words that sound like the input sample
#ATG_Main is the main file for this project, and contains more introductory information
#This file contains functions for processing lists of words to create inputs for the network, and to do some processing of outputs

#__CHANGELOG__#
#1/5/2024: 0.0.1: Initial version planned and written

#__IMPORTS__#
import math
import json
import random
from quicktions import Fraction as f
import matplotlib.pyplot as plt
#from Ars_TorchGen import unifiedWordset as uW

#__WORDS DATASET FUNCTIONS__#

def setTruncate(in_file,out_file,trunc_size):
    #Make a subset of a word dataset
    word_list = json.load(open(in_file,"r"))
    out_dict = {}
    subset = random.choices(list(word_list.keys()),k = trunc_size)
    for a in subset:
        out_dict[a] = word_list[a]
    json.dump(out_dict,open(out_file,"w"))

def makeGibberish(out_file,sizes,random_size = 1000,pad_mode = "l"):
    #Makes gibberish of the given sizes for use with comparing random to reference distributions, writes to a new .json
    #out_file: str, relative path
    #sizes: min, max, pad
    #pad_mode: "l","r","c" will pad as indicated, "a" will augment
    out_dict = {}
    for a in range(random_size):
        wl = random.choice(range(sizes[0],sizes[1]+1))
        word = "".join([random.choice("abcdefghijklmnopqrstuvwxyz") for b in range(wl)])
        if pad_mode == "a":
            for b in augmentWord(word,sizes[2]):
                out_dict[b] = [1,0]
        else:
            out_dict[pad(word,sizes[2],pad_mode)] = [1,0]
    
    json.dump(out_dict,open(out_file,"w"))

def makeWordset(out_file,sizes,in_file = "words_dictionary.json",pad_mode = "l"):
    #Reads a .json file of words, gets all words of sizes[0] to sizes[1], pads them to sizes[2] according to pad_mode, writes to a new .json
    #out_file and in_file: str, relative paths
    #sizes: list of ints, min, max, pad
    #pad_mode: "l","r","c" will call pad with that argument, "a" will call augment
    #No return, makes a file
    word_list = json.load(open(in_file,"r"))
    out_dict = {}
    for a in word_list:
        if len(a) in range(sizes[0],sizes[1]+1):
            if pad_mode == "a":
                for b in augmentWord(a,sizes[2]):
                    out_dict[b] = word_list[a]
            else:
                out_dict[pad(a,sizes[2],pad_mode)] = word_list[a]
    json.dump(out_dict,open(out_file,"w"))
    
def pad(word,padsize,lrc = "l"):
    #This function pads a word to length padsize, with the word aligned left, right, or center
    if lrc == "l":
        return word + " "*(padsize - len(word))
    elif lrc == "r":
        return " "*(padsize - len(word)) + word
    else:
        return " "*((padsize - len(word))//2) + word + " "*(math.ceil((padsize - len(word)) / 2))

def augmentWord(word,padsize):
    #This function pads a word to length padsize for all possible distributions of spaces (e.g. "hot" augmented to 5 could be "hot  ", " hot ", or "  hot")
    #This is useful for creating strings of uniform length from strings of non-uniform length, and also has the effect of doing some data augmentation
    if len(word) >= padsize:
        return [word]
    else:
        outwords = []
        margin = padsize-len(word)
        for a in range(margin + 1):
            tempword = " "*a + word + " "*(margin-a)
            outwords.append(tempword)
        return outwords

#__PROJECT UTILITY FUNCTIONS__#
#This is a catch-all section for functions that may be helpful at other places in the project

def decode(bin_list):
    #Decodes a list of binary values to a word so we can see what the generator is making.
    #Input: bin_list is either a tensor of floats outputted by the generator, or a list of tensors of floats contained in an ArsWordset
    #Output: String.
    #Step 1: Convert binary to decimal. Try/except here controls for different input formats.
    #Data from the generator comes out in the range 0 to 1, round() converts it to a 0 or a 1, which we can then stick together into a string
    #and interpret as a base-2 int
    try: decimal = int("".join([str(round(a)) for a in bin_list.tolist()[0]]),2) #Runs if given tensor of floats (generator output)
    except: decimal = int("".join([str(round(a.item())) for a in bin_list]),2) #Runs if given list of single-element tensors of floats (dataset data)
    #Step 2: Convert decimal to list of numbers from 0-26 (base-27 integer)
    digits = []
    while decimal:
        digits.append(int(decimal % 27))
        decimal = decimal // 27
    #Step 3: Convert base 27 list to string using chr (chr(97) = "a", chr(98) = "b", and so on)
    #This system interprets 0 as a space
    return "".join([chr(96+a) if a != 0 else " " for a in digits])

def binLen(word_len,chr_set = 27):
    #Gets the length of binary values needed to represent words of a given length using encode/decode v1 (word -> base 27 -> binary)
    #We do round up here, which means the output from the program can sometimes be a letter longer than designed
    return math.ceil(math.log2(chr_set**word_len))

def groupScoring(group_set,reference_set):
    #group_set consists of a list of n lists of two lists of words
    #The words in each of the two lists are the same, but the split between the groups of words is different.
    #How many ways can you divide 40 words into two groups? 40!/20!, I think, since once you've got 20 the other 20 are fixed.
    #Our metric: For any two words how often are they placed in the same group?
    full_words = group_set[0][0] + group_set[0][1] #List of all words
    num_sets = len(group_set) #Number of sets
    pairdict = {a:{b:0 for b in full_words} for a in full_words} #Dictionary of word:word:instances
    #For each group:
    for a in group_set:
        for b in full_words:
            for c in full_words:
                if b != c:
                    #For each pair of different words, if they're in the same set add 1 to pair instances
                    if (b in a[0] and c in a[0]) or (b in a[1] and c in a[1]):
                        pairdict[b][c] += 1
    #Divide instances by possible instances
    for a in pairdict:
        for b in pairdict[a]:
            pairdict[a][b] = pairdict[a][b] / num_sets
    
    #Now: How do we get a score out of this?
    score = 0
    for a in reference_set:
        adist = 0
        for b in a:
            bdist = 0
            for c in a:
                if c != b:
                    bdist += (1-pairdict[b][c])
            adist += (bdist / (len(a) - 1))
        score += (adist / len(a))
    #Worst-case score swaps half the set
    #worst = len(reference_set[0]) / (2*len(reference_set[0]) - 2)
    #We can't swap a half value, so the worst-case for an odd number is equal to the worst case for the next even number
    #Scaling this way gets us distances from 0 (same) to 1 (maximum distance) regardless of length
    #Experimentally, then, looking at groups of up to 250 random shuffles they don't vary much, from .96 to .98
    #Shorter shuffles did occasionally produce something with exactly 50% swaps.
    #What, then, are the odds of a set producing something else?
    #Not sure how we do this analytically.
    #On doing a perfectly random shuffle the odds of any word being in the front half is 50% and the back half is 50%.
    #Which means the odds of two words ending up in the same half as each other after shuffling is 50%
    #Does the dice binomial spreadsheet help us much here?
    #Alternate approach: There are 40!/20! different configuratons. Combinations?
    #How many ways to choose 20 things from a set of 40? 40!/(20!*(40-20)!)
    #Roughly 1.38e11 combinations
    #How many ways to choose 20 things from a set of 40 that are exactly n swaps from the original setup?
    #n swaps = n!/(20-n)! (20, 380, 6840, etc.)
    #Not sure this is entirely correct, given that 20!/10! = 6.7e11.
    #Let's do some simpler cases.
    #Case for 2 things from a set of 4:
    '''
    [a,b],[c,d]
    Can be:
    [a,b],[c,d]
    [a,c],[b,d]
    [a,d],[c,b]
    
    Anything else is an isomer of this.
    Does the combinations formula hold here?
    4!/2!*(4-2)! = 24 / (2*2) = 24 / 4 = 6
    This treats [a,b],[c,d] as separate from [c,d],[a,b]
    So with "which group is which doesn't matter" we're also dividing that by 2
    
    Does this hold for the case of 3 things from a set of 6?
    
    [a,b,c],[d,e,f]
    
    [a,b,d],[c,e,f]
    [a,b,e],[c,d,f]
    [a,b,f],[c,d,e]
    
    [a,d,c],[b,e,f]
    [a,e,c],[b,d,f]
    [a,f,c],[b,d,e]
    
    [a,d,e],[b,c,f]
    [a,d,f],[b,c,e]
    [a,e,f],[b,c,d]
    
    6!/(3!*(6-3)!)*2 = 720 / 72 = 10
    
    Okay. For the case of 4 from a group of 8, then, we need 8!/(4!**2 *2 ) = 35
    How many of those are single swaps vs double swaps?
    '''
    n = len(reference_set[0]) + (len(reference_set[0]) % 2)
    worst = n / (2*n - 2)
    return (score / 2) / worst

#Now: Randomness

def diff(l1,l2):
    return sum([1 if a not in l2 else 0 for a in l1])

def swaps(ref,comp,fold = False):
    if fold:
        mdif = len(ref[0])//2
        tdif = diff(comp[0],ref[0])
        if tdif > mdif: dif = len(ref[0]) - tdif
        else: dif = tdif
        return dif
    else:
        return diff(comp[0],ref[0])

def combinations(inl,fold = False):
    templ = [a for a in inl]
    for b in range(len(inl)//2 - 1):
        newl = []
        for c in templ:
            for d in inl:
                if d not in c:
                    e = sorted("".join(c)+d)
                    if e not in newl:
                        if fold:
                            if inl[0] in e:
                                newl.append(e)
                        else:
                            newl.append(e)
        templ = newl
    outl = []
    for a in templ:
        outl.append([a,[b for b in inl if b not in a]])
    return outl

def swapdistro(inl,fold = False):
    c = combinations(inl,fold = fold)
    swapdict = {0:1}
    for a in c[1:]:
        i = swaps(a,c[0],fold = fold)
        if i in swapdict: swapdict[i] += 1
        else: swapdict[i] = 1
    return swapdict

def nswaps(n):
    subline = [(math.factorial(n)/(math.factorial(a) * math.factorial(n-a)))**2 for a in range(n+1)]
    #To fold in half:
    if n%2 == 1:
        return subline[:(n+1)//2]
    else:
        return subline[:n//2] + [subline[(n//2)] / 2]

def resultChance(refn,testn):
    #First: Probability of n swaps
    pv = nswaps(len(refn[0]))
    tv = sum(pv)
    pv = [a/tv for a in pv]
    
    #Next: How many swaps?
    s = swaps(refn,testn,fold = True)
    return pv[s]

#PRONOUNCABILITY METRIC

def letChain(in_set):
    #Defines a Markov probability space of letter->letter, including starts and ends
    #Takes as input a list of words
    all_letters = "^$ " + "".join([chr(97+a) for a in range(31)]) #32 letters, ASCII 97-128 plus space, ^ (start of word), and $ (end of word)
    statechain = {a:{b:0 for b in all_letters} for a in all_letters} #Each letter has a probability of being followed by another letter
    for a in in_set:
        t = "^"+a.strip()+"$" #Add start and end as explicit characters, but strip out leading/trailing spaces, we're trying to make words of varying length here
        for b in range(len(t)-1):
            #We're comparing letters to next-letter, so we stop one short of the end
            statechain[t[b]][t[b+1]] += 1
    
    #Once we have our dictionary of letter->next-letter probabilities we need to divide them out to get from integer quanitities to proportions
    for a in statechain:
        tot_a = sum([statechain[a][b] for b in all_letters])
        for b in all_letters:
            if tot_a != 0: #There'll be a div/0 error when we hit the end of line character if we don't do this
                statechain[a][b] = statechain[a][b] / tot_a
    return statechain

def letChain2(in_set):
    all_letters = " " + "".join([chr(97+a) for a in range(31)])
    all_pairs = [c for d in [[b+a for b in all_letters] for a in all_letters] for c in d]
    statechain = {a:{b:0 for b in all_letters} for a in all_pairs}
    for a in in_set:
        t = a.strip()
        for b in range(len(t)-2):
            statechain[t[b:b+2]][t[b+2]] += 1
    for a in statechain:
        tot_a = sum([statechain[a][b] for b in all_letters])
        for b in all_letters:
            if tot_a != 0: #There'll be a div/0 error when we hit the end of line character if we don't do this
                statechain[a][b] = statechain[a][b] / tot_a
    return statechain    

def wordScore(word,state_chain):
    #Gets the score of a word from the Markov probability chain
    t = "^"+word.strip()+"$" #Strip extraneous spaces and add start-end characters
    score = 0
    for a in range(len(t) - 1): #We're comparing letters to the next letter, so we need to stop one short to avoid overshooting the string
        score += state_chain[t[a]][t[a+1]]
    return score / (len(t) - 1) #We want to return an average score, so longer words don't automatically score higher

def score2(words,state_chain):
    score = 0
    for a in words:
        t = a.strip()
        ts = 0
        for b in range(len(t) - 2):
            ts += state_chain[t[b:b+2]][t[b+2]]
        score += (ts / len(t))
    return score / len(words)

def listScore(words,state_chain):
    score = 0
    for a in words:
        score += wordScore(a,state_chain)
    return score / len(words)

#Next problem: Can we genuinely score pronouncability this way?

def wordsByEpoch(in_file):
    d1 = [a for a in open("ArsOut/{}.txt".format(in_file).replace(".txt.txt",".txt")).read().splitlines() if a.split("||")[0] == "Words"]
    
    #Using | as a separator was a mistake, since it's in the possible character set, but we know we're producing uniform outputs of 10 pipe-separated len 7 words separated by double-pipes
    
    d2 = []
    
    for a in d1:
        #There should be splits at 5-6
        splitdexes = [(7,86),(88,167),(169,248),(250,329),(331,410)]
        l2 = []
        for b in splitdexes:
            l2.append(a[b[0]:b[1]])
        d2.append(l2)
    
    d3 = []
    
    for a in d2:
        splitdexes = [(0,7),(8,15),(16,23),(24,31),(32,39),(40,47),(48,55),(56,63),(64,71),(72,79)]
        l3 = []
        for b in a:
            for c in splitdexes:
                l3.append(b[c[0]:c[1]])
        d3.append(l3)
    
    return d3

def trendLine(in_list,trend_window):
    out_list = []
    for a in range(len(in_list)):
        if a < trend_window:
            out_list.append(sum(in_list[:a+1]) / (a+1))
        else:
            out_list.append(sum(in_list[a-trend_window:a+1]) / (trend_window + 1))
    return out_list

r1 = [a for a in json.load(open("ArsData/5_7_la.json","r"))]
m1 = letChain(r1)
m12 = letChain2(r1)
r2 = [a for a in json.load(open("ArsData/5_7_pokel.json","r"))]
m2 = letChain(r2)
m22 = letChain2(r2)

r3 = [a for a in json.load(open("ArsData/5_7_lb.json","r"))]
m3 = letChain(r3)
m32 = letChain2(r3)

g1 = [a for a in json.load(open("ArsData/5_7_l_g.json","r"))]

t1 = wordsByEpoch("022824 Regtest 1")
t2 = wordsByEpoch("022724 Poketest")

t3 = wordsByEpoch("022824 Regtest 2")

#Okay. We have Markov chains for real and Pokemon reference, gibberish as a comparison point, and test sets.
#First: Self-reference.
#Average self-reference score for r1 is...0.111688.
#For r2: 0.10367
#g1 vs m1: 0.04734
#g1 vs m2: 0.04856
#We'll have to test more, but overall let's say that's our rough range of measure.

#What do the best/worst words look like?

#flat1 = [c for d in [[[a,score2([a],m12),b] for a in t1[b]] for b in range(len(t1))] for c in d]
#flat1.sort(key = lambda a:a[1])

#for a in flat1:
#    print(a)

#t12l = [score2(a,m12)/score2(r1,m12) for a in t1]

#First: Graph reference lines
ref1 = [listScore(r3,m3) for a in t3]
ref2 = [score2(r3,m32) for a in t3]
gib1 = [listScore(g1,m3) for a in t3]
gib2 = [score2(g1,m32) for a in t3]

test1 = [listScore(a,m3) for a in t3]
smooth1 = trendLine(test1,50)
test2 = [score2(a,m32) for a in t3]
smooth2 = trendLine(test2,50)

#plt.plot(ref1,label = "Ref w1")
plt.plot(ref2,label = "Ref w2")
#plt.plot(gib1,label = "Gib w1")
plt.plot(gib2,label = "Gib w2")
#plt.plot(test1,label = "Run w1")
plt.plot(test2,label = "Run w2")
#plt.plot(smooth1,label = "Smoothed w1")
plt.plot(smooth2,label = "Smoothed w2")

plt.legend()
plt.show()




'''
#Regular test vs. regular, pokemon
t1l = [listScore(a,m1) for a in t1]
t1l2 = [listScore(a,m2) for a in t1]
#Poketest vs. regular, pokemon
t2l = [listScore(a,m1) for a in t2]
t2l2 = [listScore(a,m2) for a in t2]

#Regular ref vs. regular, pokemon
r1l = [listScore(r1,m1)]*len(t2l)
r1l2 = [listScore(r1,m2)]*len(t2l)
#Pokeref vs. regular, pokemon
r2l = [listScore(r2,m1)]*len(t2l)
r2l2 = [listScore(r2,m2)]*len(t2l)

#Gibberish
gl1 = [listScore(g1,m1)]*len(t2l)
gl2 = [listScore(g1,m2)]*len(t2l)

smth = 20

#plt.plot(t1l,label = "Treg v. reg")
#plt.plot(t1l2,label = "Treg v. poke")
#plt.plot(trendLine(t1l,smth),label = "Treg v. reg")
#plt.plot(trendLine(t1l2,smth),label = "Treg v. poke")
#plt.plot(t2l,label = "Tpoke v. reg")
#plt.plot(t2l2,label = "Tpoke v. poke")
plt.plot(trendLine(t2l,smth),label = "Tpoke v. reg")
plt.plot(trendLine(t2l2,smth),label = "Tpoke v. poke")
#plt.plot(r1l,label = "Rreg v. reg")
#plt.plot(r1l2,label = "Rreg v. poke")
plt.plot(r2l,label = "Rpoke v. reg")
plt.plot(r2l2,label = "Rpoke v. poke")
plt.plot(gl1,label = "Gib v. reg")
plt.plot(gl2,label = "Gib v. poke")

plt.legend()
plt.show()
'''