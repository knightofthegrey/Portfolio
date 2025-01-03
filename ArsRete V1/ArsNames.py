#Ars Namemodule

#This module contains structures to make names
#This was my initial attempt at the problem, before I first tried applying neural networks.

from ArsUtilities import *
import random
import time

alph = "abcdefghijklmnopqrstuvwxyz"
volph = "aeiouy"
colph = "bcdfghjklmnpqrstvwxz"

#Some Computational Linguistics
#Problem: Pronouncable syllables?

def changeDictionary(filename):
    #tempfile = open(filename,"r",encoding = "utf-8").read().splitlines()
    tempfilelist = [a.lower().split(" ")[0] for a in open(filename,"r",encoding = "utf-8").read().splitlines() if "abbr." not in a and a.split(" ")[0] != "usage"]
    infilelist = []
    for a in tempfilelist:
        if a.isalpha(): infilelist.append(accentreplace(a))
        elif a[:-1].isalpha():
            if accentreplace(a[:-1]) not in infilelist: infilelist.append(accentreplace(a[:-1]))
    vposlist = ["".join(["1" if a in "aeiouy" else "0" for a in b]) for b in infilelist]
    vchunklist = ["".join([b[a] for a in range(len(b)-1) if b[a] == "0" or b[a+1] == 0]) for b in vposlist]
    wordsfile = open("words.txt","w")
    vposfile = open("vpos.txt","w")
    vchunkfile = open("vchunk.txt","w")
    for a in range(len(infilelist)):
        wordsfile.write(infilelist[a] + "\n")
        vposfile.write(vposlist[a] + "\n")
        vchunkfile.write(vchunklist[a] + "\n")
    wordsfile.close()
    vposfile.close()
    vchunkfile.close()

def accentreplace(inword):
    outword = inword.replace(u"œ","oe")
    letterpairs = [u"áâàäéêèîïôöüûñç",u"aaaaeeeiioouunc"]
    for x in range(15):
        outword = outword.replace(letterpairs[0][x],letterpairs[1][x])
    return outword

def openDictionary():
    predata = [open(a,"r").read().splitlines() for a in ["words.txt","vpos.txt","vchunk.txt"]]
    intdata = [[predata[a][b] for a in range(3)]for b in range(len(predata[0]))]
    intdata.sort(key = lambda a:a[0])
    return [intdata[a] for a in range(len(intdata) - 1) if intdata[a] != intdata[a+1]]

def possibleSyllables(word):
    #Any given vowel could be a new syllable or a compound vowel.
    #Assume for the time being all vowels are compound
    #Data we have on input is: word, vowel/consonant breakdown, compressed vowel version
    padword = "0" + word[1]
    padbreaks = []
    vowelbreaks = [a for a in range(len(word[0])) if padword[a] == "0" and padword[a+1] == "1"] + [len(word[0])]
    #A primitive take on syllables, then, would be simply to take from vowelbreak to vowelbreak
    sylblocks1 = [word[0][vowelbreaks[a]:vowelbreaks[a+1]] for a in range(len(vowelbreaks) - 1)]
    #Alternate primitive take: assume leading consonants?
    vowelbreaks2 = [0] + [a+1 for a in vowelbreaks[:-1]] + [vowelbreaks[-1]]
    sylblocks2 = [word[0][vowelbreaks2[a]:vowelbreaks2[a+1]] for a in range(len(vowelbreaks2) - 1)]
    #Neither approach is very good?
    #Let's try an alternate approach: prioritize breaking up consonants?
    #Given a list of 0(c) and 1(v):
    cv = word[1]
    padcv = "0" + word[1]
    #Multi-consonant breaks doesn't really help either...
    #'Kay, breaking into syllables isn't helping.
    #Let's try frequency analysis:

def adjgroup(dictionary):
    #How likely is it that any given letter is adjacent to any other letter?
    letterprecededict = {a:{b:0 for b in "_abcdefghijklmnopqrstuvwxyz"} for a in "abcdefghijklmnopqrstuvwxyz"}
    letterfollowdict = {a:{b:0 for b in "_abcdefghijklmnopqrstuvwxyz"} for a in "abcdefghijklmnopqrstuvwxyz"}
    lettertotaldict = {a:0 for a in "abcdefghijklmnopqrstuvwxyz"}
    posdict = {a:{b:0 for b in "abcdefghijklmnopqrstuvwxyz"} for a in range(25)}
    postotaldict = {a:0 for a in range(25)}
    wordlendict = {a:0 for a in range(25)}
    totalwords = 0
    for word in dictionary:
        wordlendict[len(word[0])] += 1
        totalwords += 1
        for x in range(len(word[0])):
            #We've found a letter! Increment instances of the letter
            lettertotaldict[word[0][x]] += 1
            postotaldict[x] += 1
            posdict[x][word[0][x]] += 1
            #Letter before word[0][x] is the letter before it in word[0]
            try: letterprecededict[word[0][x]][word[0][x-1]] += 1
            #If out of range then it's the start of a word and we increment space
            except: letterprecededict[word[0][x]]["_"] += 1
            #Letter after word[0][x] is the letter after it in word[0]
            try: letterfollowdict[word[0][x]][word[0][x+1]] += 1
            #If out of range then it's the end of a word and we increment space
            except: letterfollowdict[word[0][x]]["_"] += 1
    for letter in letterprecededict.keys():
        for key in letterprecededict[letter].keys():
            letterprecededict[letter][key] = letterprecededict[letter][key] / lettertotaldict[letter]
            letterfollowdict[letter][key] = letterfollowdict[letter][key] / lettertotaldict[letter]
    for x in range(25):
        for y in posdict[x].keys():
            try: posdict[x][y] = posdict[x][y] / postotaldict[x]
            except: posdict[x][y] = 0
    inverseposdict = {a:{b:posdict[b][a] for b in range(25)} for a in "abcdefghijklmnopqrstuvwxyz"}
    letterfreqdict = {a:lettertotaldict[a] / sum([lettertotaldict[b] for b in "abcdefghijklmnopqrstuvwxyz"]) for a in "abcdefghijklmnopqrstuvwxyz"}
    lenfreqdict = {a:wordlendict[a] / totalwords for a in range(25)}
    return (letterfreqdict,letterprecededict,letterfollowdict,posdict,inverseposdict,lenfreqdict)

def freqsort(inDict):
    #Inputs: Dictionary of letter:float
    #Outputs: List of letter,float in sorted order and print representation
    qflist = [[a,inDict[a]] for a in inDict.keys()]
    qflist.sort(key = lambda a:a[1],reverse = True)
    return (qflist," ".join(["{}: {:.5f}".format(a[0],a[1]) for a in qflist]))

def letterText(inDict):
    #Writes various info about letter frequencies to a file for later review
    letterData = adjgroup(inDict)
    lettertext = open("lettertext.txt","w")
    lettertext.write("INFORMATION ABOUT LETTER FREQUENCIES IN THE ENGLISH LANGUAGE\n\n")
    lettertext.write("BASIC LETTER FREQUENCIES:\n")
    lettertext.write(freqsort(letterData[0])[1] + "\n\n")
    lettertext.write("WORD LENGTH FREQUENCIES:\n" + freqsort(letterData[5])[1] + "\n\n")
    for letter in "abcdefghijklmnopqrstuvwxyz":
        lettertext.write("INFORMATION ABOUT THE LETTER {}\n".format(letter))
        lettertext.write("Most common preceding letters: " + freqsort(letterData[1][letter])[1] + "\n")
        lettertext.write("Most common following letters: " + freqsort(letterData[2][letter])[1] + "\n")
        lettertext.write("Most common positions in a word: " + freqsort(letterData[4][letter])[1] + "\n\n")
    lettertext.close()

def averageness(dictData):
    #Inputs: list of lists of strings outputted by openDictionary
    #Outputs: Prints the most average and least average words in that dictionary
    adjData = adjgroup(dictData)
    avgLens = {}
    for a in dictData:
        if len(a[0]) in avgLens.keys():
            avgLens[len(a[0])] += 1
        else:
            avgLens[len(a[0])] = 1
    for a in avgLens.keys():
        avgLens[a] = avgLens[a] / len(dictData)
    
    #print(freqsort(avgLens)[1])
    avglist = []
    
    for word in dictData:
        lenavg = avgLens[len(word[0])]
        letterposavg = sum([adjData[3][x][word[0][x]] for x in range(len(word[0]))]) / len(word[0])
        #Pairweights:
        pairscore = 0
        padword = "_" + word[0] + "_"
        for x in range(1,len(padword) - 1):
            pairscore += adjData[2][padword[x]][padword[x+1]] * adjData[0][padword[x]]
            pairscore += adjData[1][padword[x]][padword[x-1]] * adjData[0][padword[x]]
        avgpairscore = pairscore / (2*len(word[0]))
        avgletterfreq = sum([adjData[0][word[0][x]] for x in range(len(word[0]))]) / len(word[0]) / len(word[0])
        avgall = (lenavg + letterposavg + avgpairscore + avgletterfreq)/4
        avglist.append((word[0],avgall))
    
    avglist.sort(key = lambda a:a[1])
    clippedlist = [avglist[a] for a in range(1,len(avglist)) if avglist[a] != avglist[a-1]]
    avgavg = sum([a[1] for a in clippedlist]) / len(clippedlist)
    
    avgavglist = [(a[0],a[1],abs(avgavg - a[1])) for a in clippedlist]
    avgavglist.sort(key = lambda a:a[2])
    for x in avglist: print(x)

def randomWord(adjData):
    #Problem: Combined probability score
    #We know four things:
    #-The raw odds of a letter being any letter
    #-The odds of the letter following a letter to be a letter
    #-The odds of a letter at a point in a word being a letter
    #-The odds of a word being a particular length
    #Determine word length at random
    lenset = [a for a in range(1,22)]
    lenweights = [adjData[5][a] for a in lenset]
    wordlen = random.choices(lenset,lenweights)
    
    word = "_"
    for x in range(wordlen[0]):
        #Adding a letter?
        #Average letter: adjData[0]
        #Letter following another letter: adjData[1]
        #Letter at position: adjData[3][x]
        letterFreqs = [adjData[0][a] for a in alph]
        letterAtPos = [adjData[3][x][a] for a in alph]
        letterFollowing = [adjData[2][a][word[-1]] for a in alph]
        avgLetters = [(letterFreqs[a] + letterAtPos[a] + 2*letterFollowing[a]) / 4 for a in range(26)]
        word += random.choices(alph,weights = avgLetters)[0]
    return word[1:]

#Word Analysis:
#In concept we should have sufficient computing power to look at a letter in relation to a bunch of other letters in the word, or to look at lettergroups

def wordlenfreq(inDict):
    #Determines the relative frequency of words of various lengths in the dictionary provided
    #Input: A list of string,string,string tuples representing a word and its consonant-vowel pattern, this only uses the word
    #Output: A dictionary of int:float pairs representing word lengths and their frequency in the dictionary
    maxlen = max([len(a[0]) for a in inDict]) + 1
    wordlendict = {a:0 for a in range(maxlen)}
    for word in inDict:
        wordlendict[len(word[0])] += 1
    return {a:wordlendict[a] / len(inDict) for a in range(maxlen)}

def vPattern(inDict,plen = 3):
    #Determines the relative frequency of vowel-consonant patterns from words in the dictionary provided.
    #Input: A list of string,string,string tuples representing a word and its consonant-vowel pattern
    #The c-v string uses 1 for vowels and 0 for consonants
    cvGroups = []
    cvPos = []
    for x in range(plen):
        #Possible vowel patterns. Could theoretically represent these as integers? Sticking to strings for now.
        patterns = ["{:0>{}}".format(str(bin(a))[2:],x+1) for a in range(2**(x+1))]
        cvX = {a:0 for a in patterns}
        cvP = {a:{b:0 for b in range(21)} for a in patterns}
        totalX = 0
        for word in inDict:
            #For each word, look at patterns in word[1]
            wordgroups = [word[1][n:n+x+1] for n in range(len(word[1]) - x)]
            for a,y in enumerate(wordgroups):
                cvX[y] += 1
                cvP[y][a] += 1
                totalX += 1
        cvGroups.append({a:cvX[a] / totalX for a in cvX})
        cvPos.append({a:{b:cvP[a][b] / sum([cvP[a][c] for c in range(21)]) for b in range(21)} for a in cvP})
    return cvGroups,cvPos

def dupset(inDict):
    #Determines the number of duplicates of each letter in the average word
    letterdict = {a:{b:0 for b in range(21)} for a in alph}
    for word in inDict:
        lettercounts = {a:0 for a in alph}
        for a in word[0]:
            lettercounts[a] += 1
        for b in alph:
            letterdict[a][lettercounts[a]] += 1
    return {a:{b:letterdict[a][b] / len(inDict) for b in range(21)} for a in alph}
        

def groupfreqdict(inDict,groupsize = 3):
    #Determines the relative frequency of sequences of letters in the dictionary provided
    #Input: A list of string,string,string tuples representing a word and its consonant-vowel pattern, this only uses the word
    #Output: A list of dictionaries of string-float pairs, representing runs of letters of length up to groupsize and their relative frequency in the dictionary
    letterpairdict = [{a:0 for a in b} for b in lettergroups(groupsize)]
    letterpairposdict = [{a:{b:0 for b in range(21)} for a in b} for b in lettergroups(groupsize)]
    totalsizelist = [0 for a in range(groupsize)]
    for word in inDict:
        for group in range(groupsize):
            wordgroups = [word[0][n:n+group+1] for n in range(len(word[0]) - group)]
            totalsizelist[group] += len(wordgroups)
            for x,letters in enumerate(wordgroups):
                letterpairdict[group][letters] += 1
                letterpairposdict[group][letters][x] += 1
    letterpairfreq = [{a:letterpairdict[b][a] / totalsizelist[b] for a in letterpairdict[b].keys()} for b in range(groupsize)]
    letterposfreq = []
    for a in range(groupsize):
        #For each group:
        groupdict = {}
        for b in letterpairposdict[a].keys():
            #For each letter in that group:
            grouptotal = sum([letterpairposdict[a][b][c] for c in letterpairposdict[a][b].keys()])
            subgroupdict = {c:letterpairposdict[a][b][c] / grouptotal if grouptotal != 0 else 0 for c in letterpairposdict[a][b].keys() }
            groupdict[b] = subgroupdict
        letterposfreq.append(groupdict)
    return letterpairfreq,letterposfreq
                
def lettergroups(n):
    lettergroups = [[a for a in alph]]
    for x in range(n-1):
        lettergroups.append([a+b for a in alph for b in lettergroups[-1]])
    return lettergroups

def averagenessScore(word,lendict,groupdict,posdict):
    #print(word)
    #First: Word length, this one's easy
    lenscore = lendict[len(word)]
    #Second: Average group frequency score
    groupScoreSum = 0
    groupLensRun = 0
    for x in range(len(groupdict)):
        #Divide word into list of groups of length (x+1)
        if x < len(word):
            wordgroups = [word[n:n+x+1] for n in range(len(word) - x)]
            groupScoreX = sum([groupdict[x][n] for n in wordgroups]) / len(wordgroups)
            groupScoreSum += groupScoreX
            groupLensRun += 1
    groupScore = groupScoreSum / groupLensRun
    #Third: Average group position score
    groupPosSum = 0
    groupLensRun = 0
    for x in range(len(groupdict)):
        if x < len(word):
            #Divide word into list of groups of length(x+1)
            wordgroups = [word[n:n+x+1] for n in range(len(word) - x)]
            posSumX = 0
            for y in range(len(wordgroups)):
                posSumX += posdict[x][wordgroups[y]][y] * groupdict[x][wordgroups[y]]
            groupPosSum += posSumX / len(wordgroups)
            groupLensRun += 1
    posScore = groupPosSum / groupLensRun
    return [lenscore,groupScore,posScore]

def averagenessSubscores(word,lendict,groupdict,posdict):
    groupscores = []
    posscores = []
    lenscore = lendict[len(word)]
    groupsRun = 0
    for x in range(len(groupdict)):
        #print(x)
        if x < len(word):
            wordgroups = [word[n:n+x+1] for n in range(len(word) - x)]
            groupScoreX = sum([groupdict[x][n] for n in wordgroups]) / len(wordgroups)
            posSumX = sum([posdict[x][wordgroups[y]][y] * groupdict[x][wordgroups[y]] for y in range(len(wordgroups))]) / len(wordgroups)
            groupscores.append(groupScoreX)
            posscores.append(posSumX)
    outscores = groupscores + posscores + [lenscore]
    return (word,outscores)

def equalWeights(inList):
    #Re-weight parameters equally?
    maxlist = []
    for x in range(len(inList[0][1])):
        maxlist.append(max([a[1][x] for a in inList]))
    outlist = []
    for x in range(len(inList)):
        newParameters = [inList[x][1][n] / maxlist[n] for n in range(len(inList[0][1]))]
        totalScore = sum(newParameters) / len(newParameters)
        outlist.append((inList[x][0],totalScore,newParameters,inList[x][1]))
    return outlist
        
def averagenessCalc(inputDicts):
    #Compute the averageness score across all words in dictData
    midlist = []
    for word in inputDicts[0]:
        if len(word[0]) > 2:
            midlist.append(averagenessSubscores(word[0],inputDicts[1],inputDicts[2],inputDicts[3]))
    outList = equalWeights(midlist)
    outList.sort(key = lambda a:a[1])
    return outList


def quickScore(word,lens,freqs,dups):
    #print(len(freqs))
    lenscore = lens[len(word)] * 2
    dupint = {a:list(word).count(a) for a in alph}
    dupscore = sum([dups[a][dupint[a]] for a in alph]) / len(word) / 3
    freqscore = sum([freqs[a] for a in word]) / len(word) * 5
    vscore = (1-abs(0.5-sum([1 if a in volph else 0 for a in word]) / len(word))) / 3
    return sum([lenscore,dupscore,freqscore,vscore])

#What other metrics should we use?

def genWord(inputDicts):
    #Generates a random gibberish word using letter frequencies and letter group frequencies
    lenData = inputDicts[0]
    groupData = inputDicts[1]
    posData = inputDicts[2]
    vData = inputDicts[3]
    vPosData = inputDicts[4]
    dupData = inputDicts[5]
    
    lenword = random.choices(list(lenData.keys()),weights = [lenData[a] for a in lenData.keys()])[0]
    word = " " * lenword
    #Pick a random letter
    for x in range(lenword):
        nextIndex = random.choice([a for a in range(lenword) if word[a] == " "])
        #We now need the combined probability of nextIndex being any given letter
        #Which is the sum of the probability of all patterns that could match.
        #Adjstarts gives the possible start positions of a group of n letters that includes nextIndex, e.g. for groups of 3 at position 4
        #the group could be 2,3,4 at position 2, 3,4,5 at position 3, or 4,5,6 at position 4
        adjstarts = [[a for a in range(nextIndex - b,nextIndex + 1) if a+b in range(lenword) and a in range(lenword)] for b in range(3)]
        #Now: combined probability
        #letterprobs = {a:0 for a in alph}
        groupprobs = []
        for y in range(3):
            #Next problem: consider c-v groups.
            #For each category of letter grouping:
            letterprobs = {a:0 for a in alph}
            for a in range(nextIndex - y,nextIndex + 1):
                if a in range(lenword) and a+y in range(lenword):
                    #print("A:",a)
                    #For group starting at index a:
                    testgroup = word[a:a+y+1]
                    #What index in the subgroup is the index we're replacing?
                    #Consider case "ab " (a = nextindex - y): replacement is index 2. In case "b c" (a = nextindex - y + 1) replacement is index 1
                    replacedex = y-nextIndex+a
                    for b in groupData[y]:
                        #For matches:
                        if possmatch(word[a:a+y+1],b):
                            #Then we look at the letter at the replace index in b, and give that letter the points for group b
                            #Let's multiply this by the probability of a c-v pattern of that size for now
                            letterprobs[b[replacedex]] += groupData[y][b] * vData[y]["".join(["1" if c in volph else "0" for c in b])]
            groupprobs.append(letterprobs)
        #Problem: Weights of letter groups differ. Multiply by 1/sum.
        weights = []
        for b in range(3):
            subweight = sum([groupprobs[b][a] for a in groupprobs[b]])
            if subweight == 0: weights.append(subweight)
            else: weights.append(1/subweight)
                #[1/sum([groupprobs[b][a] for a in groupprobs[b]]) for b in range(3)]
        #Hrm. Let's fiddle with weights a bit, see what that does.
        weights = [weights[0],weights[1],weights[2]]
        problist = [{a:groupprobs[b][a] * weights[b] for a in alph} for b in range(3)]
        #Weighted problist works
        #Hrm. Sum or average? Zeroes present a problem. Product?
        choicelist = [sum([problist[0][a] * problist[1][a] * problist[2][a]]) for a in alph]
        choicelist = [a if a > 0 else 0.000001 for a in choicelist]
        letterCounts = {a:list(word).count(a) for a in alph}
        choicelist = [choicelist[alph.index(a)] * dupData[a][letterCounts[a] + 1] for a in alph]
        randletter = random.choices(alph,choicelist)[0]
        word = word[:nextIndex] + randletter + word[nextIndex+1:]
        #print(word)
    return word

def possmatch(set1,set2):
    #Used to determine if strings set1 and set2 match, treating spaces as wild
    #Used as a helper function for generating words
    matches = [1 if set1[a] == set2[a] or set1[a] == " " else 0 for a in range(len(set1))]
    return sum(matches) == len(set1)
    
def genWordset(numwords):
    start1 = time.time()
    dictData = openDictionary()
    adjData = adjgroup(dictData)
    lenData = wordlenfreq(dictData)
    groupData,groupPosData = groupfreqdict(dictData)
    vData,vPosData = vPattern(dictData)
    dupData = dupset(dictData)
    print("Pattern check done in {}".format(time.time() - start1))
    outList = []
    tries = 0
    start2 = time.time()
    while len(outList) < numwords:
        tries += 1
        testWord = genWord([lenData,groupData,groupPosData,vData,vPosData,dupData])
        outList.append(testWord)
    print("Words generated in {}".format(time.time() - start2))
    #This is *terrible*. But hopefully enough to help with the training data tool.
    return outList

def initialize():
    dictData = [a for a in open("qwords.txt","r").read().splitlines() if len(a) > 2]
    dictSubData = [[a,0] for a in dictData]
    lenData = wordlenfreq(dictSubData)
    freqData = groupfreqdict(dictSubData,groupsize = 1)[0][0]
    dupData = dupset(dictSubData)
    return [dictData,lenData,freqData,dupData]

def moreFakeWords():
    #Generates fake words and appends them to the fakeWords.txt file
    dictData,lenData,freqData,dupData = initialize()
    finalWords = []
    running = True
    while running:
        fakeWords = []
        while len(fakeWords) < 10:
            tempWords = genWordset(5)
            for word in tempWords:
                if quickScore(word,lenData,freqData,dupData) > 0.8:
                    fakeWords.append("".join([word[0]] + [word[a] for a in range(1,len(word)) if word[a] != word[a-1]]))
        finalWords += fakeWords
        print("New words added to list: {}".format(fakeWords))
        command = input("Continue? (Y/N)")
        if command.lower() == "n":
            running = False
    print("Writing to fakeDict")
    fakeDict = open("fakeWords.txt","a")
    for a in finalWords:
        fakeDict.write("{}\n".format(a))
    fakeDict.close()

def checkTest(numwords):
    dictData,lenData,freqData,dupData = initialize()
    testset = random.choices(dictData,k = numwords)
    for word in testset:
        print(word,quickScore(word,lenData,freqData,dupData))

def bigAvg():
    dictData,lenData,freqData,dupData = initialize()
    avgList = [[a,quickScore(a,lenData,freqData,dupData)] for a in dictData]
    avgList.sort(key = lambda a:a[1])
    for line in avgList: print(line)
    
def genRandomFakeWords():
    dictData,lenData,freqData,dupData = initialize()
    #Makes random gibberish with consideration for quickScore only
    outlist = []
    for x in range(500):
        randlen = random.choice(range(4,13))
        randword = "".join([random.choice(alph) for a in range(randlen)])
        if quickScore(randword,lenData,freqData,dupData) > 0.8: outlist.append(randword)
    fakeDict = open("fakeWords.txt","a")
    for line in outlist:
        fakeDict.write("{}\n".format(line))
    fakeDict.close()

#bigAvg()

#dictData,lenData,freqData,posData = initialize()

#print(len(dictData))

    
    

#testGenerator = genWordset(5)

#for line in testGenerator: print(line)
        
    

