#Grab-bag of utility functions for use across the program


import random

def tableToFile(indexes,data,filename = "tableOut.txt"):
    #Same basic idea as debugTable in DataProcessor
    widthlist = [len(a) for a in indexes]
    outFile = open("outputData/{}".format(filename),"w")
    for row in data:
        for x in range(len(row)):
            widthlist[x] = max(len(row[x]),widthlist[x])
    for x in range(len(indexes)):
        if x == (len(indexes) - 1): outFile.write("{0: <{1}}".format(indexes[x],widthlist[x]))
        else: outFile.write("{0: <{1}}|".format(indexes[x],widthlist[x]))
    outFile.write("\n")
    for row in data:
        outputRow = ""
        for x in range(len(row)):
            outputRow += "{0: <{1}}|".format(row[x],widthlist[x])
        outFile.write("{}\n".format(outputRow[:-1]))
    outFile.close()

def dateComparison(indate,comparedate):
    #Determines the relationship between two dates, if they're within a year of each other
    try:
        testdate = numericDay(indate)
        testcompare = numericDay(comparedate)
        if testdate[2] == testcompare[2]: return testdate[0] - testcompare[0]
        elif testdate[2] == testcompare[2] + 1: return testdate[0] + testcompare[1]
        elif testdate[2] == testcompare[2] - 1: return (-1) * (testdate[1] + testcompare[0])
        else: return "Not within a year"
    except:
        return "Not a date"

def numericDay(indate):
    #Converts a date in the format mmddyy to a count of days since the last new year
    monthlist = [0,31,28,31,30,31,30,31,31,30,31,30,31]
    numyear = 2000 + int(indate[4:])
    isLeapyear = (numyear % 4 == 0 and not numyear % 100 == 0) or numyear % 400 == 0
    if isLeapyear: monthlist[2] += 1
    nummonth = int(indate[:2])
    numday = int(indate[2:4])
    sumday = sum(monthlist[:nummonth]) + numday
    toend = sum(monthlist) - sumday
    return(sumday,toend,numyear)

def dictAppend(inDict,key,value):
    #Helper function for dictionary use
    outDict = inDict
    if key in outDict.keys(): outDict[key].append(value)
    else: outDict[key] = [value]
    return outDict

def uniqueDictAppend(inDict,key,value):
    #Another helper function for dictionary use
    outDict = inDict
    if key in outDict.keys():
        if value in outDict[key]: pass
        else: outDict[key].append(value)
    else: outDict[key] = [value]
    return outDict

def freqDict(inDict,value):
    outDict = inDict
    if value in outDict.keys():
        outDict[value] += 1
    else:
        outDict[value] = 1
    return outDict

def sortfreqDict(inDict):
    tempindex = [a for a in inDict.keys()]
    tempindex.sort()
    return {a:inDict[a] for a in tempindex}

def dictCompare(dict1,dict2):
    #Compare two dictionaries for debugging
    unionkeys = [a for a in dict1.keys() if a in dict2.keys()]
    keylist1 = [a for a in dict1.keys() if a not in dict2.keys()]
    keylist2 = [a for a in dict2.keys() if a not in dict1.keys()]
    print("Comparison:\nsize {} v {} with {} overlap".format(len(dict1.keys()),len(dict2.keys()),len(unionkeys)))
    matchkeys = [a for a in unionkeys if dict1[a] == dict2[a]]
    splitkeys = [a for a in unionkeys if a not in matchkeys]
    print("Of matched keys: {} match, {} differ\nDiffering keys:".format(len(matchkeys),len(splitkeys)))
    for a in splitkeys:
        print("Key {}: {} {}".format(a,dict1[a],dict2[a]))

def capitalize(instr):
    return instr[0].upper() + instr[1:].lower()

def bestmatch(instring,patternlist):
    #Check a string for its closest match in a list of strings
    #Used to help with OCR accuracy if you know what the possible answers should be
    #This and wfmatch below are largely concerned with single letter swaps or additions to the start/end of a string
    #Cases where things get inserted into the middle probably won't work well
    matchqual = []
    for pattern in patternlist:
        #For each pattern:
        if len(pattern) <= len(instring):
            #If the pattern is short enough to be found in instring:
            #Check by word, not by character, OCR errors thus far have been letter swaps
            subpatternlist = pattern.strip().split(" ")
            instringlist = instring.strip().split(" ")
            buffer = len(instringlist) - len(subpatternlist)
            #This tells us how many times we need to check the pattern against various sub-lists of the instringlist
            bestmatch = 0
            for x in range(buffer+1):
                matches = 0
                #Starts at x, ends at len(instringlist) - (buffer - x)
                teststringlist = instringlist[x:len(instringlist)-(buffer - x)]
                #At this point we have teststringlist and patternlist of the same length, and now need to fuzzymatch them word by word
                for x in range(len(teststringlist)):
                    matches += wfmatch(teststringlist[x],subpatternlist[x])
                bestmatch = max(bestmatch,matches)
            matchqual.append(bestmatch / len(subpatternlist))
        else:
            matchqual.append(0)
    return (patternlist[matchqual.index(max(matchqual))],max(matchqual))

def wfmatch(str1,str2):
    #Returns a float representing the percentage overlap of two strings
    if len(str1) != len(str2):
        longer = max(str1,str2,key = len)
        shorter = min(str1,str2,key = len)
        buffer = len(longer) - len(shorter)
    else:
        longer = str1
        shorter = str2
        buffer = 0
    #print(longer,shorter,buffer)
    bestmatch = 0
    for x in range(buffer + 1):
        matches = 0
        teststr = longer[x:len(longer) - (buffer - x)]
        #print("Test {}: {} vs {}".format(x,teststr,shorter))
        for a in range(len(teststr)):
            #print("Checking: {} vs {}".format(teststr[a],shorter[a]))
            if teststr[a] == shorter[a]:
                matches += 1
        bestmatch = max(bestmatch,matches)
    return bestmatch / len(shorter)

def scramblepatterns(inpatterns,numchanges = 2,dupes = 10):
    #Generates scrambled patterns for testing bestmatch
    outpatterns = []
    for pattern in inpatterns:
        #For each pattern: choose (numchanges) indexes to scramble
        #Gr. Would like to preserve spaces.
        cutpattern = cutspaces(pattern)
        for b in range(dupes):
            scrambleindexes = random.sample(range(0,len(cutpattern[0])),numchanges)
            duppattern = cutpattern[0][:]
            for a in scrambleindexes:
                duppattern = duppattern[:a] + chr(ord(duppattern[a]) + 1) + duppattern[a+1:]
            outpatterns.append(restorespaces(duppattern,cutpattern[1]))
    return outpatterns

def cutspaces(instr):
    #Removes spaces and preserves the indexes they should be at for restoring them to help with bestmatch
    spaceindexes = [a for a in range(len(instr)) if instr[a] == " "]
    outstr = instr.replace(" ","")
    cutindexes = []
    for x in range(len(spaceindexes)):
        cutindexes.append(spaceindexes[x] - x - 1)
    return (outstr,cutindexes)

def restorespaces(newstr,spaceindexes):
    #Restores spaces to a string that were removed by cutspaces to help with bestmatch
    restoredstr = ""
    for a in range(len(newstr)):
        restoredstr += newstr[a]
        if a in spaceindexes: restoredstr += " "
    return restoredstr

def bestnummatch(innum,numlist,debug = False):
    #Earlier version of bestmatch with less flexibility
    matchequal = [wfmatch(innum,a) for a in numlist]
    outlist = []
    for x in range(len(matchequal)):
        if matchequal[x] == max(matchequal):
            outlist.append(numlist[x])
    if debug:
        for a in range(len(matchequal)):
            print("{} compare to {}: {:.2f}%".format(innum,numlist[a],matchequal[a]))
    #Problem: Equal confidence matches? We'll run the test and see how common those turn out to be.
    return (outlist,max(matchequal))

def nummatchtest():
    #Test function for bestnummatch
    numlist = ["".join([str(random.randrange(0,9)) for a in range(6)]) for b in range(100)]
    testlist = []
    for a in numlist:
        for b in range(10):
            addex = random.randrange(1,8)
            tempnum = a[:]
            changedex = random.sample(range(6),2)
            for c in changedex:
                tempnum = tempnum[:c] + "a" + tempnum[c+1:]
            if addex in [1,3]: tempnum = "a" + tempnum
            if addex in [2,3]: tempnum = tempnum + "a"
            testlist.append([tempnum,a])
    #We've now generated a test set of test data, expected answer.
    correct = 0
    dupedict = {}
    guesslist = [bestnummatch(a[0],numlist) for a in testlist]
    #singleGuesses = [a[0][0] for a in guesslist]
    singleGuesses = []
    for a in range(len(guesslist)):
        try: adjcompare = [guesslist[a-1],guesslist[a+1]]
        except:
            try: adjcompare = [guesslist[a-1]]
            except: adjcompare = [guesslist[a+1]]
        if len(guesslist[a][0]) > 1:
            guessed = False
            for b in adjcompare:
                if b in guesslist[a][0]:
                    singleGuesses.append(b)
                    guessed = True
            if not guessed:
                singleGuesses.append(guesslist[a][0][0])
        else:
            singleGuesses.append(guesslist[a][0][0])
    
    correct = 0
    for a in range(len(testlist)):
        #print(len(singleGuesses),len(testlist))
        if singleGuesses[a] == testlist[a][1]:
            correct += 1
    print("Run: Correctly guessed {} of {} ({:.2f}%)".format(correct,len(testlist),correct / len(testlist)))
    #print("Duplicate counts: {}".format(dupedict.keys()))
    return correct / len(testlist)

