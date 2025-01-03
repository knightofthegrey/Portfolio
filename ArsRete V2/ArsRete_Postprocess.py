#ArsRete_Postprocess
#Analyze runs after the fact

'''
Changelog:
3/1/24: Initial version. Parser for dump files, and group analysis tools for grabbing data from multiple linked runs and comparing them.
'''

import numpy as np
import re
import os
import ArsRete_Datasets as ARD
import matplotlib.pyplot as plt

#__FUNCTIONS__

def loadDump(filename):
    #Quick function to read dump text files from an ArsRete GAN run.
    #This is fine-tuned to the current format and if the format changes it will need to be rewritten.
    
    #Input parser to account for me having a hard time remembering to put prefixes and suffixes on my function calls
    found = False
    for a in [filename,filename + ".txt","AROut/"+filename, "AROut/"+filename+".txt"]:
        if not found: 
            try:
                words = [b.split("|") for b in open(a,"r").read().splitlines()[12:]]
                found = True
            except:
                pass
    print(found)
    if found:
        return words
    else:
        return []

def loadNewDump(filename):
    #Quick function to read dump text files from an ArsRete GAN run.
    #This uses the ArsRete_Model dump format
    #Outputs are a dictionary of: "Words":2d list of words by batch, anything else:[list of floats by batch,list of floats by epoch]
    out_dict = {}
    temp = [b for b in open(filename,"r").read().split("\n\n") if b != ""][3:]
    for b in temp:
        sub_b = b.split("\n")[1:]
        for c in sub_b:
            index = c.split(": ")[0]
            if index == "Words":
                subdata = [d.split("|") for d in c.split(": ")[1].split("||")]
                if index in out_dict:
                    out_dict[index].append(subdata)
                else:
                    out_dict[index] = [subdata]
            else:
                subdata = [float(d) for d in c.split(": ")[1].split("|")]
                if index in out_dict:
                    out_dict[index][0] += subdata
                    out_dict[index][1].append(sum(subdata)/len(subdata))
                else:
                    out_dict[index] = [subdata,[sum(subdata)/len(subdata)]]
    return out_dict
    

def loadGroups(prefix,loopdexes):
    bi = batchIndex(loopdexes)
    fn = namegroups(prefix,bi)
    
    tot_sublists = {a:{b:[] for b in fn[a]} for a in fn}
    avg_sublists = {a:{b:{} for b in fn[a]} for a in fn}
    for a in fn:
        for b in fn[a]:
            for c in fn[a][b]:
                tot_sublists[a][b].append(loadNewDump(c))
    
    for a in tot_sublists:
        for b in tot_sublists[a]:
            indexes = [c for c in tot_sublists[a][b][0] if c != "Words"]
            sub_indexlists = [lineAvg([c[d][0] for c in tot_sublists[a][b]]) for d in indexes]
            
            for d in range(len(indexes)):
                avg_sublists[a][b][indexes[d]] = sub_indexlists[d]
    
    return avg_sublists,tot_sublists
            
def spg(filename,refset):
    t1 = [a for b in loadNewDump(filename)["Words"] for a in b]
    return [ARD.setpro(a,refset) for a in t1]

def graphDump(filename,words,gibberish):
    g = [[],[],[]]
    for a in loadDump(filename):
        t = ARD.Compare(a,words,gibberish)
        for b in range(3):
            g[b].append(t[b])
    
    for i,a in enumerate(["Lfreq","M1","M2"]):
        plt.plot(g[i],label = a)
        plt.plot(trendLine(g[i],10),label = "{} avg".format(a))
    
    plt.legend()
    plt.show()

def namegroups(prefix,fileindexes):
    #Function to take a prefix and set of fileindexes, and construct from them which numbered files are of which index
    out_groups = {}
    for a in range(len(fileindexes[0])):
        out_groups[a] = {}
        for b in range(len(fileindexes)):
            if fileindexes[b][a] in out_groups[a]:
                out_groups[a][fileindexes[b][a]].append("{} {}.txt".format(prefix,b))
            else:
                out_groups[a][fileindexes[b][a]] = ["{} {}.txt".format(prefix,b)]
    
    return out_groups

def subgroups(fileouts,fileindexes):
    #Function to take a list of file outputs, group them into sub-groups using batchIndex, and average the sub-groups
    #Inputs are a list of lists of floats generated by ARD.Compare() and other functions
    #fileindexes should be the output of batchIndex, and should have the same length as fileouts
    num_g = len(fileindexes)
    out_g = {} #Dictionary of possible groups, indexed by group index:group value
    
    for a in range(len(fileouts)):
        #For each file, we want to add it to its value in each of fileindexes[a]
        indexes = fileindexes[a]
        for b in range(len(indexes)):
            #For each index level, check if it's already in out_g
            if b in out_g:
                #If it is, check if the value is in the sub-dict
                if indexes[b] in out_g[b]:
                    #If it is, append
                    out_g[b][indexes[b]].append(fileouts[a])
                #Otherwise, make a new value in out_g[b]
                else:
                    out_g[b][indexes[b]] = [fileouts[a]]
            #If it isn't, make a new entry in out_g
            else:
                out_g[b] = {indexes[b]:[fileouts[a]]}
    
    #Once we've grouped all indexes, we need to average them
    for a in out_g:
        #For each index level:
        for b in out_g[a]:
            #For each index value at that index level:
            #Let's be safe and future-proof this against different implementations with different numbers of data fields
            #Complex line breakdown:
            '''
            [out_g[a][b][c][d] for c in range(len(out_g[a][b]))] = data table d for each line in out_g[a][b]
            [lineavg(that) for d in range(len(out_g[a][b]))] = element-wise average of that metric for each metric
            '''
            #Aaand...this immediately broke!
            #Obviously.
            sub_lists = [out_g[a][b][c] for c in range(len(out_g[a][b]))]
            avg_list = [lineAvg([out_g[a][b][c][d] for c in range(len(out_g[a][b]))]) for d in range(len(out_g[a][b]))]
            out_g[a][b]["Avg"] = avg_list
    
    return out_g

def batchIndex(li):
    #Reverse-engineers an absolute index:sub-groups mapping
    #Run sets are made using nested for loops, which means we can go backwards and get what the state of each for loop was when each number was produced.
    #I don't know if I've explained this well, so, an example:
    '''
    batchIndex([3,3]):
    
    c = 0
    for a in range(3): for b in range(3): c++
    Gives us qi of [[3,3],[1,3]]
    Produces --> Reverses to:
    a,b,c        c              a             b
    0,0,0        0-> (0//3)%3 = 0, (0//1)%3 = 0
    0,1,1        1-> (1//3)%3 = 0, (1//1)%3 = 0
    0,2,2        2-> (2//3)%3 = 0, (2//1)%3 = 0
    1,0,3        3-> (3//3)%3 = 1, (3//1)%3 = 0
    1,1,4        4-> (4//3)%3 = 1, (4//1)%3 = 0
    1,2,5        5-> (5//3)%3 = 1, (5//1)%3 = 0
    2,0,6        6-> (6//3)%3 = 2, (6//1)%3 = 0
    2,1,7        7-> (7//3)%3 = 2, (7//1)%3 = 0
    2,2,8        8-> (8//3)%3 = 2, (8//1)%3 = 0
    
    Returns [[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]]
    '''
    
    qi = [[int(np.prod(li[b+1:])),li[b]] for b in range(len(li))]
    return [[(a//qi[b][0]) % qi[b][1] for b in range(len(qi))] for a in range(np.prod(li))]    

#__FUNCTIONS FOR PROCESSING NUMERICAL DATA__

def lineAvg(in_lists):
    #Element-wise averages several lists of floats
    #Inputs must be of uniform length
    out_l = []
    for a in range(len(in_lists[0])):
        out_l.append(sum([in_lists[b][a] for b in range(len(in_lists))]) / len(in_lists))
    return out_l

def stdDev(in_list):
    mean = sum(in_list) / len(in_list)
    var = [(a-mean)**2 for a in in_list]
    return (sum(var) / len(var)) ** 2

def trendLine(in_list,trend_window):
    #This function returns a list containing a running average of the last n values
    #It is a simple visualization tool for smoothing choppy results and getting a sense of where they're going
    #It shouldn't be relied upon within (trend_window) of the y-axis of your graph, since the denominator is changing there to avoid running off the edge.
    #
    '''
    Inputs:
    in_list: List of floats, data to be smoothed
    trend_window: Int, number of prior samples to average at each value
    '''
    out_list = []
    for a in range(len(in_list)):
        if a < trend_window:
            out_list.append(sum(in_list[:a+1]) / (a+1))
        else:
            out_list.append(sum(in_list[a-trend_window:a+1]) / (trend_window + 1))
    return out_list

def stateCompare(state1,state2):
    #Compare NN save states to see which values change and which saturate
    #First, read in two states
    d1 = [a.split("\n") for a in open(state1,"r").read().split("\n\n")]
    d2 = [a.split("\n") for a in open(state2,"r").read().split("\n\n")]
    
    d1_nums = [[[float(b) for b in c.split(" ")] for c in a[1:]] for a in d1]
    d2_nums = [[[float(b) for b in c.split(" ")] for c in a[1:]] for a in d2]
    
    deltas = []
    
    for a in range(len(d1_nums)):
        delta_a = []
        for b in range(len(d1_nums[a])):
            delta_a.append([d2_nums[a][b][c] - d1_nums[a][b][c] for c in range(len(d1_nums[a][b]))])
        deltas.append(delta_a)
    
    #Problem: How do we use this information?
    
    z_c = 0
    a_c = 0
    d_c = 0
    t_c = 0
    for a in deltas:
        for b in a:
            for c in b:
                t_c += 1
                if c == 0: z_c += 1
                a_c += abs(c)
                d_c += c
                
    
    return [z_c/t_c,a_c/t_c,d_c/t_c]

def stateProcess(prefix,index,out):
    #Compare NN save states, and remove the excess files to clean up disk space
    outf = open(out,"w")
    outd = [[],[],[]]
    for a in range(index-1):
        print(a)
        t1 = stateCompare("{} {}.txt".format(prefix,a),"{} {}.txt".format(prefix,a+1))
        for b in range(3):
            outd[b].append(t1[b])
    
    for a in range(index):
        os.remove("{} {}.txt".format(prefix,a))
    
    for i,a in enumerate(["Zeros","Abs","Total"]):
        outf.write(a + "|" + " ".join([str(b) for b in outd[i]]))
        if i != 2: outf.write("\n")
    
    outf.close()

def wordFilter(in_file,words,how_many,proscore = 2):
    cussfilter = "fuck|cunt|shit|nigg"
    #And then:
    gen = loadNewDump(in_file)["Words"]
    test_m = ARD.uniqueM(words)
    test_v = ARD.vcpPrev(words)
    
    out_words = []
    
    n = len(gen) - 1
    
    while True:
        flatn = [a for b in gen[n] for a in b]
        for a in flatn:
            if ARD.proscore(a,test_m,test_v,0.01) >= proscore:
                if not re.search(cussfilter,a):
                    if a not in words:
                        out_words.append(a)
        if len(out_words) > how_many:
            return out_words
        else:
            n = n-1
            if n < 0:
                return out_words
        
    
    
        
def progressBar(percent,width = 20):
    #Returns a string representing a progress bar
    #Uses unicode █ U+2588 (filled-in box) for the purpose
    return "|"+"█"*int(width*percent)+"_"*(width - int(width*percent)) + "|"