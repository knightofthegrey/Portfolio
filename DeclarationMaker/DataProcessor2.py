#DataProc2
#This takes functions from the original DataProc and cleans them up for use

import re
import os
import ConstantWidthTable
import Utilities

def searchInput(field,pattern,debug = False):
    #This function tries to read all files in the inputData folder to see if any of them have the right parameter
    #field is the name of the field, pattern is the thing the data in that field should match
    for file in os.listdir("inputData"):
        if file.split(".") == "txt":
            try:
                tempData = newParser(file)
                for case in tempData:
                    if field in case.keys():
                        if re.search(pattern,case[field]):
                            print("Found match in:",file,"at",field,":",case[field])
            except:
                pass

def debugTable(inFile,debug = False):
    #Parses a raw text file and creates a txt file containing the parsed data for debugging purposes
    inData = newParser(inFile)
    outData = open("debugTable.txt","w")
    #Create a list of keys that represents all dictionaries in inData
    masterKeyList = []
    for line in inData:
        for key in line.keys():
            if key not in masterKeyList:
                masterKeyList.append(key)
    #Find the maximum width of each column for uniform formatting width
    widthTable = [len(x) for x in masterKeyList]
    for case in inData:
        for x in range(len(widthTable)):
            if masterKeyList[x] in case.keys():
                widthTable[x] = max(widthTable[x],len(case[masterKeyList[x]]))
    #Write the header data to the file
    outData.write(ConstantWidthTable.makeLine(masterKeyList,masterKeyList,widthTable," \ "))
    outData.write("\n")
    #Write the table data to the file
    for case in inData:
        outData.write(ConstantWidthTable.makeLine(case,masterKeyList,widthTable," \ "))
        outData.write("\n")
    outData.close()
    

def readInputFile(inFile,debug = False):
    #This is an older, more primitive approach to this function. It has a lot of hardcoded indexes and produces a list of lists that need to be further
    #refined into a dictionary
    #This function reads a text file of PAMAR output, clears out excess space, matches headers to data, and returns the results as a list
    #Input is the name of a text file
    #This line separates the input data into cases using the search screen as a separator
    #Skip the last one, because that's the search screen at the end of the file
    inputData = open("inputData/" + inFile,"r").read().split("MERCHANTS CREDIT ASSOCIATION\n legal update & maint v04 ")[:-1]
    outData = [["FHeader","FData","FCosts","InfoP","InfoD","InfoDocs","InfoSvc","LastGInfo","LastGPoE","LastGDB","LastGCosts","FullGTable","Memos"]]
    for case in inputData:
        #Split the case into pages
        tempData = case.split("selection ?")
        caseData = []
        caseDict = {}
        #We really need to identify screens
        #for screen in tempData:
            #screenParse(caseDict,screen)
        #()
        #We know we're going to see a front page with 67 lines, an info page with 84 lines, a garn page with 55 lines, and a long memo page?
        recordG = False
        gTable = []
        for block in tempData:
            tempList = block.split("\n")
            if len(tempList) == 67:
                #Front page, run front page separators
                caseData.append(tempList[:7])
                frontpageData = []
                #Table will include identifiers and data at specific indexes
                for x in range (7,11):
                    frontpageData.append([tempList[x],tempList[25+x]])
                for x in range (12,32):
                    frontpageData.append([tempList[x],tempList[24+x]])
                caseData.append(frontpageData)
                caseData.append(tempList[56:])
            elif len(tempList) == 84:
                #Case header information and address information from the info page
                caseData.append(tempList[4:8])
                caseData.append(tempList[10:14])
                caseData.append(tempList[16:20])
                caseData.append(tempList[22:27])
                recordG = True #Pages after this one but before len55 are garnishment tables
            elif len(tempList) == 55:
                #Garn page, run garn info separators
                #Yes, these are awkward and silly, but too much of that page is out of order
                caseData.append([[tempList[a],tempList[b]] for a,b in [[1,29],[2,31],[3,32],[4,30],[5,33],[6,34],[7,35],[8,36]]])
                caseData.append([[tempList[a],tempList[b]] for a,b in [[15, 40], [17, 41], [19, 42], [21, 43]]])
                caseData.append([[tempList[a],tempList[b]] for a,b in [[11, 38], [13, 39], [26, 51], [27, 52]]])
                caseData.append([[tempList[a],tempList[b]] for a,b in [[12, 44], [14, 45], [16, 46], [18, 47], [20, 48], [22, 49], [24, 50]]])
                caseData.append(gTable)
                recordG = False #Garnishment tables are done
            elif len(tempList) > 50:
                #The memo page is of variable length, so we use re to identify the lines that are memos and save them
                memolist = []
                for line in tempList:
                    if re.search("^ {,1}\d+ \d\d\-\d\d\-\d\d\D \d\d:\d\d",line):
                        memolist.append(line)
                caseData.append(memolist)
            elif recordG:
                #Garnishment table, if we're seeing this we're in a block in between info and garnishment
                gTable += tempList[2:-2] #Skip the first and last two lines of these blocks
        outData.append(caseData)
    #Output from this function is a list of 12-element lists, element 0 is the identifiers for what the 12 items are in the others.
    return outData

def parseWrapper(inFile,batchnum):
    #This is a quick wrapper used to keep the dictionaries in sync with the output folders if output folders change
    return [a for a in newParser(inFile) if a["FolderName"] in os.listdir("outputData/Batch{}".format(batchnum))]

def newParser(inFile,debug = False,skips = []):
    #This is the currently working parser
    #inFile is a text file containing PAMAR output, skips are indexes to skip for debug purposes
    outList = []
    #Here we split by the search screen, which guarantees outData will contain individual cases
    outData = open("inputData/{}".format(inFile),"r").read().split("MERCHANTS CREDIT ASSOCIATION\n legal update & maint v04 ")[:-1]
    for x in range(len(outData)):
        #print(x)
        if x not in skips:
            case = outData[x]
            caseDict = {}
            #Here we split each case into sections by menu selection, then parse each individual screen
            for screen in case.split("selection ?"):
                caseDict = screenParse(caseDict,screen)
            #Here we reformat, convert, and double-check the contents of caseDict
            caseDict = postParse(caseDict)
            outList.append(caseDict)
    #print(len(outList))
    return outList

def postParse(inDict):
    #This is a postprocessing setup to reformat the newParser data to match data produced by the original parseInputFile function
    #Things plug into DeclarationMaker in specific ways, and this makes sure they all fit
    outDict = inDict #I'm not 100% on this being important.
    #print(outDict.keys())
    if "g/def1" in outDict.keys():
        #If the data we're running includes a garnishment screen, do quick reformats of the data we found on the garnishment screen
        outDict["DtSvc"] = outDict["svc"]
        outDict["Garnishee"] = outDict["g/def1"] + "|" + outDict["g/def2"]
        if outDict["Garnishee"][0] == "|": outDict["Garnishee"] = outDict["Garnishee"][1:]
        if outDict["Garnishee"][-1] == "|": outDict["Garnishee"] = outDict["Garnishee"][:-1]
        outDict["AddrG"] = outDict["g/addr"] + ", " + outDict["g/csz"]
        for key in ['file','svcX','ans', 'atty', 'parte', 'TOTAL']: outDict[key[0].upper() + key[1:].lower() + "C"] = outDict[key]
        outDict["MailC"] = outDict["c/mail"]
        outDict["Svcst1"] = outDict["SvcxC"]
        if outDict["TotalC"][-2:] == "43":
            outDict["Svcst2"] = "8.05"
            outDict["Svcst3"] = "1.53"
        else:
            outDict["Svcst2"] = outDict["SvcxC"]
            outDict["Svcst3"] = str(round(float(outDict["MailC"]) - float(outDict["SvcxC"]),2))
        outDict["Svcst4"] = str(round(float(outDict["MailC"]) + float(outDict["SvcxC"]),2))
        outDict["SubtotalC"] = str(round(float(outDict["TotalC"]) - float(outDict["AttyC"]),2))
        #Double check to make sure we don't have any strings stored with just one decimal place here
        for element in ["FileC","SvcxC","AnsC","AttyC","MailC","ParteC","TotalC","Svcst1","Svcst2","Svcst3","Svcst4","SubtotalC"]:
            if re.search("\.\d$",outDict[element]):
                outDict[element] += "0"
    #print(outDict.keys())
    try:
        if "IntA" in outDict.keys() and outDict["IntA"]: outDict["IntPost"] = floatStrMath(outDict["j/intA"],outDict["IntA"])
        else: outDict["IntPost"] = "0"
    except:
        outDict["IntPost"] = "0"
    try:
        outDict["currentCosts"] = floatStrMath(floatStrMath(outDict["totalA"],outDict["j/intA"]),outDict["princA"])
    except:
        outDict["currentCosts"] = "0"
    try:
        outDict["DtExp"] = outDict["expires"].replace("-","")
    except:
        pass
    #Get the venue county, level, division out of the venue line
    v = outDict["venue"].replace(",","").split(" ")
    leveldict = {"DC":"DISTRICT","SC":"SUPERIOR"}
    if v[0] in leveldict: outDict["TestLevel"] = leveldict[v[0]]
    else: outDict["TestLevel"] = "NOT WA"
    if "DIVISION" in v: outDict["TestDivision"] = v[1]
    else: outDict["TestDivision"] = ""
    if len(v) == 3: outDict["County"] = v[1] #Case: DC County WA
    elif len(v) == 4: outDict["County"] = "{} {}".format(v[1],v[2]) #Case: DC County County WA
    elif len(v) == 5: outDict["County"] = v[3] #Case: DC X Division County WA
    elif len(v) == 6: outDict["County"] = "{} {}".format(v[3],v[4]) #Case: DC X Division County County WA
    elif len(v) == 7: outDict["County"] = "KING" #Case: DC X Division, Y Courthouse King WA
    if v[0][0] == "D": outDict["Level"] = "DISTRICT"
    else: outDict["Level"] = "SUPERIOR"
    if v[2].replace(",","") == "DIVISION": outDict["Division"] = v[1]
    else: outDict["Division"] = ""
    #outDict["County"] = v[-2]
    #countyFixDict = {"HARBOR": "GRAYS HARBOR","OREILLE":"PEND OREILLE","JUAN":"SAN JUAN", "WALLA":"WALLA WALLA"}
    #if outDict["County"] in ["HARBOR","OREILLE","JUAN","WALLA"]: outDict["County"] = countyFixDict[outDict["County"]]
    outDict["DtJ"] = outDict["jdmt dt"].split(" ")[0].replace("-","")
    intRaw = outDict["jdmt rt"].split(" ")[0]
    #print(intRaw)
    try:
        if "(" in intRaw: outDict["IntR"] = "0.0"
        else: outDict["IntR"] = str(float(intRaw.split(".")[1][:2] + "." + intRaw.split(".")[1][2:]))
    except:
        pass
    #outDict["IntR"] = str(int(outDict["jdmt rt"].split(" ")[0].split(".")[1].replace(")",""))).replace("00","")
    #print(outDict["IntR"])
    #Get fees info from programData
    outDict = feeRead(outDict)
    if "ExtFee" in outDict.keys(): outDict["ExtBalance"] = floatStrMath(outDict["totalB"],outDict["ExtFee"],2)
    if "ExtFee" in outDict.keys() and "ExtBalance" in outDict.keys():
        outDict["ExtBalance"] = floatStrMath(outDict["totalB"],outDict["ExtFee"],2)
    for key in outDict.keys():
        if "Dt" in key:
            outDict[key] = outDict[key].replace("-","")
    if "DtExp" in outDict.keys() and "DtJ" in outDict.keys():
        if outDict["DtExp"] == "" and len(outDict["DtJ"]) == 6:
            outDict["DtExp"] = outDict["DtJ"][:4] + str(int(outDict["DtJ"][4]) + 1) + outDict["DtJ"][5]
    try: outDict["AddrG"] = outDict["AddrG"].replace("208 MC","208 MC-CSC1").replace("C1-C","C1")
    except: pass
    #Longdefname:
    #print(outDict["Defendant"])
    if "Defendant" in outDict.keys():
        if "DOE" in outDict["Defendant"] and "DefS" in outDict.keys(): outDict["LongDef"] = outDict["DefS"]
        else:
            try:
                #print(outDict["Defendant"])
                tempname = []
                for word in outDict["Defendant"].replace("|"," ").split(" "):
                    if not any(x.islower() for x in word):
                        tempname.append(word)
                tempname = " ".join(tempname).replace("HIS WIFE","").replace("HER HUSBAND","")
                if "," in tempname:
                    tempname = tempname.split(" ")
                    namewordlist = []
                    lastlist = []
                    index = -1
                    for x in range(len(tempname)):
                        if "," in tempname[x]:
                            namewordlist.append([])
                            lastlist.append([tempname[x][:-1]])
                            index += 1
                        else:
                            namewordlist[index].append(tempname[x])
                    akaflag = -1
                    for x in range(len(namewordlist)):
                        if namewordlist[x][-1] == "AKA":
                            akaflag = x + 1
                            namewordlist[x] = " ".join(namewordlist[x][:-1] + lastlist[x])
                        else: namewordlist[x] = " ".join(namewordlist[x] + lastlist[x])
                    if akaflag != -1:
                        namewordlist.pop(akaflag)
                    tempname = " and ".join(namewordlist)
                outDict["LongDef"] = tempname
            except:
                outDict["LongDef"] = "DEF NAME ERROR"
    else:
        if "," not in outDict["topdefs"][0]: outDict["LongDef"] = outDict["topdefs"][0]
        else: outDict["LongDef"] = outDict["topdefs"][0].split(",")[1] + " " + outDict["topdefs"][0].split(",")[0]
    #print(outDict["Legal"])
    try:
        try: outDict["FolderName"] = outDict["Last"][0] + outDict["Last"][1:].lower() + " " + outDict["case#"]
        except: outDict["FolderName"] = outDict["LongDef"].split(" ")[-1][0] + outDict["LongDef"].split(" ")[-1][1:].lower() + " " + outDict["case#"]
    except:
        outDict["FolderName"] = outDict["case#"]
    return outDict

def screenParse(inDict,inScreen,debug = False):
    #This program takes an inputted string, categorizes it, and sends it to the appropriate parser
    tempScreen = inScreen.split("\n")
    outDict = inDict
    #print(tempScreen)
    #print("Screen IDs: {} {}".format(tempScreen[0],tempScreen[1]))
    #Screen Identifiers: blank, then day of the week means it's safe to ignore
    if re.search("^\D+ +\d\d:\d\d\D\D",tempScreen[1]):
        #print("Search screen leftovers")
        pass
    elif re.search("^ *\d+$",tempScreen[0]) and re.search("^\d+ +L\d+ P\d+$",tempScreen[1]):
        #print("Front screen")
        #outDict = parseFrontScreen(outDict,tempScreen)
        outDict = newFrontParser(outDict,tempScreen)
    elif tempScreen[0] == " 2" and tempScreen[1] == "":
        #Found front screen via case search
        outDict = newFrontParser(outDict,tempScreen[3:])
    elif tempScreen[0] == " i":
        #print("Info screen")
        outDict = parseInfoScreen(outDict,tempScreen)
    elif tempScreen[0] == " g":
        #print("Garn index")
        pass
    elif re.search("^ *\d+$",tempScreen[0]) and tempScreen[1] == "garn# ":
        #print("Garn screen")
        outDict = parseGarn(outDict,tempScreen)
    elif re.search("^seq   garn#",tempScreen[1]):
        #print("Duplicate garn index")
        pass
    elif re.search("garn options",tempScreen[1]):
        #print("Duplicate garn index?")
        pass
    elif tempScreen[0] == " n":
        outDict = parseNotes(outDict,tempScreen)
    elif tempScreen == [" e",""]:
        #print("Junk screen")
        pass
    elif tempScreen[0] == " e":
        #print("Duplicate front screen")
        pass
    else:
        #print("Unknown screen:", tempScreen[:4])
        pass
    return outDict

def newFrontParser(inDict,frontScreen):
    #Parses the front info screen
    #Most of this consists of figuring out how the list of keys and the list of values match up, both are printed on screen but not in the same order.
    #print(len(frontScreen))
    seps = []
    outDict = inDict
    #print(len(inDict.keys()))
    for x,line in enumerate(frontScreen):
        #Separating by boxes:
        if line == "dsk": seps.append(x)
        elif line == "spc/instr": seps.append(x+1)
        elif line.strip() == "assessed      paid   balance": seps.append(x)
    topbox = frontScreen[:seps[0]]
    #print(topbox)
        #Main problem: 2 defendants pushes addresses to another line
    outDict["Legal"] = topbox[1].split("L")[1].split(" ")[0]
    outDict["Packet"] = topbox[1].split("P")[1]
    outDict["topdefs"] = [topbox[2].split("-")[-1][2:]]
    if not any(char.isdigit() for char in topbox[3]): 
        outDict["topdefs"].append(topbox[3].strip())
        outDict["AddrD"] = ", ".join([topbox[4].strip(),topbox[5].strip()])
    else:
        outDict["AddrD"] = ", ".join([topbox[3].strip(),topbox[4].strip()])
    #Not worrying about phone numbers right now
    headerbox = frontScreen[seps[0]:seps[1]]
    databox = frontScreen[seps[1]:seps[2]]
    #protoHeader = ["dsk", "init dt", "file dt", "rtn-   ", "0  ", "orig sv", "last sv", "expires", "l/acty ", "stat dt", "jdmt dt", "jdmt rt", "case# ", "venue ", "c/addr", "p/atty", "a/atty", "cause ", "memos ", "debts", "notes", "garns", "f/date", "f/desc", "spc/instr"]
    #Headers are always the same 25 things, but we don't care about header3 or header4.
    headerbox = headerbox[:3] + headerbox[5:]
    databox = databox[:3] + databox[4:]
    if databox[6].strip() == "expired": databox = databox[:6] + databox[7:]
    #print(len(headerbox))
    #print(len(databox))
    for x,line in enumerate(headerbox):
        if x < len(databox):
            outDict[line.strip()] = databox[x].strip()
            #print(line.strip(), databox[x].strip())
    finbox = [[x for x in a.split(" ") if x != ""] for a in frontScreen[seps[2]:][:8]]
    for a in finbox[1:]:
        for b in range(1,4):
            outDict[a[0].lower() + finbox[0][b-1][0].upper()] = a[b]
    #print(len(outDict.keys()))
    return outDict

def parseFrontScreen(inDict,frontScreen):
    #Old front parser, breaks on some test cases
    #print(len(frontScreen))
    #print("In FrontScreen: Len {}".format(len(frontScreen)))
    outDict = inDict
    topData = frontScreen[:7]
    outDict["Legal"] = topData[1].split("L")[1].split(" ")[0]
    outDict["Packet"] = topData[1].split("P")[1]
    #print(len(frontScreen))
    if len(frontScreen) == 67:
        #print("Parsed as 67")
        headers = frontScreen[7:11] + frontScreen[12:32]
        for x in range (len(headers)):
            #if len(frontScreen) != 67: print(headers[x].strip(),frontScreen[32+x].strip())
            outDict[headers[x].strip()] = frontScreen[32+x].strip()
        if "assessed" in frontScreen[55]:
            frontTable = [[x for x in a.split(" ") if x != ""] for a in frontScreen[55:63]]
        else:
            frontTable = [[x for x in a.split(" ") if x != ""] for a in frontScreen[56:64]]
        for a in frontTable[1:]:
            for b in range(1,4):
                outDict[a[0].lower() + frontTable[0][b-1][0].upper()] = a[b]
    elif len(frontScreen) == 68: #68
        #print("Parsed as 68")
        headers = frontScreen[7:10] + frontScreen[12:32]
        data = frontScreen[32:39] + frontScreen[41:57]
        for x in range(len(headers)):
            outDict[headers[x].strip()] = data[x].strip()
        frontTable = [[x for x in a.split(" ") if x != ""] for a in frontScreen[57:65]]
        for a in frontTable[1:]:
            for b in range(1,4):
                outDict[a[0].lower() + frontTable[0][b-1][0].upper()] = a[b]
    elif len(frontScreen) == 66:
        #print("Parsed as 66")
        headers = frontScreen[7:10] + frontScreen[12:31]
        data = frontScreen[32:35] + frontScreen[36:55]
        for x in range(22): outDict[headers[x].strip()] = data[x].strip()
        frontTable = [[x for x in a.split(" ") if x != ""] for a in frontScreen[55:63]]
        for a in frontTable[1:]:
            for b in range(1,4):
                outDict[a[0].lower() + frontTable[0][b-1][0].upper()] = a[b]
        #for x in range(25): print(x,rawHeaders[x],rawData[x])
    else:
        "Not parsed as 66, 67, or 68"
        print("New length! {}".format(len(frontScreen)))
        
    return outDict

def parseInfoScreen(inDict,infoScreen):
    #Text parser for infoscreen
    #print("In InfoScreen: Len {}".format(len(infoScreen)))
    outDict = inDict
    pblock = infoScreen[4:9]
    pdata = pblock[0][5:].strip()
    if "CORPORATION" in pdata and pdata[-1] != ",": pdata += ","
    for line in pblock[1:]:
        if line.strip() != "":
            pdata += "|{}".format(line[5:].strip())
    if pdata[-1] == ",": pdata = pdata[:-1]
    outDict["Plaintiff"] = pdata
    dblock = infoScreen[10:15]
    ddata = dblock[0][5:].strip()
    for line in dblock[1:]:
        if line.strip() != "":
            ddata += "|{}".format(line[5:].strip())
    if ddata[-1] == ",": ddata = ddata[:-1]
    outDict["Defendant"] = ddata
    docblock = infoScreen[16:20]
    svcblock = infoScreen[22:27]
    outDict["DefS"] = (svcblock[0][4:].strip().split(",")[1] + " " + svcblock[0][4:].strip().split(",")[0]).replace("  "," ")
    if " " in svcblock[0][4:].strip().split(", ")[0]: outDict["Last"] = svcblock[0][4:].strip().split(", ")[0].split(" ")[0]
    else: outDict["Last"] = outDict["DefS"].split(" ")[-1]
    outDict["OrigAddr"] = svcblock[1][4:].strip() + ", " + svcblock[2][4:].strip()
    outDict["Aka"] = svcblock[4][4:].strip()
    dupdblock = infoScreen[27:33]
    #print("We got to the end of the screen section!")
    return outDict

def parseGarn(inDict,garnScreen):
    #Text parser for garn screen
    #These are out of order, some in the same way the frontscreen are, but unlike the frontscreen they're consistently out of order
    outDict = inDict
    indexes = [1,2,3,4,5,6,7,8,15,17,19,21,11,13,26,27,12,14,16,18,20,22,24]
    datalist = [29,31,32,30,33,34,35,36,40,41,42,43,38,39,51,52,44,45,46,47,48,49,50]
    for x in range (len(indexes)):
        if garnScreen[indexes[x]].strip() not in outDict.keys():
            outDict[garnScreen[indexes[x]].strip()] = garnScreen[datalist[x]].strip()
        else:
            outDict[garnScreen[indexes[x]].strip() + "X"] = garnScreen[datalist[x]].strip()
    return outDict

def parseNotes(inDict,notesScreen):
    #Parses notes screens. Notes are in the format noteindex date time info
    outDict = inDict
    #poeCleanup,dbCleanup,dateOfWrit,intA,Dt2A,Dt2R,Fnds
    tempData = []
    for line in notesScreen:
        if re.search("^ {,1}\d+ \d\d\-\d\d\-\d\d\D \d\d:\d\d",line):
            tempData.append(line)
    memoData = processNotes(tempData)
    indexes = ["MailnoG","MailnoD","DtMail","IntA","Dt2ans","Dt2R","Fnds"]
    for x in range(len(indexes)):
        outDict[indexes[x]] = memoData[x]
    return outDict




def packetParser(inFile,debug = False,skips = []):
    #Replicates the newParser above, but for the packet interface rather than the legal
    outList = []
    outData = open("inputData/{}".format(inFile),"r").read().split("MERCHANTS CREDIT ASSOCIATION\n online collector v04.02")[:-1]
    #print(len(outData))
    for x in range(len(outData)):
        if x not in skips:
            case = outData[x]
            caseDict = {}
            #Split each case into sections by menu selection, then parse each individual screen
            screens = [case.split("option ? ")[0]] + (" ".join(case.split("option ? "))).split("selection ? ")
            #print(len(screens))
            for screen in screens:
                caseDict = packetScreen(caseDict,screen)
            #print()
            caseDict = packetPost(caseDict)
            outList.append(caseDict)
    return outList

def packetScreen(inDict,inScreen,debug = False):
    #Identifies each screen
    outDict = inDict
    tempScreen = inScreen.split("\n")
    #print(inScreen.replace("\n"," "))
    #Screen identifiers:
    #print(tempScreen[3])
    #With current screen breaks?
    if tempScreen[0] == "d" and "demo" in tempScreen[3]:
        #print("Demos: ",inScreen.replace("\n"," "))
        outDict = fieldScreen(outDict,tempScreen,"d")
    elif tempScreen[0] == "5" or (tempScreen[0] == "d" and "emp" in tempScreen[3]):
        #print("Employer: ",inScreen.replace("\n"," "))
        outDict = fieldScreen(outDict,tempScreen,"e")
    elif re.search("^\d+$",tempScreen[0]) and len(tempScreen[0]) > 3:
        #print("Intro screen: ",inScreen.replace("\n"," "))
        outDict["Packet"] = tempScreen[0]
    elif tempScreen[0] in ["e","11"]:
        #print("Exiting somewhere we've been already")
        pass
    else:
        #print("Unknown")
        pass
    return outDict

def packetPost(inDict):
    outDict = inDict
    #print()
    return outDict

def fieldScreen(inDict,inScreen,screenType):
    #Screenparser for the demographics screen
    outDict = inDict
    dnum = ""
    for line in inScreen:
        if "debtor (1/2) ? " in line:
            dnum = line.split(" ")[-1]
        #print(line)
        if re.search("^ +\d+ +\D",line):
            linebreakdown = re.split("\.\.+",line) #[a for a in line.split(".") if a != ""]
            if len(linebreakdown) != 2: print("Demo screen parse error: Extraneous periods",line)
            else:
                index = " ".join([a for a in linebreakdown[0].split(" ") if a != ""][1:]) + " " + screenType + dnum
                data = " ".join([a for a in linebreakdown[1].split(" ") if a != ""])
                outDict[index] = data
    return outDict

def parseInputFile(inFile,debug = False):
    #Original parser, combines the functions of the new parser and postprocesser
    inData = readInputFile(inFile,debug)
    outList = []
    #We really need to ID screens
    for case in inData[1:]:
        #print(len(case))
        if len(case) != 13:
            pass
        else:
            #print()
            #print(case)
            #print(inData[0][2])
            caseDict = {}
            #Identifying properties of block 0:
            #"^ *\d+ *$" - Irrelevant, duplicate data
            unmatched = []
            for element in case[0]:
                #Header data section
                if re.search("^\d+ +L\d+ P\d+",element):
                    #This is the p/d line
                    caseDict["Legal"] = element.split(" ")[2][1:] #Strip leading L
                    caseDict["Packet"] = element.split(" ")[3][1:] #Strip leading P
                #The next line is access data and a name
                elif re.search("^\D+ +\d\d:\d\d",element):
                    caseDict["DName1"] = re.split("\d\d\-\d\d\-\d\d",element)[1].strip()
                elif re.search("^\D*,\D*$",element):
                    caseDict["DName2"] = element.strip()
                elif re.search("^\D*, \D\D \d{5}",element):
                    caseDict["AddrA2"] = element.strip()
                elif re.search("^\D/ph",element):
                    #phonum = element.strip().split(" ")[1].replace("%","")
                    phonum = "".join(element.strip().split(" ")[1:]).replace("%","").replace("-","")
                    if "phone" in caseDict.keys():
                        caseDict["phone"].append(phonum)
                    else:
                        caseDict["phone"] = [phonum]
                elif re.search("^ *\d*$",element):
                    pass
                else:
                    caseDict["AddrA1"] = element.strip()
            try:
                caseDict["AddrD"] = caseDict["AddrA1"] + ", " + caseDict["AddrA2"]
            except:
                caseDict["AddrD"] = "PLACEHOLDER"
                #print(quickmatch("^\D*, \D\D \d{5}",element),element)
            if "DName2" not in caseDict.keys():
                caseDict["DName2"] = ""
            for element in case[1]:
                #Front page data section: desk, dates, judgment amount/rate, caseno, venue
                if element[0] in caseDict.keys():
                    print("Element found!")
                else:
                    caseDict[element[0].strip()] = element[1].strip()
            #print(case[2])
            caseDict["FCosts"] = [x.strip() for x in case[2]] #Frontpage costs, totals
            tempPList = []
            for element in case[3]:
                #Caption plaintiff name
                if element.strip() != "":
                    tempPList.append(element[5:].strip().replace(",","").replace("on DBA","on, DBA"))
            if len(tempPList) > 1: plaintiff = "|".join(tempPList)
            else: plaintiff = tempPList[0]
            caseDict["Plaintiff"] = plaintiff
            tempDList = []
            for element in case[4]:
                #Caption defendant name
                if element.strip() != "":
                    tempDList.append(element[5:].strip())
            if len(tempDList) > 1: caseDict["Defendant"] = "|".join(tempDList)
            else: caseDict["Defendant"] = tempDList[0]
            caseDict["Docs"] = [x.strip() for x in case[5]] #Infoscreen docs entry
            caseDict["DNameA1"] = [x.strip() for x in case[6]] #Infoscreen service name/address
            try:
                for element in case[7]:
                    #Garnishment header info
                    caseDict[element[0].strip()] = element[1].strip().replace("-","")            
                caseDict["DtSvc"] = caseDict["svc"]
                #Garnishment defendant info
                if case[8][0][1].strip() != "": caseDict["Garnishee"] = "{}|{}".format(case[8][0][1].strip(),case[8][1][1].strip())
                else: caseDict["Garnishee"] = case[8][1][1].strip()
                caseDict["AddrG"] = case[8][2][1].strip().replace("208 MC-C","208 MC-CSC1") + ", " + case[8][3][1].strip()
                #Backup name check
                caseDict["NamesB"] = [case[9][x][1].strip() for x in range (len(case[9])) if case[9][x][1].strip()]
                #Garnishment cost table calculations
                for element in case[10]: caseDict[element[0].split("/")[-1].strip()[0].upper() + element[0].split("/")[-1].strip()[1:].lower() + "C"] = element[1].strip()
                caseDict["Svcst1"] = caseDict["SvcC"]
                if caseDict["TotalC"][-2:] == "43":
                    caseDict["Svcst2"] = "8.05"
                    caseDict["Svcst3"] = "1.53"
                else:
                    caseDict["Svcst2"] = caseDict["SvcC"]
                    caseDict["Svcst3"] = str(round(float(caseDict["MailC"]) - float(caseDict["SvcC"]),2))
                caseDict["Svcst4"] = str(round(float(caseDict["MailC"]) + float(caseDict["SvcC"]),2))
                caseDict["SubtotalC"] = str(round(float(caseDict["TotalC"]) - float(caseDict["AttyC"]),2))
                #Double check to make sure we don't have any strings stored with just one decimal place here
                for element in ["FileC","SvcC","AnsC","AttyC","MailC","ParteC","TotalC","Svcst1","Svcst2","Svcst3","Svcst4","SubtotalC"]:
                    if re.search("\.\d$",caseDict[element]):
                        caseDict[element] += "0"
                #Garnishment table, not very important
                caseDict["Garnlist"] = case[11]
                caseDict["Raw Memos"] = case[12]
                caseDict["Raw Memos"].sort(key = sortByDate)
            except:
                caseDict["Raw Memos"] = case[7]
                caseDict["Raw Memos"].sort(key = sortByDate)
            #Index names of the values that come out of processNotes
            indexes = ["MailnoG","MailnoD","DtMail","IntA","Dt2ans","Dt2R","Fnds"]
            notes = processNotes(caseDict["Raw Memos"])
            for x in range(len(indexes)):
                caseDict[indexes[x]] = notes[x]
            v = caseDict["venue"].split(" ")
            if v[0][0] == "D": caseDict["Level"] = "DISTRICT"
            else: caseDict["Level"] = "SUPERIOR"
            if v[2].replace(",","") == "DIVISION": caseDict["Division"] = v[1]
            else: caseDict["Division"] = ""
            caseDict["County"] = v[-2]
            if caseDict["County"] == "HARBOR": caseDict["County"] = "GRAYS HARBOR"
            caseDict["DtJ"] = caseDict["jdmt dt"].split(" ")[0].replace("-","")
            caseDict["IntR"] = str(int(caseDict["jdmt rt"].split(" ")[0].split(".")[1].replace(")",""))).replace("00","")
            defRS = caseDict["DName1"].split(",")
            fmid = defRS[1].strip().replace(";","")
            last = defRS[0].strip()
            if " " in fmid:
                if "JR" in fmid.split(" "):
                    fmid = "".join(fmid.split(" ")[:-1])
                    last += " JR"
            caseDict["DefS"] = fmid + " " + last
            caseDict["FolderName"] = last[0] + last[1:].lower() + " " + caseDict["case#"]
            #FCosts breakdown:
            tempfield = []
            tempnames = []
            for line in caseDict["FCosts"]:
                templine = [x for x in line.split(" ") if x != ""]
                #print(templine)
                tempfield.append([x for x in line.split(" ") if x != ""])
            for x in range (1,len(tempfield) - 3):
                #This gets us the correct rows
                for y in range(1,4):
                    #This gets is the correct columns
                    tempFieldName = tempfield[x][0].lower() + tempfield[0][y-1][0].upper() #Field names in format e.g. princA, princP, princB
                    tempnames.append(tempFieldName)
                    caseDict[tempFieldName] = tempfield[x][y] #Lets us refer quickly to specific amounts without having to remember grid squares
            #Things we need to precompute for extensions:
            if caseDict["IntA"]: caseDict["IntPost"] = floatStrMath(caseDict["j/intB"],caseDict["IntA"])
            else: caseDict["IntPost"] = "0"
            caseDict["currentCosts"] = floatStrMath(floatStrMath(caseDict["totalA"],caseDict["j/intA"]),caseDict["princA"])
            #if caseDict["$ xfer"] == "NO": caseDict["currentCosts"] = floatStrMath(caseDict["currentCosts"],caseDict["TotalC"])
            caseDict["DtExp"] = caseDict["expires"].replace("-","")
            caseDict = feeRead(caseDict)
            caseDict["ExtBalance"] = floatStrMath(caseDict["totalB"],caseDict["ExtFee"],2)
            #This should, theoretically, now be correct enough that we can make templates for extensions
        outList.append(caseDict)
    return outList

def feeRead(inDict):
    #Gets filing fees from the filing fee table in case we need them
    tempDict = inDict
    feeData = open("programData/fileFees.txt","r").read().splitlines()[1:]
    for entry in feeData:
        tempdata = entry.split("|")
        #print("Checking if",tempdata[0].upper(),"is",tempDict["County"].upper())
        #print(tempdata[0].upper())
        #print(tempDict["County"].upper())
        if tempdata[0].upper() == tempDict["County"].upper():
            #print("Match found! {}, {}".format(tempDict["County"],tempDict["Level"]))
            if tempDict["Level"].upper() == "SUPERIOR":
                #print("In superior court!")
                tempDict["ExtFee"] = tempdata[2]
                tempDict["ExpFee"] = tempdata[3]
            else:
                #print("In district court!")
                tempDict["ExtFee"] = tempdata[1]
                tempDict["ExpFee"] = tempdata[4]
    #if "ExtFee" not in tempDict.keys(): print("ERROR: Missed {}".format(tempDict["County"]))
    return tempDict

def processNotes(inList):
    #Separate function to split out the process of figuring out which notes are which
    tempList = inList
    tempList.sort(key = sortByDate) #Sort using the sortByDate helper function
    tempList.reverse() #Newest on top
    memoList = [re.split("\d\d:\d\d\D\D ",x) for x in tempList] #Splits around the time to get a 2-d list of dates and memos
    poeNum = getFirstMatch("(^poe|^poe cert) *\d\d\d\d",memoList) #Find the most recent poe mailing's certified mail number. Problem: POE CERT #
    if poeNum: poeCleanup = poeNum[1].lower().replace("poe","").replace(" ","").replace("cert","")
    else: poeCleanup = ""
    dbNum = getFirstMatch("(^db|^db cert) *\d\d\d\d",memoList) #Find the most recent db mailing's certified mail number
    if dbNum: dbCleanup = dbNum[1].lower().replace("db","").replace(" ","").replace("cert","")
    else: dbCleanup = ""
    writLine = getFirstMatch("^writ \d",memoList) #Find the most recent writ
    if writLine: dateOfWrit = writLine[0].split(" ")[1][:-1].replace("-","")
    else: dateOfWrit = ""
    govLine = getFirstMatch("governor",memoList) #Find the governor's interest suspension order of 2020
    if govLine: intA = govLine[1].strip().split(" ")[-1]
    else: intA = ""
    dt2ALine = getFirstMatch("2nd ans out",memoList) #Find the day we mailed out the 2nd answer
    if dt2ALine: Dt2A = dt2ALine[0].strip().split(" ")[1][:-1].replace("-","")
    else: Dt2A = ""
    dt2RLine = getFirstMatch("\**2nd ans rcvd",memoList) #Find the day we received the 2nd answer
    if dt2RLine: Dt2R = dt2RLine[0].strip().split(" ")[1][:-1].replace("-","")
    else: Dt2R = ""
    pmtList = []
    try:
        for entry in memoList: #Searching for writ payments received after the date of the writ
            if re.search("WRIT PMT",entry[1]) and sortByDate(entry[0]) > sortByDate(writLine[0]):
                if entry not in pmtList:
                    pmtList.append(entry)
    except:
        pass
    if pmtList: #If we found any, sum them
        tempsum = 0
        for entry in pmtList:
            try:
                tempsum += float(entry[1].split("$")[-1].split(" ")[0])
            except:
                #print(entry)
                tempnum = float(re.search("\d+\.\d+ ",entry[-1]).group())
                tempsum += tempnum
        Fnds = str(round(tempsum,2))
        if len(Fnds.split(".")[-1]) == 1:
            Fnds += "0"
    else: Fnds = "0"
    return [poeCleanup,dbCleanup,dateOfWrit,intA,Dt2A,Dt2R,Fnds]

def sortByDate(memo):
    #Helper function used to sort strings by date
    #Used to get most recent things in the note list
    return(memo[10:12] + memo[4:6] + memo[7:9] + memo[:3])

def processMilData(inFile,milDate,indexDicts = [],mode = 0):
    #This function takes a PAMAR printout, gets names, DoBs, and SSNs from it, and formats it for upload to SCRA
    inData = open("inputData/{}".format(inFile),"r").read().split("MERCHANTS CREDIT ASSOCIATION\n online collector v04.02")[:-1]
    #print(len(inData))
    dataBatches = []
    #The way this ended up working the split isn't really important
    indexData = open("outputData/MilDIndex.txt","w")
    for x,person in enumerate(inData):
        tempsubdata = [] #Data for an individual person
        if indexDicts: indexnum = indexDicts[x]["FolderName"]
        else: indexnum = re.search("P\d+ ",person).group()
        indexData.write("{}\n".format(indexnum))
        for subline in person.split("\n"):
            if len(tempsubdata) == 3: #After finding all three pieces of data
                #print(tempsubdata)
                if "..." not in tempsubdata[1] and "#" not in tempsubdata[1] and "..." not in tempsubdata[0]: #If the person has name and SSN, save their info
                    dataBatches.append(tempsubdata)
                    indexData.write("File\n")
                tempsubdata = [] #Clear the buffer
            if re.search(" +\d+  \D+\.+",subline): #Check to determine if the line is from the info screen
                if subline[:3] in ["  3"," 11"," 13"]: #Fields 3 (name), 11 (SSN), and DOB (13) are what we care about
                    tempsubdata.append(subline.strip().split("  ")[-1])
    indexData.close()
    if mode == 0:
        outFile = open("outputData/MilDOut.txt","w")
        for x in range(len(dataBatches)): #Write saved data to the MilDOut file
            outFile.write(fixedWidthTable(dataBatches[x],milDate))
        outFile.close()
    elif mode == 1:
        indexes = open("outputData/MilDIndex.txt","r").read().splitlines()
        indexList = []
        for line in indexes:
            if line == "File":
                indexList[-1][1] += 1
            else:
                indexList.append([line,0])
        print(len(dataBatches),len(indexList),len(inData))
        outFile = open("outputData/SSNs.txt","w")
        dataindex = 0
        for x in range(len(indexList)):
            outFile.write("In case {}\n".format(indexList[x][0]))
            for y in range(indexList[x][1]):
                outFile.write("{}\n".format(dataBatches[dataindex][1]))
                dataindex += 1
        outFile.close()

def extractRAg(inFile):
    #This function takes a PAMAR printout and gets the registered agents from it for label making
    inData = open("inputData/{}".format(inFile),"r").read().split("MERCHANTS CREDIT ASSOCIATION\n online collector v04.02")
    dataBatches = []
    for person in inData:
        persondata = []
        testpersondata = []
        tempsubdata = []
        outdata = []
        for subline in person.split("\n"):
            #print(subline)
            if len(tempsubdata) == 2:
                #print(tempsubdata[1])
                #persondata.append(tempsubdata)
                if not re.search("\.\.$",tempsubdata[1]):
                    persondata.append(tempsubdata)
                tempsubdata = []
            if re.search("employer|reg agent",subline):
                tempsubdata.append(subline.strip())
        dataBatches.append(persondata)
        #print(len(testpersondata),len(persondata))
    return dataBatches

def fixedWidthTable(inData,milDate):
    #Helper function to format and lay out the data in the right format for a SCRA upload
    #Layout will take uploads with empty data fields, and place and run them correctly
    #I don't know what the minimum the SCRA system will accept is, but this function should always get at least first, last, SSN
    first,middle,last = nameParse(inData[0])
    if "." in inData[2]: bday = "        "
    else: bday = inData[2][6:] + inData[2][:2] + inData[2][3:5]
    if "." in inData[1]: ssn = "         "
    else: ssn = inData[1]
    return "{0}{1}{2: <26}{3: <20}{4: <28}{5}{6: <20}\n".format(ssn,bday,last,first," ",milDate,middle)
    

def nameParse(name):
    #Helper function to split a name string into first, last, middle
    if "," not in name:
        last = name.split(" ")[-1].strip()
        firmid = " ".join(name.split(" ")[:-1]).strip()
    else:
        last = name.split(",")[0].strip()
        firmid = name.split(",")[1].strip()
    if ";" in firmid: firmid = firmid.split(";")[0]
    if " " in firmid: return firmid.split(" ")[0],firmid.split(" ")[1],last
    else: return firmid,"",last

def getFirstMatch(pattern,inList,caseCheck = True):
    #Find the first match of an RE pattern in a 2-d list of dates/memos
    #By default this runs with case-insensitive RE patterns because hand-entered data isn't super consistent
    for entry in inList:
        if caseCheck:
            if re.search(pattern,entry[1].lower()):
                return entry
        else:
            if re.search(pattern,entry[1]):
                return entry
    #print("PATTERN NOT FOUND")
    return ""


def quickmatch(pattern,string):
    #I was typing this out a lot!
    if re.search(pattern,string):
        return("YES")
    else:
        return("NO ")
    
def decimalPad(instr):
    if len(instr.split(".")[-1]) == 1:
        return instr + "0"
    else:
        return instr

def floatStrMath(in1,in2,mode = 1):
    #Given two strings that can be floats: convert to floats, do math, convert back to strings
    #Mode is 1 to subtract, 2 to add
    if in1 == "": ina = "0"
    else: ina = in1
    if in2 == "": inb = "0"
    else: inb = in2
    if mode == 1: return decimalPad(str(round(float(ina) - float(inb),2)))
    elif mode == 2: return decimalPad(str(round(float(ina) + float(inb),2)))
    

def dictCompare(dict1,dict2):
    #Quickly compare two dictionaries for debugging purposes
    for key in dict1.keys():
        if key in dict2.keys():
            #print("Match:",key,dict1[key],dict2[key])
            pass
        else:
            print("Mismatch:",key)
    #print("\nDict2 keys:")
    for key in dict2.keys():
        if key not in dict2.keys():
            print("Mismatch:",key)
    '''
    for key in dict1.keys():
        for key2 in dict2.keys():
            if dict1[key] == dict2[key2] and key != key2 and dict1[key] != "":
                print("Value match:",dict1[key],key,key2)'''

def compareData(file1,file2):
    #Compare whether legal/packet numbers from one file are in another
    input1 = open(file1,"r").readlines()
    input2 = open(file2,"r").readlines()
    for line in input2:
        lnum = line.split(" ")[0].replace("L","")
        pnum = line.split(" ")[1].replace("P","")
        found = False
        for line in input1:
            if lnum in line and pnum in line:
                found = True
        if not found:
            print("New line: L{}, P{}".format(lnum,pnum))

def saveLnumlist(inData,outName = "lnumsOut",batchnum = 0):
    #Saves a list of legal numbers from a parser output list
    #Not really important
    outFile = open("outputData/{}.txt".format(outName),"w")
    for entry in inData:
        if entry["FolderName"] in os.listdir("outputData/Batch{}".format(batchnum)):
            outFile.write("{} {} {}\n".format(entry["Legal"],entry["Packet"],entry["case#"]))
    outFile.close()

def validateMilData():
    #Checks military data to make sure it's correct
    outFile = open("outputData/MilDOut.txt").read().splitlines()
    outIndex = open("outputData/MilDIndex.txt").read().splitlines()
    for entry in outFile:
        if not all([(x.isalnum() or x == " ") for x in entry]):
            print("Non-alnumchar found!")
        print(len(entry))
        print("SSN:",entry[:9])
        print("DOB:",entry[9:17])
        print("Last:",entry[17:43])
        print("First:",entry[43:63])
        print("ID:",entry[63:91])
        print("Status:",entry[91:99])
        print("Middle:",entry[99:119])
    indexLen = [a for a in outIndex if a == "File"]
    print("Upload {} files, index {} files".format(len(outFile),len(indexLen)))

def dictFromTable(inFile):
    #Make a list of dictionaries from a table, to facilitate saving and loading parser output
    outData = []
    inData = open("inputData/{}".format(inFile),"r").read().splitlines()
    indexes = [x.strip() for x in inData[0].split("|")]
    for line in inData[1:]:
        linedict = {}
        data = [x.strip() for x in line.split("|")]
        for x in range(len(data)): linedict[indexes[x]] = data[x]
        outData.append(linedict)
    return outData
    
def sortExtReport(inFile,startdate,now,runtime):
    #Finds info in the extension index that meets certain properties
    data = dictFromTable(inFile)
    candidates = []
    for line in data:
        if Utilities.dateComparison(line["Expdate"].replace("-",""),startdate) in range(0,runtime) and line["State"] == "WA": #Expires in WA within two weeks of the 23rd
            if Utilities.dateComparison(line["Paydate"].replace("-",""),now) in range(-180,0): candidates.append(line["Legal"]) #Last payment in the last six months
            elif line["Desk"] in ["88","888"]: candidates.append(line["Legal"]) #Current garnishment running
            elif float(line["Balance"].replace(" ","")) >= 1000: candidates.append(line["Legal"]) #Balance high enough
    outFile = open("outputData/ExtCandidates.txt","w")
    for candidate in candidates:
        outFile.write("{}\n".format(candidate))
    outFile.close()

def feeSort(inData):
    #Produces a list of desk numbers indexed by county, level for requesting checks
    feeDict = {}
    for case in inData: feeDict = Utilities.dictAppend(feeDict," ".join([case["County"],case["Level"],"Court -"]),case["dsk"].replace("999","99"))
    return feeDict

def feeList(inData,filename,batchnum = 0):
    #Does the feeSort to file
    if batchnum >= 0: choplist = [x for x in inData if x["FolderName"] in os.listdir("outputData/Batch{}".format(batchnum))]
    else: choplist = inData
    procdata = feeSort(choplist)
    outFile = open("outputData/{}.txt".format(filename),"w")
    for case in procdata.keys():
        procdata[case].sort()
        outFile.write("{}\n".format(" ".join([a[0].upper() + a[1:].lower() for a in case.split(" ")])))
        dsklist = [int(a) for a in procdata[case]]
        dsklist.sort()
        for dsk in dsklist: outFile.write("\t{}\n".format(dsk))
    outFile.close()

def packetStatus(inFile):
    rawData = open("inputData/{}".format(inFile),"r").read()
    duplist = []
    outlist = []
    for line in rawData.splitlines():
        if re.search("^P\d+ +\D+ +D\d+",line) and line not in duplist:
            duplist.append(line)
            splitdata = [a for a in line.split(" ") if a != ""]
            outlist.append((splitdata[0][1:],splitdata[1]))
    return outlist
            