#PdfHandler
#This file uses various functions to do things with PDFs

import datetime
import os
import PyPDF2
import re
import docx2pdf
import shutil
import OCRrun
import math
import Utilities

now = str(datetime.date.today()).split("-")
currentDate = "{}{}{}".format(now[1],now[2],now[0][-2:])

def sortDecs(filename,mode,declen = 2,batchnum = 0,indexData = "", outname = "miscfile"):
    #Pre-OCR sorter, given a mode would assume the files would all be in the same order as the folder and file them to that folder.
    #Modes are the type of file we were sorting.
    quickpath = "outputData/Batch{}".format(batchnum)
    if mode == 0:
        #Splitting unspecified docs
        splitData = quickSplit("inputData/" + filename,[declen])
        for x in range (len(splitData)):
            splitData[x].write("outputData/{} {}.pdf".format(outname,x))
    elif mode == 1:
        #Splitting motions and orders
        splitData = quickSplit("inputData/" + filename,[declen])
        indexes = os.listdir("outputData/Batch{}".format(batchnum))
        for x in range(len(indexes)):
            #Sorting 2x and 2x+1 into indexes[x] as 00 and 01
            splitData[2*x].write("outputData/Batch{}/{}/00 {} Motion.pdf".format(batchnum,indexes[x],indexes[x]))
            splitData[2*x+1].write("outputData/Batch{}/{}/01 {} Order.pdf".format(batchnum,indexes[x],indexes[x]))
    elif mode == 8:
        #Splitting dec08 cert of garn costs
        indexes = os.listdir(quickpath)
        splitData = quickSplit("inputData/" + filename,[1])
        for x in range (len(splitData)):
            splitData[x].write(quickpath + "/" +  indexes[x] + "/08 " + indexes[x] + " cert garn costs.pdf")
    elif mode == 9:
        #Splitting milstatus declaration printouts
        splitData = quickSplit("inputData/" + filename,[2])
        indexes = open("outputData/MilDOut.txt","r").read().splitlines()
        print(len(splitData))
        splitFileDict = {}
        newFileList = []
        for x in range(len(splitData)):
            intname = indexes[x][17:43].strip()
            name = intname[0] + intname[1:].lower()
            if name in splitFileDict.keys():
                filename = "{0}_{1:0>{2}}.pdf".format(name,len(splitFileDict[name]),2)
                splitFileDict[name].append(filename)
                splitData[x].write(filename)
            else:
                filename = "{0}_00.pdf".format(name)
                splitFileDict[name] = [filename]
                splitData[x].write(filename)
        for y in splitFileDict.keys():
            if len(splitFileDict[y]) == 1:
                os.rename(splitFileDict[y][0],"09_1 {} milcert.pdf".format(splitFileDict[y][0][:-7]))
                newFileList.append("09_1 {} milcert.pdf".format(splitFileDict[y][0][:-7]))
            else:
                merger = PyPDF2.PdfMerger()
                for file in splitFileDict[y]:
                    tempfile = open(file,"rb")
                    merger.append(tempfile)
                    tempfile.close()
                merger.write("09_1 {} milcert.pdf".format(splitFileDict[y][0][:-7]))
                newFileList.append("09_1 {} milcert.pdf".format(splitFileDict[y][0][:-7]))
                for file in splitFileDict[y]:
                    os.remove(file)
        for z in newFileList:
            name = z[5:-12]
            for folder in os.listdir(quickpath):
                if name in folder:
                    os.rename(z,quickpath + "/" + folder + "/" + z)
    elif mode == 10:
        #Extension dec splitter
        splitData = quickSplit("inputData/" + filename,[2])
        indexes = open("Batch{} Print Order.txt".format(batchnum),"r").read().splitlines()
        for x in range(len(indexes)):
            splitData[2*x].write("outputData/Batch{}/{}/02 {} Dec re Ext.pdf".format(batchnum,indexes[x],indexes[x].split(" ")[1]))
            splitData[2*x + 1].write("outputData/Batch{}/{}/03 {} Dec re Int.pdf".format(batchnum,indexes[x],indexes[x].split(" ")[1]))
    elif mode == 11:
        #Extension motion and order splitter
        #Uneven splitter
        splitData = quickSplit("inputData/" + filename,[2,3])
        indexes = os.listdir("outputData/Batch{}".format(batchnum))
        for x in range(len(indexes)):
            splitData[2*x].write("outputData/Batch{}/{}/00 {} Motion.pdf".format(batchnum,indexes[x],indexes[x].split(" ")[1]))
            splitData[2*x + 1].write("outputData/Batch{}/{}/01 {} Order.pdf".format(batchnum,indexes[x],indexes[x].split(" ")[1]))
    elif mode == 12:
        #WDS splitter
        splitData = quickSplit("inputData/" + filename,[2])
        indexes = os.listdir("outputData/Batch{}".format(batchnum))
        for x in range(len(indexes)):
            splitData[x].write("outputData/Batch{}/{}/04 {} WDS.pdf".format(batchnum,indexes[x],indexes[x].split(" ")[1]))
        
        

def errorTruncate(batchnum = 0,mode = 0):
    #Correct problems caused by running other functions in this program in the wrong order
    quickpath = "outputData/Batch{}".format(batchnum)
    indexes = os.listdir(quickpath)
    for x in indexes:
        subpath = "{}/{}".format(quickpath,x)
        if mode == 0:
            for y in os.listdir(subpath):
                if y.split(" ")[0] == "04_0" and y.split(".")[-1] == "pdf":
                    #print("Found: {}".format(y))
                    corlen = 3
                    for z in ["04_1.pdf","04_2.pdf"]:
                        if z in os.listdir(subpath):
                            corlen += 1
                    tempReader = PyPDF2.PdfReader("{}/{}".format(subpath,y))
                    if len(tempReader.pages) != corlen:
                        #Shorten y
                        tempWriter = PyPDF2.PdfWriter()
                        tempWriter.append(tempReader,pages = PyPDF2.PageRange("0:{}".format(corlen)))
                        tempWriter.write("{}/T04_0.pdf".format(subpath))
        elif mode == 1:
            for y in os.listdir(subpath):
                if y.split(" ")[0] == "04_0" and y.split(".")[-1] == "pdf":
                    os.remove("{}/{}".format(subpath,y))
                    os.rename("{}/T04_0.pdf".format(subpath),"{}/{}".format(subpath,y))

def collectDocs(pattern = "",batchnum = 0):
    #Merge all files in a given batch file whose names contain the pattern
    quickpath = "outputData/Batch{}".format(batchnum)
    indexes = os.listdir(quickpath)
    merger = PyPDF2.PdfMerger()
    for x in indexes:
        subpath = "{}/{}".format(quickpath,x)
        for y in os.listdir(subpath):
            if pattern in y and ".pdf" in y:
                merger.append("{}/{}".format(subpath,y))
    merger.write("outputData/Batch{} Pattern Collection.pdf")

def splitByList(inDoc,pageList):
    #Basic split at page numbers function
    reader = PyPDF2.PdfReader("inputData/{}".format(inDoc))
    for x in range(len(pageList) - 1):
        pagenums = (pageList[x],pageList[x+1])
        merger = PyPDF2.PdfMerger()
        merger.append(reader,pages = pagenums)
        merger.write("outputData/File {}.pdf".format(x))

def printBatch(batchnum = 0):
    #Given a batch number put everything into it in one big PDF to preserve the order while printing
    #Superseded by countyGroup below
    quickpath = "outputData/Batch{}".format(batchnum)
    indexes = os.listdir(quickpath)
    for x in indexes:
        subpath = "{}/{}".format(quickpath,x)
        merger = PyPDF2.PdfMerger()
        for y in os.listdir(subpath):
            if re.search("^0\d_0.*\.pdf$",y) or re.search("^0\d .*\.pdf$",y):
                merger.append("{}/{}".format(subpath,y))
        merger.write("{}/_00 {} FULL.pdf".format(subpath,x.split(" ")[1]))
    bigmerger = PyPDF2.PdfMerger()
    for x in indexes:
        bigmerger.append("{}/{}/_00 {} FULL.pdf".format(quickpath,x,x.split(" ")[1]))
        bigmerger.append("{}/{}/01 {} Order.pdf".format(quickpath,x,x))
    bigmerger.write("outputData/Batch{} Full Merge.pdf")

def sortedPrintBatch(indexData,batchnum = 0,skipstep = [],skippattern = ""):
    #First go at grouping by county for printing
    quickpath = "outputData/Batch{}".format(batchnum)
    indexes = os.listdir(quickpath)
    if 0 not in skipstep:
        for x in indexes:
            if x != "Efile Section":
                subpath = "{}/{}".format(quickpath,x)
                merger = PyPDF2.PdfMerger()
                for y in os.listdir(subpath):
                    if re.search("^0\d_0.*\.pdf$",y) or re.search("^0\d .*\.pdf$",y):
                        if skippattern and skippattern not in y:
                            merger.append("{}/{}".format(subpath,y))
                        else:
                            merger.append("{}/{}".format(subpath,y))
                merger.write("{}/_00 {} FULL.pdf".format(subpath,x.split(" ")[1]))
    if 1 not in skipstep:
        printOrder = {}
        efileList = open("outputData/Batch{} Efile List.txt".format(batchnum),"w")
        for x in indexes:
            for y in indexData:
                if y["FolderName"] == x:
                    if y["County"] not in ["BENTON","KING","YAKIMA"]:
                        printOrder = dictAppend(printOrder,y["County"],x)
                    else:
                        efileList.write("{}\n".format(x))
        bigMerger = PyPDF2.PdfMerger()
        for x in printOrder.keys():
            print("Cases for {}: {}".format(x,printOrder[x]))
            for y in printOrder[x]:
                bigMerger.append("{}/{}/_00 {} FULL.pdf".format(quickpath,y,y.split(" ")[-1]))
                try: bigMerger.append("{}/{}/01 {} Order.pdf".format(quickpath,y,y))
                except: bigMerger.append("{}/{}/01 {} Order.pdf".format(quickpath,y,y.split(" ")[-1]))
        bigMerger.write("outputData/Batch{} Full Merge.pdf".format(batchnum))
        efileList.close()

def dictAppend(inDict,key,value):
    #Quick dictionary tool, helpful for sorting things into bins
    outDict = inDict
    if key in outDict.keys(): outDict[key].append(value)
    else: outDict[key] = [value]
    return outDict

def batchSort(indexData,batchnum = 0):
    #Moves all files that need to be efiled to a separate folder
    quickpath = "outputData/Batch{}".format(batchnum)
    indexes = os.listdir(quickpath)
    for x in indexes:
        if x not in ["Efile Section"]:
            for case in indexData:
                if case["FolderName"] == x:
                    if case["County"] in ["BENTON","KING","YAKIMA"]:
                        #Then we need to move to the efile section
                        os.mkdir("{}/Efile Section/{}".format(quickpath,x))
                        for file in os.listdir("{}/{}".format(quickpath,x)):
                            os.rename("{}/{}/{}".format(quickpath,x,file),"{}/Efile Section/{}/{}".format(quickpath,x,file))

def mergeToPrint():
    #Merges all files in a given folder
    quickPath = "inputData/MergerFolder"
    merger = PyPDF2.PdfMerger()
    for file in os.listdir(quickPath):
        tempFile = PyPDF2.PdfReader("{}/{}".format(quickPath,file))
        merger.append(tempFile)
        #merger.append(tempFile,pages=PyPDF2.PageRange("2:7"))
    merger.write("{}/printOrder.pdf".format(quickPath))

def sharePackage(dataList,batchnum = 0,mode = 0):
    #Separate out superior court extensions to be sent to someone else to file
    quickPath = "outputData/Batch{}".format(batchnum)
    if "ShareFolder" not in os.listdir("outputData"): os.mkdir("outputData/ShareFolder")
    for file in os.listdir(quickPath):
        for case in dataList:
            if case["FolderName"] == file and case["Level"] == "SUPERIOR":
                extpackage = ["00 {} Motion.pdf","01 {} Order.pdf","02 {} Dec re Ext.pdf","03 {} Dec re Int.pdf"]
                for doc in extpackage:
                    oldfilename = doc.format(" ".join(file.split(" ")[1:]))
                    newfilename = "{} {} {}".format(" ".join(file.split(" ")[1:]),oldfilename.split(" ")[0]," ".join(oldfilename.split(" ")[2:]))
                    shutil.copy("{}/{}/{}".format(quickPath,file,oldfilename),"outputData/ShareFolder/{}".format(newfilename))
    

def BentonRename(batchnum = 0,indexes = []):
    #Rename files to be filed in Benton County in the naming format Benton likes
    quickpath = "outputData/Batch{}".format(batchnum)
    files = os.listdir(quickpath)
    for a in range(len(files)):
        if len(indexes) == 0 or a in indexes:
            subpath = quickpath + "/" + files[a]
            deflast = files[a].split(" ")[0]
            caseno = files[a].split(" ")[1]
            for y in os.listdir(subpath):
                fileindex = y.split(" ")[0]
                if (int(fileindex[1]) in range(10)) and (y.split(".")[-1] == "pdf") and (not re.search("_1|_2",fileindex)):
                    newfilename = "{}_MCA vs {}_{}".format(caseno,deflast.upper(),y.split(caseno)[1][1:])
                    shutil.copy("{}/{}".format(subpath,y),"{}/{}".format(subpath,newfilename))
            
def batchMerger(batchnum = 0,indexes = [],filepatterns = []):
    #Merges declarations with attached supporting information for those declarations
    quickpath = "outputData/Batch{}".format(batchnum)
    files = os.listdir(quickpath)
    for a in range(len(files)):
        if len(indexes) == 0 or a in indexes:
            subpath = quickpath + "/" + files[a]
            for y in os.listdir(subpath):
                if y.split(".")[-1] == "pdf" and y.split(" ")[0] in ["04_0","05_0","06_0","09_0"]:
                    if len(filepatterns) == 0 or y.split(" ")[0] in filepatterns:
                        findex = y.split("_")[0]
                        print("Found mergable:",y)
                        for z in range (1,3):
                            for file in os.listdir(subpath):
                                #print(file.split(" ")[0], "{}_{}".format(findex,z))
                                if file.split(" ")[0].split(".")[0] == "{}_{}".format(findex,z):
                                    #print("Found!")
                                    quickJoin("{}/{}".format(subpath,y),"{}/{}".format(subpath,file),"{}/{}".format(subpath,y))

def printOrders(filename):
    #Orders come out of the PAMAR printout as 2pg motion/2pg order/1pg cert/1pg blank
    #Makes one file of motions/orders to print for the attorney to sign, one file of certs for me to sign, and leaves the blank page behind.
    quickpath = "inputData"
    reader = PyPDF2.PdfReader("{}/{}".format(quickpath,filename))
    mikeout = PyPDF2.PdfMerger()
    meout = PyPDF2.PdfMerger()
    for x in range (int(len(reader.pages) / 6)):
        mikeout.append(reader, pages = (6*x,6*x+4))
        meout.append(reader, pages = (6*x + 4,6*x+5))
    mikeout.write("Mikedata.pdf")
    meout.write("Medata.pdf")

def pdfIzeBatch(batchnum = 0,postprocess = False,indexes = [],filepatterns = []):
    #Renames files and turns Word documents into PDFs for a file that's all set to go
    quickpath = "outputData/Batch{}".format(batchnum)
    for x in range(len(os.listdir(quickpath))):
        element = os.listdir(quickpath)[x]
        if len(indexes) == 0 or x in indexes:
            caseno = element.split(" ")[-1]
            subpath = quickpath + "/" + element
            for file in os.listdir(subpath):
                runflag = True
                if filepatterns:
                    runflag = False
                    for pattern in filepatterns:
                        if pattern in file:
                            runflag = True
                if runflag and "~" not in file:
                    if file == "00.pdf":
                        os.rename(subpath + "/00.pdf","{}/00 {} Motion.pdf".format(subpath,caseno))
                    elif file == "01.pdf":
                        os.rename("{}/01.pdf".format(subpath),"{}/01 {} Order.pdf".format(subpath,caseno))
                    elif file == "02.pdf":
                        os.rename(subpath + "/02.pdf",subpath + "/02 " + caseno + " 2nd Answer.pdf")
                    elif file == "03.pdf":
                        os.rename(subpath + "/03.pdf",subpath + "/03 " + caseno + " 1st Answer.pdf")
                    elif file.split(".")[-1] == "docx" and not postprocess:
                        docx2pdf.convert(subpath + "/" + file, subpath + "/" + file[:-4] + "pdf")

def mergeByPattern(batchnum,patternlist,pagemerges):
    #Advanced version of collectdocs, merges documents within the batch that fit a pattern in patternlist, and can keep only some pages from them
    quickpath = "outputData/Batch{}".format(batchnum)
    merger = PyPDF2.PdfMerger()
    for folder in os.listdir(quickpath):
        subpath = "{}/{}".format(quickpath,folder)
        for file in os.listdir(subpath):
            for x in range(len(patternlist)):
                if re.search(patternlist[x],file):
                    merger.append("{}/{}".format(subpath,file),pages = pagemerges[x])
    merger.write("outputData/Batch {} Patternmerge.pdf".format(batchnum))

def splitNewGarns(inFolder = "MergerFolder",mode = 0):
    #Given a new garnishmentprintout from PAMAR chop up the documents and consolidate into large categorical docs to make fiing easier
    #Essentially obsolete
    quickpath = "inputData/{}".format(inFolder)
    meblock = PyPDF2.PdfMerger()
    mikeblock = PyPDF2.PdfMerger()
    ans2block = PyPDF2.PdfMerger()
    ans1block = PyPDF2.PdfMerger()
    exemptblock = PyPDF2.PdfMerger()
    if mode == 0:
        fileData = os.listdir(quickpath)
        for file in fileData:
            if ".pdf" in file:
                meblock.append("{}/{}".format(quickpath,file),pages = (0,2))
                mikeblock.append("{}/{}".format(quickpath,file),pages = (2,7))
                ans2block.append("{}/{}".format(quickpath,file),pages = (13,17))
                ans1block.append("{}/{}".format(quickpath,file),pages = (9,13))
                exemptblock.append("{}/{}".format(quickpath,file),pages = (7,9))
    elif mode == 1:
        fileData = os.listdir(quickpath)
        for file in fileData:
            if ".pdf" in file:
                fileData = PyPDF2.PdfReader("{}/{}".format(quickpath,file))
                indexes = int(len(fileData.pages) / 17)
                for x in range(indexes):
                    sti = 6*x
                    meblock.append("{}/{}".format(quickpath,file),pages = (sti,sti+2))
                    mikeblock.append("{}/{}".format(quickpath,file),pages = (sti+2,sti+7))
                    ans2block.append("{}/{}".format(quickpath,file),pages = (sti+13,sti+17))
                    ans1block.append("{}/{}".format(quickpath,file),pages = (sti+9,sti+13))
                    exemptblock.append("{}/{}".format(quickpath,file),pages = (sti+7,sti+9))                    
    meblock.write("{}/Applications.pdf".format(quickpath))
    mikeblock.write("{}/Writs.pdf".format(quickpath))
    ans2block.write("{}/2nd Answers.pdf".format(quickpath))
    ans1block.write("{}/1st Answers.pdf".format(quickpath))
    exemptblock.write("{}/Exemptions.pdf".format(quickpath))

def garnMerger(inFolder = "MergerFolder",debugRange = [],outName = "PrintBatch.pdf"):
    #Given signed outputs from splitNewGarns chop them back up and stitch them back together again.
    #Obsolete
    quickpath = "inputData/{}".format(inFolder)
    #We expect to find in that folder several large PDFs containing: apps, writs, exempts, nors, 1st ans, 2nd ans
    appList = quickSplit("inputData/{}/Final/SignedApp.pdf".format(inFolder),[2])
    ansList = quickSplit("inputData/{}/Final/1st Ans.pdf".format(inFolder),[4])
    exList = quickSplit("inputData/{}/Final/Exemption.pdf".format(inFolder),[2])
    writList = quickSplit("inputData/{}/Final/SignedWrits.pdf".format(inFolder),[5])
    redList = quickSplit("inputData/{}/Final/RedactedWrits.pdf".format(inFolder),[5])
    printData = PyPDF2.PdfMerger()
    print("Splitting:")
    for x in range(len(appList)):
        if len(debugRange) == 0 or x in debugRange:
        #We need to make two packets: the garnishee packet has the writ and the 1st answer, the defendant gets app, writ, ex, and programData/GarnRightsNotice.pdf
            writList[x].write("inputData/{}/Final/Tempfolder/{}_0.pdf".format(inFolder,x))
            ansList[x].write("inputData/{}/Final/Tempfolder/{}_1.pdf".format(inFolder,x))
            appList[x].write("inputData/{}/Final/Tempfolder/{}_2.pdf".format(inFolder,x))
            writList[x].write("inputData/{}/Final/Tempfolder/{}_3.pdf".format(inFolder,x))
            exList[x].write("inputData/{}/Final/Tempfolder/{}_4.pdf".format(inFolder,x))
            redList[x].write("inputData/{}/Final/Tempfolder/{}_5.pdf".format(inFolder,x))
    print("Merging:")
    for x in range(len(appList)):
        if len(debugRange) == 0 or x in debugRange:
            for y in range(5):
                printData.append("inputData/{}/Final/Tempfolder/{}_{}.pdf".format(inFolder,x,y))
            printData.append("programData/GarnRightsNotice.pdf")
            appWrit = PyPDF2.PdfMerger()
            appWrit.append("inputData/{}/Final/Tempfolder/{}_2.pdf".format(inFolder,x))
            appWrit.append("inputData/{}/Final/Tempfolder/{}_5.pdf".format(inFolder,x))
            appWrit.write("inputData/{}/Final/FileFolder/App and Writ {}.pdf".format(inFolder,x))
    printData.write("inputData/{}/Final/{}".format(inFolder,outName))

def garnWrapper(inFolder = "MergerFolder",groupBy = 0,numGarns = 0):
    #Use garnMerger, but divide it into subgroups for printing so we don't jam the ouput
    if not groupBy:
        garnMerger(inFolder = inFolder)
    else:
        groups = numGarns//groupBy
        lastgroup = numGarns%groupBy
        for x in range(groups):
            print("Group {}".format(x))
            garnMerger(inFolder = inFolder,debugRange = [a+(x*groupBy) for a in range(groupBy)],outName = "PrintBatch{}.pdf".format(x))
        if lastgroup != 0:
            garnMerger(inFolder = inFolder,debugRange = [a+(groups*groupBy) for a in range(lastgroup)],outName = "PrintBatch{}.pdf".format(groups))

def garnCountyGroup(groupList,inFolder = "MergerFolder",fix = False):
    #Merges new garnishments by county to be mailed to court
    path = "inputData/{}/Final/FileFolder".format(inFolder)
    countymergeorder = PyPDF2.PdfMerger()
    for x in range(len(groupList)):
        if fix:
            if groupList[x] >= len(groupList):
                countymergeorder.append("{}/App and Writ {}.pdf".format(path,groupList[x]))
        else:
            countymergeorder.append("{}/App and Writ {}.pdf".format(path,groupList[x]))
    countymergeorder.write("{}/Print Order.pdf".format(path))

def countyGroup(groupDictList,batchnum = 0,subnames = 0,debug = False):
    #Merges the contents of folders for writs and extensions that need to be filed on paper into one big PDF to make the print order consistent
    cases = os.listdir("outputData/Batch{}".format(batchnum))
    countiesDict = {}
    countiesDesks = {}
    #Formulate county list
    for case in cases:
        for file in groupDictList:
            if file["FolderName"] == case:
                countyIndex = " ".join([Utilities.capitalize(a) for a in "{} {}".format(file["County"],file["Level"]).split(" ")])
                countiesDict = Utilities.dictAppend(countiesDict,countyIndex,case)
                countiesDesks = Utilities.dictAppend(countiesDesks,countyIndex,file["dsk"])
    printList = PyPDF2.PdfMerger()
    #WDSlist = PyPDF2.PdfMerger()
    maxcaseindex = sum([len(countiesDict[a]) for a in countiesDict.keys() if a not in ["King Superior","King District","Benton District","Yakima District"]])
    caseindex = 1
    cumlen = 0
    #Merge files not in the efile locations
    for county in countiesDict.keys():
        if county not in ["King Superior","King District","Benton District","Yakima District"]:
            #Then we need to file by paper, so we need to print
            for case in countiesDict[county]:
                #This includes the files in the order they'd be found in either an extension or an order
                filenamelist = ["04 {} WDS.pdf","00 {} Motion.pdf","01 {} Order.pdf","02 {} 2nd Answer.pdf", "02 {} Dec re Ext.pdf","03 {} 1st Answer.pdf","03 {} Dec re Int.pdf",
                                "04_0 {} Dec of Serv.pdf","05_0 {} Missing POE.pdf","06_0 {} Missing DB.pdf","07_0 {} Fr Interest.pdf",
                                "08 {} Cert Garn Costs.pdf","09_0 {} Dec Mil Status.pdf", "01 {} Order.pdf"]
                caselen = 0 #Page length of each case for debugging purposes
                for file in filenamelist:
                    casepath = "outputData/Batch{}/{}".format(batchnum,case)
                    #Some versions of this program have put the full name + number in the file names, this is here for backwards-compatibility
                    filecandidates = [file.format(case),file.format(" ".join(case.split(" ")[1:]))]
                    for candidate in filecandidates:
                        if candidate in os.listdir(casepath):
                            printList.append("{}/{}".format(casepath,candidate))
                            caselen += len(PyPDF2.PdfReader("{}/{}".format(casepath,candidate)).pages)
                #Status to watch as program runs
                print("{}: File {} of {}: pages {} to {}".format(case,caseindex,maxcaseindex,cumlen,cumlen + caselen))
                #Separator pages to make it easier to split up paper filings after printing
                if caseindex < maxcaseindex: 
                    printList.append("programData/Separator Page.pdf")
                    cumlen += 1
                cumlen += caselen
                caselen = 0
                caseindex += 1
    #Write printlist to output
    printList.write("outputData/Batch{} Print List.pdf".format(batchnum))
    #WDSlist.write("outputData/Batch{} WDS List.pdf".format(batchnum))
    deskList = open("outputData/Batch{} Desk List.txt".format(batchnum),"w")
    for county in countiesDesks.keys():
        deskList.write("{}\n\n".format(county))
        for desk in countiesDesks[county]:
            deskList.write("\t{}\n".format(desk))
    deskList.close()

def WDScopy(batchnum):
    #Copy withdrawal and substitution documents from the folder to the scanner folder where they're inputted into the filing system
    for case in os.listdir("outputData/Batch{}".format(batchnum)):
        if "04 {} WDS.pdf".format(" ".join(case.split(" ")[1:])) in os.listdir("outputData/Batch{}/{}".format(batchnum,case)):
            shutil.copy("outputData/Batch{}/{}/04 {} WDS.pdf".format(batchnum,case," ".join(case.split(" ")[1:])),"P:/debtors/ocr-pkt-input/{} WDS.pdf".format(" ".join(case.split(" ")[1:])))
                
def caseMerger(foldername,batchnum = 0,mode = 0):
    #Obsolete merger that made a full merge individually for each case in its folder
    #Replaced by garnCountyGroup
    #Yes, efile counties want all documents as individual PDFs
    printList = PyPDF2.PdfMerger()
    subnamelist1 = ["00 {} Motion.pdf","01 {} Order.pdf","02 {} Dec re Ext.pdf","03 {} Dec re Int.pdf","01 {} Order.pdf"]
    subnamelist2 = ["00 {} Motion.pdf","01 {} Order.pdf","02 {} 2nd Answer.pdf","03 {} 1st Answer.pdf","04_0 {} Dec of Serv.pdf","05_0 {} Missing PoE.pdf","06_0 {} Missing DB.pdf","07_0 {} Fr Interest.pdf","08 {} Cert Garn Costs.pdf","09_0 {} Dec Mil Status.pdf", "01 {} Order.pdf"]
    if mode == 0:
        for file in subnamelist1:
            printList.append("outputData/Batch{}/{}/{}".format(batchnum,foldername,file.format(foldername.split(" ")[-1])))
    elif mode == 1:
        outpath = "outputData/Batch{}/{}".format(batchnum,foldername)
        for file in subnamelist2:
            filecandidate1 = file.format(foldername)
            filecandidate2 = file.format(foldername.split(" ")[-1])
            if filecandidate1 in os.listdir(outpath): printList.append("{}/{}".format(outpath,filecandidate1))
            elif filecandidate2 in os.listdir(outpath): printList.append("{}/{}".format(outpath,filecandidate2))
    printList.write("outputData/Batch{}/{}/_00 {} MERGED.pdf".format(batchnum,foldername,foldername.split(" ")[-1]))
    

def quickSplit(filename,blocksize):
    #Quick helper function to split a PDF file into chunks of length blocksize
    outlist = []
    reader = PyPDF2.PdfReader(open(filename,"rb"))
    if len(blocksize) == 1:
        for x in range(int(len(reader.pages) / blocksize[0])):
            writer = PyPDF2.PdfWriter()
            for y in range(blocksize[0]):
                writer.add_page(reader.pages[x * blocksize[0] + y])
            outlist.append(writer)
    else:
        largeblocksize = sum(blocksize)
        index = 0
        for x in range(int(len(reader.pages) / largeblocksize)):
            #X works right, x goes 1-4
            print(len(blocksize))
            for y in range(len(blocksize)):
                #Y does NOT work right, y is only ever 1?
                writer = PyPDF2.PdfWriter()
                for z in range(index,index+blocksize[y]):
                    print(x,y,index,index+y)
                    writer.add_page(reader.pages[index])
                    index += 1
                outlist.append(writer)
    return outlist

def quickJoin(path1,path2,outpath):
    #Helper function to concatenate two files
    #Typically used with outpath = path1 to avoid creating too many extra new files
    print("Joining!",path1,path2,outpath)
    merger = PyPDF2.PdfMerger()
    merger.append(path1)
    merger.append(path2)
    merger.write(outpath)

def getPdfData(filename,pattern = "L\d+ P\d+",debug = False):
    #Reads text data from text-extractable PDFs
    #Primarily used to get order indexes from PAMAR pdf printouts
    #outFilename = "PdfData{}_0.txt".format(currentDate)
    index = 0
    outFilename = "inputData/extrIndexes.txt"
    outFile = open(outFilename,"w")
    reader = PyPDF2.PdfReader("inputData/{}".format(filename))
    for page in reader.pages:
        data = page.extract_text().split("\n")
        for line in data:
            if re.search(pattern,line):
                outFile.write(line + "\n")
    outFile.close()

def extPrintBatch(batchnum):
    #Batches up extension declarations for signature
    qp = "outputData/Batch{}".format(batchnum)
    mikebatch = PyPDF2.PdfMerger()
    mebatch = PyPDF2.PdfMerger()
    wdsbatch = PyPDF2.PdfMerger()
    mebatchorder = open("Batch{} Print Order.txt".format(batchnum),"w")
    for folder in os.listdir(qp):
        lp = "{}/{}".format(qp,folder)
        mebatchorder.write(folder + "\n")
        for file in os.listdir(lp):
            if file.split(".")[1] == "docx":
                newfilename = "{}/{}".format(lp,file.split(".")[0] + ".pdf")
                docx2pdf.convert("{}/{}".format(lp,file),newfilename)
                if file.split(" ")[0] in ["00","01"]: mikebatch.append(newfilename)
                elif file.split(" ")[0] in ["02","03"]: mebatch.append(newfilename)
                elif file.split(" ")[0] in ["04"]: wdsbatch.append(newfilename,pages = (0,2))
    mikebatch.write("outputData/Batch{} Mike Print Order.pdf".format(batchnum))
    mebatch.write("outputData/Batch{} Me Print Order.pdf".format(batchnum))
    wdsbatch.write("outputData/Batch{} WDS Print Order.pdf".format(batchnum))

def shortExtPrint(batchnum):
    #As above, but for if you already have PDFs in the file
    qp = "outputData/Batch{}".format(batchnum)
    mikebatch = PyPDF2.PdfMerger()
    mebatch = PyPDF2.PdfMerger()
    mebatchorder = open("Batch{} Print Order.txt".format(batchnum),"w")
    for folder in os.listdir(qp):
        lp = "{}/{}".format(qp,folder)
        mebatchorder.write(folder + "\n")
        for file in os.listdir(lp):
            if file.split(".")[1] == "pdf":
                newfilename = "{}/{}".format(lp,file)
                if file.split(" ")[0] in ["00","01"]: mikebatch.append(newfilename)
                elif file.split(" ")[0] in ["02","03"]: mebatch.append(newfilename)
    mikebatch.write("outputData/Batch{}/Batch{} Mike Print Order.pdf".format(batchnum,batchnum))
    mebatch.write("outputData/Batch{}/Batch{} Me Print Order.pdf".format(batchnum,batchnum))

def supersmartsplit(indexData,inFile,batchnum = 0):
    #Unfinished, concept transferred to OCRrun
    qp = "outputData/Batch{}".format(batchnum)

def unbatchedMerger(inPath):
    #Setup to convert Word docs to PDFs and merge them when you don't have a batch filestructure
    printData = PyPDF2.PdfMerger()
    for file in os.listdir(inPath):
        if ".docx" in file:
            docx2pdf.convert("{}/{}".format(inPath,file),"{}/{}.pdf".format(inPath,file.split(".")[0]))
    for file in os.listdir(inPath):
        if ".pdf" in file:
            printData.append("{}/{}".format(inPath,file))
    printData.write("{}/Merged.pdf".format(inPath))

def smartNewCaseSplit(inFile):
    #Wrapper using OCRrun's splitlist function to determine what files are and where they should go by their footer
    splitlist = OCRrun.newCaseSplit(inFile)
    for x in range(len(splitlist) - 1):
        #Each file in splitlist is name, legal, startindex
        merger = PyPDF2.PdfMerger()
        merger.append("inputData/{}".format(inFile),pages = (splitlist[x][2],splitlist[x+1][2]))
        merger.write("inputData/NewCases/{} {} {}.pdf".format(math.floor(x/2),splitlist[x][1],splitlist[x][0]))

def newMilsplitter(inFile,inData,batchnum = 0):
    #Splits SCRA certificates by case using an index of how many SCRA certificates belong to each case
    #The index is created by the processMilData function in DataProcessor
    indexData = open("outputData/MilDIndex.txt").read().splitlines()
    splitIndexes = []
    for line in indexData:
        print(line)
        if line.strip() == "File":
            splitIndexes[-1][1] += 1
        else:
            for case in inData:
                if case["Packet"] == line[1:].strip():
                    splitIndexes.append([case["FolderName"],0])
                elif case["FolderName"] == line.strip():
                    splitIndexes.append([case["FolderName"],0])
    milCerts = PyPDF2.PdfReader("inputData/{}".format(inFile))
    currentIndex = 0
    for x in splitIndexes:
        tempWriter = PyPDF2.PdfWriter()
        tempWriter.append(milCerts,pages = (currentIndex,currentIndex + 2*x[1]))
        tempWriter.write("outputData/Batch{}/{}/09_1 {} MilCert.pdf".format(batchnum,x[0],x[0].split(" ")[-1]))
        currentIndex += 2*x[1]