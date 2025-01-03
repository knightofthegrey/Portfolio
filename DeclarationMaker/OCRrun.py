#OCR Attempt 2: instead of trying to make our own library try using an existing one

from PIL import Image, ImageDraw, ImageFont
import PyPDF2
import fitz
import pytesseract
import os
import shutil
import re
import time
import Utilities

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
filedpath = "//diskstation/Legal Team/TEAM MEMBER DIRECTORIES/NXS/Filed Documents"
#Tesseract notes: https://pyimagesearch.com/2021/11/15/tesseract-page-segmentation-modes-psms-explained-how-to-improve-your-ocr-accuracy/
#Bottom line: OCR mode 6 for paragraphs, 7 for single lines

#This file uses Tesseract, a free OCR tool originally developed by HP in the '80s and released under the Apache license in 2005

def readPDF(inPath):
    #Older reader function, reads a PDF, binarizes images of each page, and uses Tesseract to convert to text, then returns the list.
    os.mkdir("temp")
    outlist = []
    doc = fitz.open(inPath)
    for page in enumerate(doc):
        pix = page[1].get_pixmap()
        pix.save("temp/img{}.png".format(page[0]))
        pageFile = Image.open("temp/img{}.png".format(page[0]))
        pageFile = pageFile.convert("L")
        pageFile = pageFile.point(lambda p: 255 if p > 180 else 0)
        pageFile = pageFile.convert("1")
        outlist.append(pytesseract.image_to_string(Image.open("temp/img{}.png".format(page[0]))))
    for file in os.listdir("temp"):
        os.remove("temp/{}".format(file))
    os.rmdir("temp")
    return outlist

def findCasenos(inList):
    #Looks for case numbers in readPDF outputs
    outList = []
    for entry in inList:
        tempData = entry.split("\n")
        caseno = "No Case"
        for line in tempData:
            if re.search("cause no",line.lower()):
                caseno = line.split(" ")[-1].upper()
        outList.append(caseno)
    return outList

def matchCasenos(inList,batchnum = 0):
    #Matches casenos from findCasenos to the filenames in a batch
    quickpath = "outputData/Batch{}".format(batchnum)
    cases = os.listdir(quickpath)
    indexDict = {}
    unmatchedList = []
    #If anything in inList is in cases we're good, that's all set.
    for x in range(len(cases)):
        tempnum = cases[x].split(" ")[-1]
        #We need the index of its value in inList, if it's there
        for y in range(len(cases)):
            if tempnum == inList[y]:
                indexDict[y] = cases[x]
    for x in range(len(cases)):
        #We need cases[x] to not be a value in indexDict here
        unmatched = True
        for a in indexDict.keys():
            if indexDict[a] == cases[x]:
                unmatched = False
        if unmatched:
            unmatchedList.append(cases[x])
    for x in range(len(cases)):
        if x not in indexDict.keys():
            indexDict[x] = fuzzymatch(inList[x],unmatchedList)
    return indexDict

def fuzzymatch(caseNo,caseList):
    #If we don't have a match we need a fuzzy test
    if "-" in caseNo.replace("~","-"):
        return nummatch(caseNo.replace("~","-"),caseList)
    else:
        return lettermatch(caseNo,caseList)

def nummatch(caseNo,caseList):
    #Trying to match format xxx-xxxxx
    casenums = [a for a in caseNo if a.isdigit()]
    candidateList = [a for a in caseList if "-" in a]
    #print(candidateList)
    #print(casenums)
    for candidate in candidateList:
        numlist = [a for a in candidate if a.isdigit()]
        if numlist == casenums: return candidate
        else:
            if len(casenums) > len(numlist):
                a = casenums
                b = numlist
            else:
                a = numlist
                b = casenums
            delta = len(a) - len(b)
            if delta == 1:
                for x in range(len(a)):
                    temp = a[:]
                    del temp[x]
                    if temp == b:
                        return candidate
            elif delta == 2:
                for x in range(len(a)):
                    temp1 = a[:]
                    del temp1[x]
                    for y in range(len(temp1)):
                        temp2 = temp1[:]
                        del temp2[y]
                        if temp2 == b:
                            return candidate
    return "No match"

def lettermatch(caseNo,caseList):
    #Trying to match format xxCIVxxxxxKCX
    casenums = [a for a in (caseNo[:2] + caseNo[5:]) if a.isdigit()]
    candidateList = [a for a in caseList if "-" not in a]
    #print(candidateList)
    #Pull letters, check numbers only
    #print(casenums)
    for candidate in candidateList:
        numlist = [a for a in candidate if a.isdigit()]
        if numlist == casenums:
            return candidate
        elif len(numlist) != len(casenums):
            for x in range(7):
                templist = numlist[:]
                del templist[x]
                if templist == casenums:
                    return candidate
    return "No matches"

def smartSplit(inPdf,batchnum = 0,debug = False):
    #PDF splitter that uses the above functions to try and find pages with case numbers to split the file at
    outNameDict = {}
    for line in open("programData/smartFilenames.txt","r").read().splitlines():
        outNameDict[line.split("|")[0]] = line.split("|")[1]
    reader = PyPDF2.PdfReader("inputData/{}".format(inPdf))
    docData = readPDF("inputData/{}".format(inPdf))
    caseList = findCasenos(docData)
    indexDict = matchCasenos(caseList,batchnum = batchnum)
    fileIndex = 0
    tempMerger = PyPDF2.PdfMerger()
    for x in range(len(caseList)):
        if caseList[x] != "No Case":
            if x != 0:
                if debug: print("Writing {}".format(tempName))
                else: tempMerger.write(tempName)
                tempMerger = PyPDF2.PdfMerger()
            tempName = "outputData/Batch{}/placeholderName{}.pdf".format(batchnum,fileIndex)
            for line in docData[x].split("\n"):
                if line in outNameDict.keys():
                    tempName = "outputData/Batch{}/{}/{}".format(batchnum,indexDict[fileIndex],outNameDict[line].format(indexDict[fileIndex].split(" ")[-1]))
            fileIndex += 1
        tempMerger.append(reader,pages = (x,x+1))
    if debug: print("Writing {}".format(tempName))
    else: tempMerger.write(tempName)

def imageTruncate():
    #Test function to read a PDF, convert to image, and then chop into regions separated by white lines
    #Function to image input
    testDoc = fitz.open("inputData/0731 Orders.pdf")
    testPage = testDoc[0].get_pixmap(dpi = 300)
    testPage.save("testImage.png")
    testImage = Image.open("testImage.png")
    width,height = testImage.size
    pixels = testImage.load()
    #Identify which lines are all white pixels
    pxrowlist = []
    lineRegions = []
    for y in range(height):
        whiteLine = True
        for x in range(width):
            if pixels[x,y] != (255,255,255):
                whiteLine = False
        pxrowlist.append(whiteLine)
    #Use regions with all white pixels to define lines of text
    for x in range(len(pxrowlist) - 1):
        if pxrowlist[x] and not pxrowlist[x+1]:
            start = x
        elif not pxrowlist[x] and pxrowlist[x+1]:
            horizontalbox = (0,start-2,width,x+4)
            lineRegions.append(horizontalbox)
    #print(lineRegions[3])
    #Save lines of text as test images
    for x in range(len(lineRegions)):
        testImage.crop(lineRegions[x]).save("test{}.png".format(x))

def subsetOcr():
    #Test to see how long it takes Tesseract to OCR individual test lines
    for x in range(36):
        start = time.time()
        print("Test{}: {}".format(x,pytesseract.image_to_string(Image.open("test{}.png".format(x))).replace("\n","")))
        print("In {}".format(time.time() - start))

def binarize(inImage,threshold = 200):
    #Binarize scanned images
    pixels = inImage.load()
    #print(pixels[10,19])
    for i in range (inImage.size[0]):
        for j in range (inImage.size[1]):
            avg = (pixels[i,j][0] + pixels[i,j][1] + pixels[i,j][2]) // 3
            if avg < threshold:
                pixels[i,j] = (0,0,0)
            else:
                pixels[i,j] = (255,255,255)
    return inImage

def focusDown(page,mode,debug = False, ocrMode = "6",savePage = ""):
    #Crop page to a region where we expect to find important text, then OCR that region
    #At the moment these are hardcoded regions, ratios exist because not all PDFs convert to images with exactly the same dimensions
    #I'm as yet unsure exactly how or why, but PDFs coming off our scanner sometimes appear to be the same size but vary in height by +/- 10%
    #Page is a fitz page object, mode is the mode number of the region to crop to
    #ocrMode is Tesseract's page segmentation mode; with no argument tesseract.image_to_string() assumes it's looking at a full page of text
    #Mode 6 is for a paragraph
    testPage = page.get_pixmap(dpi = 300)
    ratio1 = testPage.width / 2550
    ratio2 = testPage.height / 3500
    #testPage.save("tempImage.png")
    testImage = Image.frombytes("RGB",[testPage.width,testPage.height],testPage.samples)
    #region1 = testImage.crop((1520,1150,2200,1198))
    identDict = {}
    #Each mode crops to a region and looks for specific data in that region
    if mode == 0:
        #Footer: 400-2100x, 3000-3200y
        cropRegion = testImage.crop((int(400*ratio1),int(3050*ratio2),int(2100*ratio1),int(3500*ratio2)))
        tempData = pytesseract.image_to_string(binarize(cropRegion),config = "--psm {}".format(ocrMode)).split("\n")
        for a in [["Legal","L\d+ "],["Packet","P\d+ "],["dsk","D\d+ "],["Page","P\D \d"]]:
            try: identDict[a[0]] = re.search(a[1],tempData[0]).group().strip()
            except: pass
        for b in [["Legal",0],["Packet",1],["dsk",2],["Page",5]]:
            if b[0] not in identDict.keys():
                try: identDict[b[0]] = tempData[0].split(" ")[b[1]].strip()
                except: pass
        for c in [["Legal","78",[7]],["Packet","0123456789",[7,8]],["dsk","0123456789",[2,3,4]]]:
            if c[0] in identDict.keys():
                #print(c[0],identDict[c[0]],len(identDict[c[0]]),c[1],c[2])
                if len(identDict[c[0]]) in c[2] and identDict[c[0]][0] not in c[1]:
                    identDict[c[0]] = identDict[c[0]][1:]
        if "Page" in identDict.keys() and " " in identDict["Page"]: identDict["Page"] = identDict["Page"].split(" ")[-1]
        identDict["Raw0"] = tempData
        if not tempData[1].replace(" ","").isalnum(): identDict["Doctitle"] = " ".join(tempData[1].split(" ")[:-1])
        else: identDict["Doctitle"] = tempData[1]
        if debug:
            print(tempData)
            print(identDict)
            cropRegion.save("DebugTest.png")
    elif mode == 10:
        #Medocs footer: 380-1800x, 3100-3200y
        cropRegion = testImage.crop((int(360*ratio1),int(3200*ratio2),int(1800*ratio1),int(3500*ratio2)))
        tempData = pytesseract.image_to_string(binarize(cropRegion),config = "--psm {}".format(ocrMode)).split("\n")
        if len(tempData[0].split(" ")) < 5: tempData = tempData[1:]
        identDict["Legal"] = tempData[0].split(" ")[0][1:]
        identDict["Packet"] = tempData[0].split(" ")[1][1:]
        identDict["Doctitle"] = " ".join(tempData[0].split(" ")[2:-2])
        identDict["Page"] = tempData[0].split(" ")[-1]
        identDict["Raw10"] = tempData
        if debug:
            print(tempData)
            cropRegion.save("DebugTest.png")
    elif mode == 11:
        cropRegion = testImage.crop((int(360*ratio1),int(3200*ratio2),int(1800*ratio1),int(3500*ratio2)))
        tempData = pytesseract.image_to_string(binarize(cropRegion),config = "--psm {}".format(ocrMode)).split("\n")
        if len(tempData[0].split(" ")) < 5: tempData = tempData[1:]
        tempData = " ".join(tempData[:2])
        identDict["Legal"] = tempData.split(" ")[0]
        if identDict["Legal"][0] not in ["7","8"]: identDict["Legal"] = identDict["Legal"][1:]
        identDict["Packet"] = tempData.split(" ")[1][1:]
        identDict["Doctitle"] = " ".join(tempData.split(" ")[2:-2])
        identDict["Page"] = tempData.split(" ")[-1]
        identDict["Raw11"] = tempData
        if debug:
            print(tempData)
            print(identDict)
            cropRegion.save("DebugTest.png")
    elif mode == 12:
        cropRegion = testImage.crop((int(360*ratio1),int(3130*ratio2),int(2000*ratio1),int(3500*ratio2)))
        tempData = pytesseract.image_to_string(binarize(cropRegion),config = "--psm {}".format(ocrMode)).split("\n")
        if len(tempData[0].split(" ")) < 5: tempData = tempData[1:]
        tempData = " ".join(tempData[:2]).replace(".","")
        if debug: 
            print(tempData)
            cropRegion.save("DebugTest.png")
        identDict["Legal"] = tempData.strip().split(" ")[0]
        if identDict["Legal"][0] not in ["7","8"]: identDict["Legal"] = identDict["Legal"][1:]
        identDict["Packet"] = tempData.split(" ")[1][1:]
        identDict["Doctitle"] = " ".join([a for a in tempData.split(" ") if a.isalpha() and a.isupper() and a != "G"])
        try: identDict["Page"] = re.search("^\d ",tempData[::-1]).group(0).strip()
        except: identDict["Page"] = re.search(" \d ",tempData[::-1]).group(0).strip()
        identDict["Raw12"] = tempData
        if debug:
            print(tempData)
            print(identDict)
            #cropRegion.save("DebugTest.png")
    elif mode == 13:
        cropRegion = testImage.crop((int(360*ratio1),int(3200*ratio2),int(2000*ratio1),int(3500*ratio2)))
        tempData = pytesseract.image_to_string(binarize(cropRegion),config = "--psm {}".format(ocrMode)).split("\n")
        print(tempData)
        if savePage:
            cropRegion.save("debugImg/DebugTest{}.png".format(savePage))
    elif mode == 1:
        #Caseno: 1500-2200x, 1100-1200y
        cropRegion = testImage.crop((int(1500*ratio1),int(1200*ratio2),int(2180*ratio1),int(1400*ratio2)))
        tempData = pytesseract.image_to_string(binarize(cropRegion),config = "--psm {}".format(ocrMode))
        identDict["case#"] = tempData.split("\n")[0].split(" ")[-1]
        identDict["Raw1"] = tempData
        if debug:
            print(tempData)
            #cropRegion.save("DebugTest.png")        
    elif mode == 2:
        #Partybox (tentative): 400-1500x, 1100-1900y
        cropRegion = testImage.crop((int(400*ratio1),int(1150*ratio2),int(1500*ratio1),int(2200*ratio2)))
        tempData = pytesseract.image_to_string(binarize(cropRegion),config = "--psm {}".format(ocrMode))
        if debug:
            print(tempData)
            #cropRegion.save("DebugTest.png")        
        identDict["Plaintiff"] = quickPartyCleanup(tempData.split("Plaintiff")[0])
        identDict["Defendant"] = quickPartyCleanup(tempData.split("Plaintiff")[1].split("Defendant")[0])
        identDict["Garnishee"] = quickPartyCleanup(tempData.split("Defendant")[1].split("Garnishee")[0])
        identDict["Raw2"] = tempData
    elif mode == 3:
        #Courtheader: 400-2200x, 900-1100y
        cropRegion = testImage.crop((int(400*ratio1),int(900*ratio2),int(2200*ratio1),int(1200*ratio2)))
        tempData = pytesseract.image_to_string(binarize(cropRegion),config = "--psm {}".format(ocrMode))
        identDict["County"] = tempData.split("\n")[0].split(" COUNTY ")[0].split(" THE ")[1]
        identDict["Level"] = tempData.split("\n")[0].split(" COUNTY ")[1].split(" COURT")[0]
        identDict["Raw3"] = tempData
        if "DIVISION" in tempData.split("\n")[1]: identDict["Division"] = tempData.split("\n")[1].split(" DIVISION")[0]
        if debug:
            print(tempData)
            #cropRegion.save("DebugTest.png")        
    elif mode == 4:
        #Docname: 1500-2200x, 1380-1700y
        cropRegion = testImage.crop((int(1500*ratio1),int(1400*ratio2),int(2200*ratio1),int(2000*ratio2)))
        tempData = pytesseract.image_to_string(binarize(cropRegion),config = "--psm {}".format(ocrMode))
        #print(tempData)
        identDict["FilingName"] = " ".join([x.replace("| ","") for x in tempData.split("\n") if x != ""])
        identDict["Raw4"] = tempData
        if debug:
            print(tempData)
            #cropRegion.save("DebugTest.png")
    #regionGrid(tempPage,rowlist = [950,1010,1090,1150,1180,1240,1290,1340],collist = [1680,1950,2100])
    elif mode == 5:
        #Info from writs: total (1680,950,2100,1010), princ (1680,1090,1950,1150), int (1680,1180,1950,1240), fees (1680,1290,1950,1340)
        cropzones = [(1680,950,2100,1010),(1680,1090,1950,1150),(1680,1180,1950,1240),(1680,1290,1950,1340)]
        tempData = [pytesseract.image_to_string(binarize(testImage.crop(x))) for x in cropzones]
        identkeys = ["Total","Princ","Int","Fees"]
        for x in range(4):
            identDict[identkeys[x]] = tempData[x].strip()
        if debug:
            print(tempData)
    #if len(identDict["Legal"]) == 7 and identDict["Legal"][1] == "8": identDict["Legal"] = identDict["Legal"][1:]
    #Partybox is harder, it might be of variable height. We know it starts at 1100,400 and goes to 1500,400, but ymax might be 1900 or it might be larger.
    #return [pytesseract.image_to_string(binarize(region1),config = "--psm 7"),pytesseract.image_to_string(binarize(region2),config = "--psm 6")]
    return identDict

def regionGrid(inPage,horizontalRes = 100,verticalRes = 100,xoffset = 0, yoffset = 0, normalize = False,normRatio = [2550,3500],collist = [],rowlist = [],ystart = 0, 
               xstart = 0, ycutoff = 0, xcutoff = 0):
    #Given a fitz page this converts it to an image, then draws red lines on it to aid in finding regions where important text occurs by hand
    testPage = inPage.get_pixmap(dpi = 300)
    #Normalization ratio?
    if normalize:
        ratio1 = testPage.width / normRatio[0]
        ratio2 = testPage.height / normRatio[1]
    else:
        ratio1 = 1
        ratio2 = 1
    res1 = int(horizontalRes * ratio1)
    res2 = int(verticalRes * ratio2)
    testImage = Image.frombytes("RGB",[testPage.width,testPage.height],testPage.samples)
    testPixels = testImage.load()
    for x in range(xstart,(testPage.width - xcutoff)):
        if len(collist) == 0:
            if (x + xoffset)%res1 == 0:
                for y in range(ystart,(testPage.height - ycutoff)):
                    testPixels[x,y] = (255,0,0)
        else:
            if (x + xoffset) in collist:
                for y in range(ystart,(testPage.height - ycutoff)):
                    testPixels[x,y] = (255,0,0)
    for y in range(ystart,(testPage.height - ycutoff)):
        if len(rowlist) == 0:
            if (y + yoffset)%res2 == 0:
                for x in range(xstart,(testPage.width - xcutoff)):
                    testPixels[x,y] = (255,0,0)
        else:
            if (y+yoffset) in rowlist:
                for x in range(xstart,(testPage.width - xcutoff)):
                    testPixels[x,y] = (255,0,0)
    testImage.save("gridTest.png")

def splitGuesses(inFile,mode,debug = False):
    #This uses focusDown in modes 0, 10, 11, or 12 to look at a document's footer and make guesses where to cut a scan containing a number of incoming documents
    testDoc = fitz.open("inputData/{}".format(inFile))
    guessList = []
    for x,page in enumerate(testDoc):
        tempData = focusDown(page,mode,debug = debug)
        if debug:
            print(tempData["Legal"],tempData["Page"])
        if tempData["Page"] == "1":
            guessList.append([tempData["Legal"],tempData["Doctitle"],x])
    return guessList

def docIndexes(inFile,mode = 0,width = 1):
    #This looks at a document using focusDown in modes 0,10,11,12 to find the legal numbers on every page in the document
    testDoc = fitz.open("inputData/{}".format(inFile))
    guesslist = []
    for x,page in enumerate(testDoc):
        if x%width == 0:
            tempdata = focusDown(page,mode)
            guesslist.append(tempdata["Legal"])
    return guesslist

def quickPartyCleanup(party):
    #This is a quick cleanup function to clean up OCR input of parties to a case
    splitparty = party.split("\n")
    newparty = []
    for line in splitparty:
        if line not in ["/s )","","vs. )",")"]:
            newparty.append(line.replace(", )","").replace(" )","").replace(" }",""))
    return " ".join(newparty)

def smarterSplit(inDocument):
    #Reads through a PDF and looks at document titles and page numbers to determine where to split the document
    print("Entering split")
    docslist = []
    #These are the expected titles, page numbers, and lengths of possible documents
    docpageOrder = {"JUDGMENT AND ORDER TO PAY":{1:2,3:2},"CERTIFICATION OF GARNISHMENT COSTS":{5:1},"FIRST ANDSWER TO WRIT":{1:4},"APPLICATION FOR WRIT OF GARNISHMENT":{1,2},
                    "EXEMPTION CLAIM":{6:2},"WRIT OF GARNISHMENT":{1:5}}
    testDoc = fitz.open(inDocument)
    currentPage = 0
    print("Entering loop")
    for x,page in enumerate(testDoc):
        quickIds = focusDown(page,0)
        print(x,quickIds["Doctype"])
        if quickIds["Doctype"] in docpageOrder.keys():
            #print(quickIds["Page"])
            if int(quickIds["Page"]) in docpageOrder[quickIds["Doctype"]].keys():
                #We've found the front page of something!
                print("Found new document at {}".format(x))
                quickIds = quickIds|focusDown(page,1)|focusDown(page,4)
                splitpages = (x,x+docpageOrder[quickIds["Doctype"]][int(quickIds["Page"])])
                docslist.append([quickIds["case#"],quickIds["FilingName"],splitpages])
    return docslist

def getInfo(docname,pagenum,mode,debug = False):
    #This is a wrapper to use focusDown to get data out of a PDF in other files
    testDoc = fitz.open(docname)
    testPage = testDoc[pagenum]
    return focusDown(testPage,mode,debug = debug)

def testRun():
    #Obsolete test function
    for doc in ["1st Answers.pdf","Applications.pdf","Exemptions.pdf","Writs.pdf"]:
        testDoc = fitz.open("inputData/New Garn Processing/Final Section/{}".format(doc))
        testPage = testDoc[0]
        focusDown(testPage,4,debug = True)

def fasterSplit(inFile,width):
    #First test run at using focusDown to split, now obsolete
    docslist = []
    testDoc = fitz.open(inFile)
    currentPage = 0
    end = 0
    print("Entering loop")
    for x,page in enumerate(testDoc):
        print(x,x%width)
        if x%width == 0:
            if x != 0:
                docslist.append([docnum,(start,x)])
            #New document
            start = x
            docnum = focusDown(page,0,debug = True)["Legal"]
    docslist.append([docnum,start,x])
    return docslist

def newCaseSplit(inFile,indexes = [],debug = False):
    #Obsolete early splitter draft
    docsList = []
    testDoc = fitz.open("inputData/{}".format(inFile))
    print("Entering loop")
    for x,page in enumerate(testDoc):
        if len(indexes) == 0 or x in indexes:
            testData = focusDown(page,0,debug = debug)
            #print(testData)
            if "Page" in testData.keys():
                if testData["Page"] == "1":
                    docsList.append([testData["Doctitle"],testData["Legal"],x])
    docsList.append(["","",len(testDoc)])
    return docsList

def certGCsplit(inFile,data,batchnum = 0):
    #Specific case splitter for certification of garnishment costs (a 1-page document) to test the theory of the splitter
    testDoc = fitz.open("inputData/{}".format(inFile))
    tempReader = PyPDF2.PdfReader("inputData/{}".format(inFile))
    debugData = []
    for x,page in enumerate(testDoc):
        testData = focusDown(page,0)
        print(testData["Legal"])
        debugData.append(testData["Legal"])
        tempMerger = PyPDF2.PdfMerger()
        tempMerger.append(tempReader,pages = (x,x+1))
        for y in data:
            if y["Legal"] == testData["Legal"] and y["FolderName"] in os.listdir("outputData/Batch{}".format(batchnum)):
                tempMerger.write("outputData/Batch{}/{}/08 {} Cert Garn Costs.pdf".format(batchnum,y["FolderName"],y["case#"]))
    return debugData

def numcheck(numlist):
    #Helper function to correct for errors in a list of strings that should be identical
    quickcheck = True
    for num in numlist:
        if num != numlist[0]: quickcheck = False
    if quickcheck: return numlist[0]
    else:
        cnum = ""
        for num in numlist:
            if num.isnumeric(): cnum = num
        return cnum
        #Candidate num is the first number that's all digits
        

def gensplitter(data,inFiles,batchnum = 0,debug = False):
    #Takes the basics of smarterSplit, and integrates splitting the PDFs and saving them in the correct location
    #Expected doc titles, page numbers, and the title to save them as
    expectedDocs = [["1","JUDGMENT AND ORDER","00 {} Motion.pdf",""],["3","JUDGMENT AND ORDER","01 {} Order.pdf",""],["1","MOTION FOR ORDER GRANTING","00 {} Motion.pdf",""],
                    ["1","ORDER GRANTING ADDITIONAL","01 {} Order.pdf","MOTION"],["1","NOTICE OF WITHDRAWAL","04 {} WDS.pdf",""],["1","DECLARATION FOR ORDER","02 {} Dec re Ext.pdf",""],
                    ["1","DECLARATION REGARDING","03 {} Dec re Int.pdf",""],["5","OF GARNISHMENT COSTS","08 {} Cert Garn Costs.pdf",""]]
    #For each file being split:
    print("Running on {}".format(inFiles))
    testDoc = fitz.open("inputData/{}".format(inFiles))
    tempReader = PyPDF2.PdfReader("inputData/{}".format(inFiles))
    debugData = []
    splitlist = []
    print("Splitting:")
    lchecklist = []
    for x,page in enumerate(testDoc):
        print("Page {} of {}".format(x+1,len(testDoc)))
        testData = focusDown(page,12,debug = debug)
        lchecklist.append(testData["Legal"])
        bestdoctitle = Utilities.bestmatch(testData["Doctitle"],[a[1] for a in expectedDocs])
        #if debug: print("Best match: {} at {:.3f}%".format(bestdoctitle[0],bestdoctitle[1] * 100))
        for doc in expectedDocs:
            if testData["Page"] == doc[0] and doc[1] == bestdoctitle[0]:
                #print(testData["Page"],doc[0],doc[1],testData["Doctitle"])
                if not doc[3] or doc[3] not in testData["Doctitle"]:
                    splitlist.append([doc[2],x])
    splitlist.append([testData["Legal"],len(testDoc)])
    print(splitlist)
    print("Writing:")
    for x in range(len(splitlist) - 1):
        print("Doc {} of {}".format(x+1,len(splitlist) - 1))
        docrange = (splitlist[x][1],splitlist[x+1][1])
        checkednum = numcheck(lchecklist[splitlist[x][1]:splitlist[x+1][1]])
        #print(checkednum)
        #Docname references page number and doctitle?
        for case in data:
            #print(case["Legal"])
            if case["Legal"] == checkednum and case["FolderName"] in os.listdir("outputData/Batch{}".format(batchnum)):
                print("Writing to outputData/Batch{}/{}/{}".format(batchnum,case["FolderName"],splitlist[x][0].format(case["case#"])))
                if not debug:
                    merger = PyPDF2.PdfMerger()
                    merger.append(tempReader,pages = docrange)
                    merger.write("outputData/Batch{}/{}/{}".format(batchnum,case["FolderName"],splitlist[x][0].format(case["case#"])))

def newGensplitter(data,inFile,mode,batchnum = 0,debug = False):
    #Replacement for gensplitter using the new smarter OCRfooter in place of focusDown
    #Expected documents: Note: May need to debug, some slight changes to footers lately
    expectedDocs = [["1","JUDGMENT AND ORDER","00 {} Motion.pdf",""],["3","JUDGMENT AND ORDER","01 {} Order.pdf",""],["1","MOTION FOR ORDER GRANTING","00 {} Motion.pdf",""],
                    ["1","ORDER GRANTING ADDITIONAL","01 {} Order.pdf","MOTION"],["1","NOTICE OF WITHDRAWAL","04 {} WDS.pdf",""],["1","DECLARATION FOR ORDER","02 {} Dec re Ext.pdf",""],
                    ["1","DECLARATION REGARDING","03 {} Dec re Int.pdf",""],["5","OF GARNISHMENT COSTS","08 {} Cert Garn Costs.pdf",""]]
    lindex = {a["Legal"]:a for a in data}
    tempReader = PyPDF2.PdfReader("inputData/{}".format(inFile))
    print("Splitter running")
    splitList = []
    pagelen = len(PyPDF2.PdfReader("inputData/{}".format(inFile)).pages)
    for x in range(pagelen):
        #For each page: get OCRfooter data
        testData = OCRfooter("inputData/{}".format(inFile),x,mode = mode,debug = debug)
        #Find best match for name and legal number to expected values to correct for errors
        bestdoctitle = Utilities.bestmatch(testData["Doctitle"],[a[1] for a in expectedDocs])[0]
        bestlegal = Utilities.bestmatch(testData["Legal"],[a for a in lindex.keys()])[0]
        print(bestdoctitle,bestlegal,testData["Page"])
        for doc in expectedDocs:
            if bestdoctitle == doc[1] and testData["Page"] == doc[0]:
                if not doc[3] or doc[3] not in bestdoctitle:
                    splitList.append([bestlegal,doc[2],x])
    splitList.append(["a","b",pagelen])
    print(splitList)
    print("Writing:")
    for x in range(len(splitList) - 1):
        print("Doc {} of {}".format(x+1,len(splitList) - 1))
        docrange = (splitList[x][2],splitList[x+1][2])
        lnum = splitList[x][0]
        if lindex[lnum]["FolderName"] in os.listdir("outputData/Batch{}".format(batchnum)):
            print("Writing to outputData/Batch{}/{}/{}".format(batchnum,lindex[lnum]["FolderName"],splitList[x][1].format(lindex[lnum]["case#"])))
            if not debug:
                merger = PyPDF2.PdfMerger()
                merger.append(tempReader,pages = docrange)
                merger.write("outputData/Batch{}/{}/{}".format(batchnum,lindex[lnum]["FolderName"],splitList[x][1].format(lindex[lnum]["case#"])))
        

def simpleSplitter(inFile,data,name,width = 1,debug = False):
    #Another splitter draft, this one expects a PDF containing all width-page documents
    testDoc = fitz.open("inputData/{}".format(inFile))
    tempReader = PyPDF2.PdfReader("inputData/{}".format(inFile))
    indexData = {a["Legal"]:a["case#"] for a in data}
    splitlist = []
    print("Splitting:")
    for x,page in enumerate(testDoc):
        if x%width == 0:
            print("Page {} of {}".format(x+1,len(testDoc)))
            testData = focusDown(page,12,debug = debug)
            splitlist.append([testData["Legal"],x])
    splitlist.append(["",len(testDoc)])
    #numchecklist = {}
    #for a in splitlist: numchecklist = Utilities.dictAppend(numchecklist,a[0],a[1])
    print("Writing")
    for x in range(len(splitlist) - 1):
        print("Doc {} of {}".format(x+1,len(splitlist) - 1))
        docrange = (splitlist[x][1],splitlist[x+1][1])
        merger = PyPDF2.PdfMerger()
        merger.append(tempReader,pages = docrange)
        merger.write("outputData/{}".format(name.format(indexData[splitlist[x][0]])))
        
def readDebug(inFile):
    #Debugger to see if focusDown is cropping to the right place
    testDoc = fitz.open("inputData/{}".format(inFile))
    tempReader = PyPDF2.PdfReader("inputData/{}".format(inFile))
    for x,page in enumerate(testDoc):
        focusDown(page,13,debug = True,savePage = str(x))

def extSplit(inFile,data,batchnum = 0):
    #Dedicated splitter draft for extension motions and orders
    testDoc = fitz.open("inputData/{}".format(inFile))
    tempReader = PyPDF2.PdfReader("inputData/{}".format(inFile))
    debugData = []
    splitlist = []
    print("Splitting:")
    for x,page in enumerate(testDoc):
        print("Page {} of {}".format(x+1,len(testDoc)))
        testData = focusDown(page,11)
        if testData["Page"] == "1":
            splitlist.append([testData["Legal"],testData["Doctitle"],x])
    splitlist.append(["","",len(testDoc)])
    print("Writing:")
    for x in range(len(splitlist) - 1):
        print("Doc {} of {}".format(x+1,len(splitlist) - 1))
        docrange = (splitlist[x][2],splitlist[x+1][2])
        if "MOTION" in splitlist[x][1]: title = "00 {} Motion.pdf"
        else: title = "01 {} Order.pdf"
        for y in data:
            if y["Legal"] == splitlist[x][0] and y["FolderName"] in os.listdir("outputData/Batch{}".format(batchnum)):
                merger = PyPDF2.PdfMerger()
                merger.append(tempReader,pages = docrange)
                merger.write("outputData/Batch{}/{}/{}".format(batchnum,y["FolderName"],title.format(y["case#"])))
    return splitlist

def expreportscanner(inFile,indexes = [],cols = []):
    #Takes a PDF report table and converts it to text
    testDoc = fitz.open("inputData/{}".format(inFile))
    rowslist = []
    colslist = [70,200,350,750,930,1090,1250,1330,1490,1650,1810,1885,1935]
    for x,page in enumerate(testDoc):
        if len(indexes) == 0 or x in indexes:
            print("On page: {}".format(x))
            pageInterim = page.get_pixmap(dpi = 300)
            testImage = Image.frombytes("RGB",(pageInterim.width,pageInterim.height),pageInterim.samples)
            for y in range(200,pageInterim.height - 350):
                if (y+22) % 46 == 0 and y+46 < pageInterim.height:
                    start = time.time()
                    newrow = []
                    #Then we have a row
                    ystart = y
                    yend = y+46
                    for a in range(12):
                        if len(cols) == 0 or a in cols:
                            xstart = colslist[a]
                            xend = colslist[a+1]
                            testval = pytesseract.image_to_string(testImage.crop(box = (xstart,ystart,xend,yend)),config = "--psm 6").strip()
                            #print((y+20)/46,a,testval)
                            newrow.append(testval)
                    if newrow[0] not in ["","legal"]:
                        rowslist.append(newrow)
                    print("Checked row {} in {:.4f}s".format(int((y+22)/46) - 4,time.time() - start))
    return rowslist

def cropTest(startPage,startX,startY,endX,endY):
    #Test cropping a page to the indicated region
    pixmap = startPage.get_pixmap(dpi = 300)
    testImage = Image.frombytes("RGB",[pixmap.width,pixmap.height],pixmap.samples)
    outImage = testImage.crop((startX,startY,endX,endY))
    outImage.save("croptest.png")

def noteUpdate():
    #Gets data from a writ of garnishment and produce a text file containing correctly formatted PAMAR notes for it
    casepath = "{}/080923/NewGarns".format(filedpath)
    outFile = open("noteUpdate.txt","w")
    for case in os.listdir(casepath):
        print("Running on: {}".format(case))
        tempDoc = fitz.open("{}/{}".format(casepath,case))
        tempPage = tempDoc[3]
        tempData = focusDown(tempPage,5)
        tempData.update(focusDown(tempPage,0))
        #Output format:
        #lnum\n WRIT X, PRINC, INT, FEES\n PIX ON WRIT ...NXS
        outFile.write("{}\nWRIT X, PRINC {}, INT {}, FEES {}\nPIF ON WRIT {}...NXS\n\n".format(tempData["Legal"],tempData["Princ"],tempData["Int"],tempData["Fees"],tempData["Total"]))
    outFile.close()

def gridWrapper(docpath,pagenum,vRes = 112,hRes = 60,yOff = 30,xStart = 350,yStart = 300,xCutoff = 350,yCutoff = 350):
    #Simpler call for regionGrid
    testDoc = fitz.open(docpath)
    testPage = testDoc[pagenum]
    regionGrid(testPage,verticalRes = vRes,horizontalRes = hRes, yoffset = yOff, xstart = xStart,ystart = yStart,xcutoff = xCutoff, ycutoff = yCutoff)
    
def gridchopper(docpath,pagenum,vres = 112,hres = 60,boundingbox = (350,300,2200,3000),startindex = 0):
    #Chops images down to single characters for OCR testing purposes
    testDoc = fitz.open(docpath)
    testPage = testDoc[pagenum]
    testData = testPage.get_pixmap(dpi = 300)
    testImage = Image.frombytes("RGB",(testData.width,testData.height),testData.samples)
    croppedImage = testImage.crop(boundingbox)
    cindex = startindex
    rows = int(croppedImage.size[1] / vres)
    cols = int(croppedImage.size[0] / hres)
    for y in range(rows):
        for x in range(cols):
            subboundary = (x*hres,y*vres,(x+1)*hres,(y+1)*vres)
            subimage = croppedImage.crop(subboundary)
            subimage.save("ocrtest/scrambledata/img {}.png".format(cindex))
            cindex += 1
    return cindex

def chopwrapper(docpath,vres = 112,hres = 60,boundingbox = (350,300,2200,3000)):
    #Simpler call for gridchopper
    testDoc = fitz.open(docpath)
    index = 0
    for x in range(len(testDoc)):
        index = gridchopper(docpath,x,vres = vres, hres = hres, boundingbox = boundingbox, startindex = index)

def ABCsheet(comments,county,sup,district,sheetnum):
    #Fills in information in an ABC filing coversheet
    #The sheet converted back is way too big, but at the moment all we need to do with it is print
    testDoc = fitz.open("programData/ABC Sheet.pdf")
    testPage = testDoc[0]
    testMap = testPage.get_pixmap(dpi = 300)
    testImage = Image.frombytes("RGB",(testMap.width,testMap.height),testMap.samples)
    #Safe box regions:
    #Documents: x 380-2200, y 920-1050
    #County: x 240-400, y 1840-1920
    #Superior: x 430-690, y 1840-1920
    #District: x 710-1000, y 1840-1920
    testDraw = ImageDraw.Draw(testImage)
    docpoints = [(380,920),(240,1860),(430,1860),(720,1860)]
    testDraw.text(docpoints[0],comments,font = ImageFont.truetype("arial.ttf",48),fill = (0,0,0,255))
    testDraw.text(docpoints[1],county,font = ImageFont.truetype("arial.ttf",32),fill = (0,0,0,255))
    testDraw.text(docpoints[2],sup,font = ImageFont.truetype("arial.ttf",32),fill = (0,0,0,255))
    testDraw.text(docpoints[3],district,font = ImageFont.truetype("arial.ttf",32),fill = (0,0,0,255))
    testImage.save("outSheet{}.pdf".format(sheetnum))

def imgtest():
    #Debugging test to make sure I could draw text on images
    testDoc = fitz.open("programData/ABC Sheet.pdf")
    testPage = testDoc[0]
    testMap = testPage.get_pixmap(dpi = 300)
    testImage = Image.frombytes("RGB",(testMap.width,testMap.height),testMap.samples)
    testDraw = ImageDraw.Draw(testImage)
    testDraw.text((380,920),"Hello, world!",font = ImageFont.truetype("arial.ttf",48),fill = (0,0,0,255))
    testDraw.rectangle(((380,920),(750,1050)),fill = (255,0,0,127))
    testImage.save("testSheet.pdf")

def croptest(docname,docpage,boundingbox,outimg):
    #Slight upgrade on cropTest
    testDoc = fitz.open(docname)
    testPage = testDoc[docpage]
    testMap = testPage.get_pixmap(dpi = 300)
    testImage = Image.frombytes("RGB",(testMap.width,testMap.height),testMap.samples)
    testCrop = testImage.crop(box = boundingbox)
    testCrop.save(outimg)

def cropwrapper(docslist,boundingbox):
    #Runs croptest on a number of documents
    for x in range(len(docslist)):
        croptest(docslist[x],0,boundingbox,"debugImg/debugImg {}.png".format(x))

def lineavg(inImg,inY):
    #Gets the average value of a line in an image
    pixels = inImg.load()
    imgSum = [0,0,0]
    for col in range(inImg.size[0]):
        for x in range(3):
            imgSum[x] += pixels[col,inY][x]
    for x in range(3):
        imgSum[x] = imgSum[x] / inImg.size[0]
    return sum(imgSum) / 3

def weightedLineavg(inImg,inY):
    pixels = inImg.load()
    binaryavg = [1 if sum(pixels[a,inY]) / 3 > 200 else 0 for a in range(inImg.size[0])]
    return sum(binaryavg) / len(binaryavg)

def colavg(inImg,inX):
    #Gets the average value of a column in an image
    pixels = inImg.load()
    imgSum = [0,0,0]
    for row in range(inImg.size[1]):
        for y in range(3):
            imgSum[y] += pixels[inX,row][y]
    for y in range(3):
        imgSum[y] = imgSum[y] / inImg.size[1]
    return sum(imgSum) / 3

def scrubber(inImg):
    #Unfinished tool to look for isolated pixels to scrub
    lineavglist = [lineavg(inImg,a) for a in range(inImg.size[1])]
    colavglist = [colavg(inImg,a) for a in range(inImg.size[0])]
    linebreaks = [a for a in range(len(lineavglist) - 1) if (int(lineavglist[a]) == 255 and int(lineavglist[a+1]) != 255)
                  or (int(lineavglist[a]) != 255 and int(lineavglist[a+1]) == 255)]
    colbreaks = [a for a in range(len(colavglist) - 1) if (int(colavglist[a]) == 255 and int(colavglist[a+1]) != 255) or
                 (int(colavglist[a]) != 255 and int(colavglist[a+1]) == 255)]

def debugGrid(img,xlist = [],ylist = [],savefileName = "gridtest0.png"):
    drawImage = img.load()
    for x in xlist:
        #Draw vertical lines at given x-coords
        for y in range(img.size[1]):
            try:
                drawImage[x,y] = (255,0,0)
            except:
                pass
    for y in ylist:
        #Draw horizontal lines at given y-coords
        for x in range(img.size[0]):
            drawImage[x,y] = (255,0,0)
    img.save("debugImg/{}".format(savefileName))

def loadImgFromPage(inPDF,page = 0,dpi = 300):
    testDoc = fitz.open(inPDF)
    pageImg = testDoc[page].get_pixmap(dpi = dpi)
    pilImage = Image.frombytes("RGB",[pageImg.width,pageImg.height],pageImg.samples)
    return pilImage

def lineSplit(inImg):
    #Identify which lines are all white pixels for purposes of separating a file into separate lines of text
    width,height = inImg.size
    pixels = inImg.load()
    pxrowlist = []
    lineRegions = []
    for y in range(height):
        whiteLine = True
        for x in range(width):
            if pixels[x,y] != (255,255,255):
                whiteLine = False
        pxrowlist.append(whiteLine)
    #Use regions with all white pixels to define lines of text
    start = 0
    for x in range(len(pxrowlist) - 1):
        if pxrowlist[x] and not pxrowlist[x+1]:
            start = x
        elif not pxrowlist[x] and pxrowlist[x+1] and (x - start) / height > 0.010:
            #horizontalbox = (0,start-2,width,x+4)
            lineRegions.append((start - 2,x + 4))
    return lineRegions

def colSplit(inImg):
    #Identify which columns are all white pixels for purposes of cropping lines produced via linesplit
    width,height = inImg.size
    #print(width)
    pixels = inImg.load()
    pxcollist = []
    colRegions = []
    for x in range(width):
        whiteCol = True
        for y in range(height):
            if pixels[x,y] != (255,255,255):
                whiteCol = False
        pxcollist.append(whiteCol)
    #Use regions of all white pixels to define columns
    start = 0
    for x in range(len(pxcollist) - 1):
        if pxcollist[x] and not pxcollist[x+1]:
            start = x
        elif not pxcollist[x] and pxcollist[x+1] and (x - start) / width > 0.0075:
            colRegions.append((start,x))
    return colRegions

def avgsplit(inImg):
    #Cutting down on DPI dramatically improves runtime. Problem remains: Accuracy?
    #Accuracy test: Problem: Threshold?
    #Threshold isn't the problem. Let's look at the text line threshold instead and see if that helps at all.
    choplist = []
    linelist = [True if weightedLineavg(inImg,a) < 0.95 else False for a in range(inImg.size[0])]
    start = 0
    for a in range(len(linelist) - 1):
        if linelist[a+1] and not linelist[a]:
            start = a
        elif linelist[a] and not linelist[a+1]:
            choplist.append((start - 2,a + 4))
    return choplist

def chopUnzip(choplist):
    unzipa = [a[0] for a in choplist]
    unzipb = [b[1] for b in choplist]
    ziplist = []
    for a in range(len(unzipa)):
        ziplist.append(unzipa[a])
        ziplist.append(unzipb[a])
    return ziplist
    
def OCRfooter(inDoc,pagenum,mode = 0,debug = False):
    #Get footer data from document
    outDict = {}
    #First, get footer indexes: load in the image at 50dpi and use lineSplit/colSplit to get the footer box
    testPage = loadImgFromPage(inDoc,pagenum,dpi = 50)
    list1 = lineSplit(testPage)
    heightcrop = testPage.crop((0,list1[-1][0],testPage.size[0],list1[-1][1]))
    list2 = chopUnzip(colSplit(heightcrop))
    #Next, load in document at 300dpi, crop to footer dimensions, and OCR
    OCRpage = loadImgFromPage(inDoc,pagenum,dpi = 300)
    cropbox = ((list2[0] * 6) - 3,list1[-1][0] * 6,(list2[-1] + 2) * 6,list1[-1][1] * 6)
    footer = OCRpage.crop(box = cropbox)
    footerData = pytesseract.image_to_string(binarize(footer),config = "--psm 6").replace(".","")
    #if debug: print(footerData)
    #Finally, parse OCRed data
    if mode == 0: #Run on PAMAR outputted files
        footerData = footerData.split("\n")
        patternlist = [("Legal","^L\d+ ",("L"," ")),("Packet"," P\d+ ",("P"," ")),("Pagenum","Pg \d$",("Pg "))]
        try:
            for pattern in patternlist:
                match = re.search(pattern[1],footerData[0]).group(0)
                for a in pattern[2]: match = match.replace(a,"")
                outDict[pattern[0]] = match
        except:
            print("RE match failed, guessing on space chop")
            outDict["Legal"] = footerData[0].split(" ")[0].replace("L","")
            outDict["Packet"] = footerData[0].split(" ")[1].replace("P","")
            outDict["Page"] = footerData[0].split(" ")[-1]
        outDict["Doctitle"] = footerData[1]
        outDict["Raw"] = footerData
    elif mode == 1: #Run on my outputs
        footerData = footerData.replace("\n"," ")
        wordlist = [a for a in footerData.split(" ") if a != ""]
        outDict["Legal"] = wordlist[0].replace("L","")
        outDict["Packet"] = wordlist[1].replace("P","")
        outDict["Page"] = wordlist[-1]
        outDict["Doctitle"] = " ".join(wordlist[2:-1])
        outDict["Raw"] = footerData
    if debug: print(outDict)
    return outDict
    


    
    
    

    
    

