#New OCR module

import scipy
import numpy
import random
import fitz
import time
import os
from PIL import Image

#Pseudocode beforehand

#Basic concept:
#Input layer takes weighted values in from image, then adds bias term, then uses activation function
#Internal layer takes weighted values in from input layer, then adds bias term, then uses activation function
#Output layer takes weighted values in from internal layer, then uses activation function, then gives a number, the answer is the highest number in the outputlist

#Hypothetically part of the problem becomes changing image resolution without too much loss?

#First we need some training data

def getLetters(inDoc,indexes = []):
    testDoc = fitz.open(inDoc)
    startIndex = 0
    print(startIndex)
    for x in range(len(testDoc)):
        if len(indexes) == 0 or x in indexes:
            start = time.time()
            tempPage = testDoc[x].get_pixmap(dpi = 300)
            tempImage = Image.frombytes("RGB",[tempPage.width,tempPage.height],tempPage.samples)
            croppedImage = tempImage.crop(box = (400,900,2150,2600))
            #croppedImage.save("croptest.png")
            startIndex = slicer(croppedImage,startIndex = startIndex,scanthreshold = 200)
            print("Page {} done in {:.4f}".format(x,time.time() - start))
            #print(startIndex)

def slicer(inImage,startIndex = 0,scanthreshold = 0):
    outlist = []
    linelist = horizontalchop(inImage,scanthreshold = scanthreshold)
    #print(len(linelist))
    endindex = 0
    for line in linelist:
        outlist += verticalchop(line,scanthreshold = scanthreshold)
    for x in range(len(outlist)):
        #print(x)
        outlist[x].save("ocrtest/newtrainset/img {}.png".format(startIndex + x))
        endindex = x
    #print(len(outlist))
    return endindex + 1

def horizontalchop(inImage,scanthreshold = 0):
    pixels = inImage.load()
    pxrowlist = []
    horizontalregions = []
    outList = []
    for y in range(inImage.size[1]):
        #print("Row {}".format(y))
        allWhite = True
        for x in range(inImage.size[0]):
            if thresholdcompare(pixels[x,y],scanthreshold) > 0:
                #print("Stopped at px {} with val {}".format(x,pixelavg(pixels[x,y])))
                allWhite = False
        pxrowlist.append(allWhite)
    start = 0
    for x in range(len(pxrowlist) - 1):
        if pxrowlist[x] and not pxrowlist[x+1]:
            start = x
        elif not pxrowlist[x] and pxrowlist[x+1]:
            horizontalregions.append(inImage.crop(box=(0,start,inImage.size[0],x+2)))
    for region in horizontalregions:
        outList.append(region)
    return outList

def verticalchop(inImage,scanthreshold = 0):
    pixels = inImage.load()
    pxcollist = []
    verticalregions = []
    outList = []
    for x in range(inImage.size[0]):
        allWhite = True
        for y in range(inImage.size[1]):
            if thresholdcompare(pixels[x,y],scanthreshold) > 0:
                allWhite = False
        pxcollist.append(allWhite)
    start = 0
    for x in range(len(pxcollist) - 1):
        if pxcollist[x] and not pxcollist[x+1]:
            start = x
        elif not pxcollist[x] and pxcollist[x+1]:
            verticalregions.append(inImage.crop(box=(start,0,x+2,inImage.size[1])))
    for region in verticalregions:
        outList.append(region)
    return outList

def thresholdcompare(tuple1,threshold):
    avg1 = pixelavg(tuple1)
    return avg1<threshold

def imgFromDoc(docname,pg):
    tempDoc = fitz.open(docname)
    tempPage = tempDoc[pg].get_pixmap(dpi = 300)
    tempImage = Image.frombytes("RGB",[tempPage.width,tempPage.height],tempPage.samples)
    return tempImage

def pixelavg(pixel):
    return int((pixel[0] + pixel[1] + pixel[2]) / 3)

#That's probably wrong, but hopefully for now it will work
#Next we need to work out how big of an array of neurons to set up
def sizetest():
    maxx = 0
    maxy = 0
    minx = 10000
    miny = 10000
    for x in os.listdir("ocrtest/newtrainset"):
        if x != "img 740.png":
            try:
                tempimage = Image.open("ocrtest/newtrainset/{}".format(x))
                maxx = max(tempimage.size[0],maxx)
                maxy = max(tempimage.size[1],maxy)
                minx = min(tempimage.size[0],minx)
                miny = min(tempimage.size[1],miny)
            except:
                pass
    
    for x in os.listdir("ocrtest/newtrainset"):
        if x != "img 740.png":
            try:
                tempimage = Image.open("ocrtest/newtrainset/{}".format(x))
                if tempimage.size[0] == maxx:
                    print("{} max width".format(x))
                if tempimage.size[0] == minx:
                    print("{} min width".format(x))
                #if tempimage.size[1] == maxy:
                    #print("{} max height".format(x))
                if tempimage.size[1] == miny:
                    print("{} min height".format(x))
            except:
                pass
    print("Between {},{} and {},{}".format(maxx,maxy,minx,miny))

#Sizerange is 90,49 and 3,10
#If we had a fast way of screening out multi-letter groups...hrm.

#getLetters("inputData/Batch 3 Cert Garn Costs.pdf")

'''
tempFile = fitz.open("inputData/Batch 3 Cert Garn Costs.pdf")
tempPage = tempFile[0].get_pixmap(dpi = 300)
tempImage = Image.frombytes("RGB",[tempPage.width,tempPage.height],tempPage.samples)
croppedImage = tempImage.crop(box = (400,900,2150,2600))
croppedImage.save("testpage.png")
'''