#ArsSegmenter
#Page Segmentation Tool
#This was an attempt to segment PDF pages for OCR purposes.
#It doesn't actually work.

import Ars_Image
import os
import random
import numpy as np
import fitz
from PIL import Image
import math
from time import time
import sys

#Page segmentation tool
#Implements bottom-up document layout analysis for OCR purposes
pdfPath = r"\\diskstation\Legal Team\TEAM MEMBER DIRECTORIES\NXS\Python Document Processing\Declaration Maker\inputData"

def segmentPage(in_img):
    #This is the main function of this module. It separates an inputted image into lines of text, in order to isolate regions to be OCRed.
    #The input is an image file name, and the output is a list of PIL images
    char_locs = coRegions(in_img)

def coRegions(in_img,debug = False):
    #This function attempts to divide a binarized image into regions of connected black pixels
    #It returns a list of the bounding boxes for the regions in the image, in the form (left,top,right,bottom)
    temp_img = quickPad(in_img)
    pixels = temp_img.load()
    w,h = in_img.size
    checked = []
    regions = []
    #First: Iterate through the image, width first, height second.
    for x in range(w):
        print(x)
        for y in range(h):
            if (x,y) not in checked:
                if pixels[x,y] != (255,255,255):
                    #On finding a black pixel, use the neighbors function to find all connected pixels, then add their values to checked
                    region = neighbors(pixels,x,y,[(x,y)])
                    checked += region
                    #Then find the bounding box of that object, and add it to regions
                    region_box = (min([a[0] for a in region]),min([a[1] for a in region]),max([a[0] for a in region]),max([a[1] for a in region]))
                    regions.append(region_box)
    #Use the comboBox function to correct for multi-region characters (%,?,!,=,')
    regions = comboBox(regions)
    #Debug code: Draw bounding boxes in red, and save the image so we can see what it's doing
    if debug:
        for region in regions:
            for x in range(region[0],region[2]+1):
                for y in range(region[1],region[3]+1):
                    if x in (region[0],region[2]) or y in (region[1],region[3]):
                        pixels[x,y] = (255,0,0)
        temp_img.save("boxtest.png")
    return regions

def lineFinder(in_img):
    #We often operate on pleadings, which have margins bounded with large vertical lines
    #If we could find them and crop to the area between them lots of things would be faster
    #Input is a PIL image, output is a PIL image cropped to the area between the lines
    pixels = in_img.load()
    w,h = in_img.size
    heightlist = []
    for x in range(w):
        for y in range(h):
            if pixels[x,y] == (0,0,0):
                #Then we've found a candidate pixel
                current = y
                while pixels[x,current] == (0,0,0):
                    current += 1
                heightlist.append([x,current - y])
    for a in heightlist:
        print(a)

def quickPad(in_img):
    #Pad image with a one white pixel border
    #Input and output are a PIL Image object
    w,h = in_img.size
    out_img = Image.new("RGB",(w+2,h+2),color = (255,255,255))
    out_img.paste(in_img,(1,1))
    return out_img

def comboBox(box_list,tolerance = 20):
    #Attempts to identify boxes as part of the same character
    #The neighbors function currently misidentifies ",?,!,%,= as separate boxes
    box_centers = {((a[0] + a[2] / 2),(a[1] + a[3] / 2)):a for a in box_list}
    box_dists = {a:{b:0 for b in box_centers} for a in box_centers}
    for a in box_centers:
        for b in box_centers:
            if a != b:
                box_dists[a][b] = math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
    out_boxes = []
    done = []
    for a in box_dists:
        for b in box_dists[a]:
            if box_dists[a][b] < tolerance and a != b and a not in done and b not in done:
                #If we find a possible merge with this method, then we need to make a new bounding box merging the two boxes
                merged_box = (min([box_centers[a][0],box_centers[b][0]]),min([box_centers[a][1],box_centers[b][1]]),max([box_centers[a][2],box_centers[b][2]]),max([box_centers[a][3],box_centers[b][3]]))
                out_boxes.append(merged_box)
                found = True
                done.append(a)
                done.append(b)
        if a not in done:
            #If we didn't find a merge, just add the original box to the out list
            out_boxes.append(box_centers[a])
    return out_boxes

def neighbors(pixels,x,y,neighbor_list):
    #This function recursively gets neighboring non-white pixels of a pixel to find a region of connected pixels
    #Inputs are a PIL Image.load() call, the x,y coordinates of the next point checked, and a list of points in the region so far.
    #On the initial call the only point is the first point you found
    #The return is a list of x,y tuples
    neighbor_candidates = [(a,b) for a in range(x-1,x+2) for b in range(y-1,y+2) if pixels[a,b] != (255,255,255) and not (a == x and b == y)]
    for c in neighbor_candidates:
        if c not in neighbor_list:
            neighbor_list = neighbors(pixels,c[0],c[1],neighbor_list + [c])
    return neighbor_list

def loadImgFromFile(pdf_file,page,in_dpi = 100):
    #This function gets an Image object from a PDF filepath using the fitz library
    #Inputs are a filepath string, page number integer, and resolution integer
    #Output is a PIL Image
    temp_file = fitz.open(pdf_file)
    page_pxmp = temp_file[page].get_pixmap(dpi = in_dpi)
    return Image.frombytes("RGB",[page_pxmp.width,page_pxmp.height],page_pxmp.samples)

def main():
    temp_file = loadImgFromFile("{}/110123 Ords MOs.pdf".format(pdfPath),0)
    temp_file = Ars_Image.binarize(Ars_Image.deNoise(temp_file))
    lineFinder(temp_file)
    #coRegions(Image.open("pleadingtest.png"),True)
    #sys.setrecursionlimit(100000)
    #coRegions(loadImgFromFile("{}/110123 Ords MOs.pdf".format(pdfPath),0,in_dpi = 50),True)
    #Problem: Recursion for segmenting hits depth limit and breaks for large files
    #Or for small files
    #Maybe recursion for segmenting isn't a great idea
    pass

main()