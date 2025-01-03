#KeyInput2
#This is a cleaned up version of the original that uses commands from a text file to run PAMAR through screens for output

import keyboard
import time
import shutil

abspath = r"\\diskstation\Legal Team\TEAM MEMBER DIRECTORIES\NXS\Python Document Processing\Declaration Maker"

def keysquish(index):
    #Keysequences
    #These are commands that get run through the RecordPAMAR function.
    #0: Legal numbers only
    numsonlyK = [""]
    numsonlyO = [-3]
    #1: Old sequence for orders
    decMainKeys = ["","i","e","g","n","s","poe","db","p","writ","governor","2nd"]
    decMainOrder = [1,2,3,0,2,0,2,4,5,6,5,7,5,11,5,10,8,5,9,5,11,2,2]
    #2: Demographic info for mildecs
    milDataKeys = ["","2","11","n","orange","d","1","e"]
    milDataOrder = [-2,1,2,3,4,3,-1,5,1,5,6,7,7,7]
    #3: Basic searches for extensions
    extKeys = ["","e","n","s","governor","i"]
    extOrder = [5,1,2,3,4,1,1]
    #4: Legal screen information for addresses
    labelKeys = ["","e","g","i"]
    labelOrder = [2,0,1,0,1,3,1,1]
    #5: Packet screen information for addresses
    label2Keys = ["","16","5","d","2","e"]
    label2Order = [-2,1,2,3,4,5,5,5]
    #6: New order print commands
    printOrderKeys = ["","22","y"]
    printOrderOrder = [1,0,2]
    #7: Visit only the front screen
    frontCheckKeys = ["","e","i"]
    frontCheckOrder = [2,1,1]
    #8: Tag changes for sending out new suits
    newSuitsKeys = ["","e","y","t","7","LIP","11","0"]
    newSuitsOrder = [3,4,5,6,7,-2,1,2,2]
    #Lists of lists to pull the info to return from by index
    keylist = [numsonlyK,decMainKeys,milDataKeys,extKeys,labelKeys,label2Keys,printOrderKeys,frontCheckKeys,newSuitsKeys]
    orderlist = [numsonlyO,decMainOrder,milDataOrder,extOrder,labelOrder,label2Order,printOrderOrder,frontCheckOrder,newSuitsOrder]
    return (keylist[index],orderlist[index])

#Current indexes: 0 nums only, 1 orders, 2 mildata, 3 extensions, 4 mailing labels (legal), 5 mailing labels (packet)
#6 print new order (docwriter), 7 ready new suit, 8 front screen contents only

#The keysequences for making actual edits are, as yet, still WIP

#Fast function calls to allow the user to quickly call both of the more common uses
    
def oldQuickRecord(filename,mode,inputMethod = 0, fileIndex = 0,debug = False,indexes = []):
    #The original main function of the file.
    #Uses keyboard module to input commands from an input file and from keysquish to give commands to PAMAR
    #Inputs:
    #filename: name of a .txt file in the inputData folder, or a list of strings.
    #mode: keysquish index of commands to use, integer from 0-8
    #inputMethod: 0 if reading filename from a file, 1 if reading from a list of strings
    #fileIndex: Index of the legal/packet number in a filename string, split by spaces
    #debug: If True runs the inputs more slowly so it's easier to see what's happening
    #indexes: If you don't want to run on every line in filename enter the lines to run here
    #No outputs, we get data from this using the AccuTerm frame's text file writer
    keys = keysquish(mode)[0] #Get commands
    keysequence = keysquish(mode)[1] #Get commands
    keyboard.wait("shift") #Pause to allow the user to change window focus into PAMAR
    keyboard.release("shift") #Module sometimes makes shift sticky while running
    if inputMethod == 0: #If reading legal numbers from file, read the file
        inputData = open("inputData/" + filename,"r").readlines()
    elif inputMethod == 1: #If reading legal numbers from an input list of strings
        inputData = filename
    for x in range(len(inputData)):
        #For each input case:
        if len(indexes) == 0 or x in indexes:
            #Find and clean up the lnum for the input
            entry = inputData[x]
            if len(entry.split(" ")) > 1:
                lnum = entry.split(" ")[fileIndex].strip()
            else:
                lnum = entry.strip()
            if lnum[0] not in "0123456789":
                lnum = lnum[1:]
            print(lnum)
            #lnum is the PAMAR identifier, legal or packet, of the cases we're looking for
            #Enter the lnum into PAMAR
            keyboard.write(lnum)
            keyboard.send("enter")
            #For each command:
            for key in keysequence:
                #If the command is a special command (-3 to 0), run it
                if key == 0:
                    keyboard.wait("shift")
                    keyboard.release("shift")
                    keyboard.send("enter")
                elif key == -1:
                    keyboard.send("escape")
                elif key == -2:
                    keyboard.send("enter")
                elif key == -3:
                    keyboard.wait("shift")
                    keyboard.release("shift")
                #Otherwise just enter the indexed command from keys
                else:
                    keyboard.write(keys[key])
                    if debug:
                        time.sleep(1)
                    else:
                        time.sleep(0.1)
                    keyboard.send("enter")
                    print(keys[key])
                    if keys[key] in ["2nd","writ"]:
                        #This search command can have lots of extra lines
                        keyboard.wait("shift")
                        keyboard.release("shift")
    #For modes which capture data move the captured data to the input folder
    defnamelist = [""," Ords Capture.txt"," Ords MilCap.txt","Exts Capture.txt","","",""," Precapture.txt",""]
    if defnamelist[mode] != "":
        try: defname = filename.split(" ")[0] + defnamelist[mode]
        except: defname = "DATE?" + defnamelist[mode]
        filechange = input("Enter a new name for the file, or press enter to confirm the default ({}):".format(defname))
        if filechange != "": defname = filechange
        shutil.move(r"C:\Users\nathanielslivke\Documents\capture.txt",r"{}/inputData/{}".format(abspath,defname))

def advancedRecord(filename,instructionlist = [],inputMethod = 0,indexes = [],isCapture = False,debug = False):
    #This also enters keys into PAMAR, but uses a list of strings as instructions for increased flexibility.
    #filename is the name of a txt file or a list of strings containing the legal numbers of the cases
    #instructionlist is a list of strings containing commands, the only special commands are $0-9, which get command data from the filename input
    #inputMethod 0 is for filename as name of text file, 1 is for filename as list of strings
    #indexes is a list of indexes if you don't want to run the commands on all of filename
    #debug runs the program more slowly so it can be observed if True
    #No outputs, we get data from this using the AccuTerm frame's text file writer
    keyboard.wait("shift") #Pause to allow the user to change window focus into PAMAR
    keyboard.release("shift")
    #Read filename and turn it into a list of strings for the function
    if inputMethod == 0:
        inputData = open("inputData/" + filename,"r").readlines()
    elif inputMethod == 1:
        inputData = filename
    for x in range(len(inputData)):
        #For each case:
        if len(indexes) == 0 or x in indexes:
            entry = inputData[x]
            #Run the command
            for command in instructionlist:
                if len(command) > 0:
                    if command[0] == "$":
                        keyboard.write(entry.split("|")[int(command[1:])])
                        print(entry.split("|")[int(command[1:])])
                    else:
                        keyboard.write(command.replace("~",""))
                        print(command)
                else: keyboard.write(command)
                if debug:
                    time.sleep(1)
                else:
                    time.sleep(0.2)
                if "~" not in command: keyboard.send("enter")
            #keyboard.wait("shift")
            #keyboard.release("shift")
    if isCapture:
        filename = input("Enter the name of the file to save:")
        shutil.move(r"C:\Users\nathanielslivke\Documents\capture.txt","{}/inputData/{}".format(abspath,filename))

def enterMailnos(inFile,postdate,debug = False):
    #Wrapper calling advancedRecord with a specific command list for a specific task
    maildata = [a for a in open("inputData/{}".format(inFile),"r").read().splitlines() if a != ""]
    inputdata = ["|".join(maildata[a:a+3]) for a in range(0,len(maildata),3)]    
    advancedRecord(inputdata,instructionlist = ["xx","2","$0","n","a","~poe ","$1","~db ","$2","","e","y","t","16",postdate,"writ","","e","y"],inputMethod = 1,debug = debug)

def reRecordOrds(inData,savename,debug = False):
    #Alternate record concept
    #At the moment this is a single special case; it uses a dictionary of a parsed run of oldQuickRecord to run the order sequence without additional input
    #inData is a list of dictionaries outputted by the parser in DataProcessor2
    #debug makes the program run more slowly so it can be observed if True
    #No outputs, we get data from this using the AccuTerm frame's text file writer
    keyboard.wait("shift") #Pause to allow the user to change window focus to PAMAR
    keyboard.release("shift")
    #For each case:
    for case in inData:
        commandlist = [case["Legal"],"i","e","g",case["garns"],"e","","","e","n","s","poe","","s","db","","s","governor","p","s","writ","","","","","","",
                       "","","","","","","","","","","","","","","e","e"]
        for command in commandlist:
            #Run each command
            keyboard.write(command)
            if debug: time.sleep(0.5)
            else: time.sleep(0.1)
            keyboard.send("enter")
    input("Press enter to save file:")
    shutil.move(r"C:\Users\nathanielslivke\Documents\capture.txt","{}/inputData/{}".format(abspath,savename))


def writeFromFile(inFile,mod=1,fileIndex = 0):
    #Opens a text file and writes data from it to the keyboard module
    #Originally a helper function for a specific note entry task, currently not in use
    readfile = open("inputData/{}".format(inFile)).read().splitlines()
    keyboard.wait("shift")
    keyboard.release("shift")
    for x in range (len(readfile)):
        if x%mod == 0:
            keyboard.write(readfile[x].split(" ")[fileIndex])
            keyboard.wait("shift")
            keyboard.release("shift")