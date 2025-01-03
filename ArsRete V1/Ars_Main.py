#ArsRete Main File
#Project Information:
#ArsRete is a learning project that implements a simple MLP (multi-layer perceptron) neural network for classifying data.
#The network can use a range of different activation and cost functions in order to examine the varying capabilities of the functions.
#This project also includes tools for generating labelled data to run the network on for testing purposes.

#__IMPORTS__
import Ars_Network
import Ars_Image
import Ars_Data
import Ars_Gsys

#__GLOBAL VARIABLES__
image_charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()-+=|\\{}[]:;\'\"<>?/"

def main():
    #Main function of the project
    #Set up to be run with manual function calls from within an IDE at the moment
    #At present circles will only sort of work (accuracy values won't be outputted correctly), but the other two should work.
    #__INITIALIZE TEST DATASETS__
    #For image_data the input should be size 192 (generated test images are 9x16 pixels) and the output should be size 89 (the length of image_charset)
    image_data = Ars_Image.loadDataset(char_set = image_charset)
    #For word_data the input should be size 14 (the length of the test words in the datasets) and the output should be size 2 (real or fake)
    word_data_list = [Ars_Data.getWordData(a) for b in [["{}test{}".format(d,c) for c in ["A","B","C","D","E","F"]] for d in ["100","500","1000"]] for a in b]
    #...Complicated list comprehension way of writing out "100testA" through "1000testF", which are the current
    
    #__CREATE A NETWORK__
    #Two hidden layers, sized for word data, using tanh activations and sigmoid on the final layer, with cross-entropy cost function
    sample_words_network = Ars_Network.Network([14,45,35,2],modes = [1,1,0])
    #Three hidden layers, sized for image data, using sigmoid activations, with mean squared error cost function
    sample_images_network = Ars_Network.Network([192,93,75,53,89],modes = [0,0,0])
    
    #__TRAIN A NETWORK__
    #Train sample_words_network on sample set 100testA, for 100 loops at a learning rate of 0.3, in constant learning rate mode, evaluate and visualize, and use 100testB for the evaluate step
    sample_words_network.learn(word_data_list[0],100,0.3,0,"",True,True,word_data_list[1])
    #Train sample_images_network on the first 1000 values in image_data, for 1000 loops at a learning rate of 0.5, in RMSProp learning rate mode, evaluate and visualize, and use the initial training data for the evaluate step
    sample_images_network.learn(image_data[:1000],1000,0.5,-1,"",True,True)
    
    #__SAVE A NETWORK__
    #Save sample_words_network to file
    sample_words_network.saveToFile("111923_test_network")
    #__LOAD A NETWORK__
    #Make a new network, and load its parameters from 111923_test_network
    new_sample_network = ArsNetwork.Network([1,1,1])
    new_sample_network.loadFromFile("111923_test_network")
    
    
main()