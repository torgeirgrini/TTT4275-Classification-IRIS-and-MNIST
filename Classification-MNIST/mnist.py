from keras.datasets import mnist
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
import utilities as utils
import sys

if __name__ == '__main__':
    task = sys.argv[1]

    #Load the MNIST dataset
    (trainX, trainy), (testX, testy) = mnist.load_data()
    numTrainingSamples, imageW, imageH = trainX.shape
    numTestingSamples = len(testX)
    
    #Adjustable parameters
    trainingSampleSize = 1000 #number between 0 and 60000
    testingSampleSize = 500 #number between 0 and 10000
    trainChunkIndex = 0
    testChunkIndex = 0
    numNeighbors = 3
    numPlotImages = 9
    numTemplates = 64

    assert len(trainX) == len(trainy)
    assert len(testX) == len(testy)
    assert trainingSampleSize <= numTrainingSamples
    assert testingSampleSize <= numTestingSamples
    
    numTrainChunks = numTrainingSamples//trainingSampleSize
    numTestChunks = numTestingSamples//testingSampleSize

    # Change to np.array and type to int/float
    trainX = np.asarray(trainX).astype(float)
    trainy = np.asarray(trainy).astype(int)
    testX = np.asarray(testX).astype(float)
    testy = np.asarray(testy).astype(int)
    
    #Need to preprocess the data by chunking it into sizes of chunkSize datapoints in each chunk
    trainChunks, testChunks = utils.reshapeAndChunk(trainX, trainy, testX, testy, numTrainChunks, numTestChunks)
    (trainX, trainy), (testX, testy) = trainChunks[trainChunkIndex], testChunks[testChunkIndex]

    print("############")
    print("## " + task + " ##")
    print("############")
    if task == 'TASK1' or task == 'TASK2':
        if task == 'TASK2':
            classes = utils.binByLabel(trainX, trainy, imageW, imageH)     
            trainX, trainy = utils.clusterDataSet(classes, numTemplates)
            print("Time of sorting data into classes: ", 0, " seconds")
            print("Time of clustering data into : ", numTemplates, " templates: ", 0," seconds")
        
        #Run the kNearestNeighbor algorithm with euclidean distance
        predy, calcTime, predTime = utils.knnEuclideanClassifier(trainX, trainy, testX, numNeighbors)
        
        print("Time of fitting data: ", calcTime, " seconds")
        print("Time of prediciting data: ", predTime, " seconds")
        
        #Creating and plotting the confusion matrix
        utils.createAndPlotConfusionMatrix(testy, predy)
        
        #Creating and printing the error rate
        accRate = metrics.accuracy_score(testy, predy)
        print("Accuracy Rate of the classification: ", accRate)
        print("Error Rate of the classification: ", 1-accRate)
        
        #Identify and plot some of the misclassifications
        misclassifiedTrue, misclassifiedFalse = utils.plotClassifications(testX, testy, predy, imageW, imageH, numPlotImages, False)   
        print("Actual values of misclassified digits: ", misclassifiedTrue)
        print("Predicted values of misclassified digits: ", misclassifiedFalse)    
        plt.show()
        
        #Identify and plot some of the correct classifications
        classifiedTrue, _ = utils.plotClassifications(testX, testy, predy, imageW, imageH, numPlotImages, True)
        print("Correct classifications: ", classifiedTrue)
        plt.show()
        
    if task == 'COMPARE':
        classes = utils.binByLabel(trainX, trainy, imageW, imageH)     
        trainX_C, trainy_C = utils.clusterDataSet(classes, numTemplates)
        print("Time of sorting data into classes: ", 0, " seconds")
        print("Time of clustering data into ", numTemplates, " templates: ", 0," seconds")
        
        predy, calcTime, predTime = utils.knnEuclideanClassifier(trainX, trainy, testX, numNeighbors)
        print("Time of fitting data: ", calcTime, " seconds")
        print("Time of prediciting data: ", predTime, " seconds")        
        
        #Run the kNearestNeighbor algorithm with euclidean distance
        predyC, calcTimeC, predTimeC = utils.knnEuclideanClassifier(trainX_C, trainy_C, testX, numNeighbors)
        print("Time of fitting clustered data: ", calcTimeC, " seconds")
        print("Time of prediciting data when clustered: ", predTimeC, " seconds")
        
        #Creating and printing the error rate
        accRate = metrics.accuracy_score(testy, predy)
        accRateC = metrics.accuracy_score(testy, predyC)
        print("Accuracy Rate of the classification: ", accRate)
        print("Error Rate of the classification: ", 1-accRate)
        print("Accuracy Rate of the classification w/ clustering: ", accRateC)
        print("Error Rate of the classification w/ clustering: ", 1-accRateC)