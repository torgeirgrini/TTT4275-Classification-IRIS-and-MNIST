from keras.datasets import mnist
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import timeit

def reshapeDataSet(trainX, trainy, testX, testy, numTrainChunks, numTestChunks): 
    numTrainingSamples, imageSizeX, imageSizeY = trainX.shape
    numTestSamples = len(testX)
    
    trainX = trainX.reshape(numTrainingSamples, imageSizeX*imageSizeY)
    testX = testX.reshape(numTestSamples, imageSizeX*imageSizeY)
    
    trainX = np.array_split(trainX, numTrainChunks)
    trainy = np.array_split(trainy, numTrainChunks)
    testX = np.array_split(testX, numTestChunks)
    testy = np.array_split(testy, numTestChunks)

    trainingChunks, testingChunks = [], []
    for i in range(numTrainChunks):
        trainingChunks.append((trainX[i], trainy[i]))
    for j in range(numTestChunks):
        testingChunks.append((testX[j], testy[j]))
        
    return trainingChunks, testingChunks
    

def knnEuclideanClassifier(trainX, trainy, testX, numNeighbors):
    neigh = KNeighborsClassifier(n_neighbors=numNeighbors) #euclidean distance by default
    start = timeit.default_timer()
    neigh.fit(trainX, trainy)
    calcTime = timeit.default_timer() - start
    start = timeit.default_timer()
    predy = neigh.predict(testX)
    predTime = timeit.default_timer() - start
    return predy, calcTime, predTime
    
    
if __name__ == '__main__':
    
    #Load the MNIST dataset
    (trainX, trainy), (testX, testy) = mnist.load_data()
    numTrainingSamples, imageW, imageH = trainX.shape
    numTestingSamples = len(testX)
    
    #Adjustable parameters
    trainingSampleSize = 60000 #number between 0 and 60000
    testingSampleSize = 10000 #number between 0 and 10000
    trainChunkIndex = 0
    testChunkIndex = 0
    numNeighbors = 3
    numPlotImages = 9

    assert len(trainX) == len(trainy)
    assert len(testX) == len(testy)
    assert trainingSampleSize <= numTrainingSamples
    assert testingSampleSize <= numTestingSamples
    
    numTrainChunks = numTrainingSamples//trainingSampleSize
    numTestChunks = numTrainChunks
    
    #Need to preprocess the data by chunking it into sizes of chunkSize datapoints in each chunk
    trainChunks, testChunks = reshapeDataSet(trainX, trainy, testX, testy, numTrainChunks, numTestChunks)
    (trainX, trainy), (testX, testy) = trainChunks[trainChunkIndex], testChunks[testChunkIndex]
        
    #Run the kNearestNeighbor algorithm with euclidean distance
    predy, calcTime, predTime = knnEuclideanClassifier(trainX, trainy, testX, numNeighbors)
    
    print("Time of fitting data: ", calcTime, " seconds")
    print("Time of prediciting data: ", predTime, " seconds")
    
    #Creating and plotting the confusion matrix
    confusionMatrix = metrics.confusion_matrix(testy, predy)
    dispConfusionMatrix = metrics.ConfusionMatrixDisplay(confusionMatrix)
    dispConfusionMatrix.plot()
    plt.show()
    #Creating and plotting the error rate
    errorRate = metrics.accuracy_score(testy, predy)
    print("Error Rate of the classification: ", errorRate)
    
    #Identify and plot some of the misclassifications
    misclassified = np.where(testy != predy)[0]
    misclassifiedTrue, misclassifiedFalse = [],[]
    for i in range(min(len(misclassified), numPlotImages)):  
        misclassifiedImage = testX[misclassified[i]]
        misclassifiedImage = misclassifiedImage.reshape(imageW,imageH)
        plt.subplot(330 + 1 + i)
        plt.imshow(misclassifiedImage, cmap=plt.get_cmap('gray'))
        misclassifiedTrue.append(testy[misclassified[i]])
        misclassifiedFalse.append(predy[misclassified[i]])
    
    print("Actual values of misclassified digits: ", misclassifiedTrue)
    print("Predicted values of misclassified digits: ", misclassifiedFalse)    
    plt.show()
    
    #Identify and plot some of the correct classifications
    correctClassified = np.where(testy == predy)[0]
    classifiedTrue = []
    for i in range(min(len(correctClassified), numPlotImages)):  
        classifiedImage = testX[correctClassified[i]]
        classifiedImage = classifiedImage.reshape(imageW,imageH)
        plt.subplot(330 + 1 + i)
        plt.imshow(classifiedImage, cmap=plt.get_cmap('gray'))
        classifiedTrue.append(predy[correctClassified[i]])
        
    print("Correct classifications: ", classifiedTrue)
    plt.show()
    