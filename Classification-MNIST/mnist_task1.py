from turtle import distance
from keras.datasets import mnist
import numpy as np
from matplotlib import pyplot
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from scipy.spatial import distance

def prepareTrainingData(trainX, trainy, testX, testy, chunkSize):
    trainingSetSize = trainX.shape[0]
    imageSizeX = trainX.shape[1]
    imageSizeY = trainX.shape[2]
    numChunks = trainingSetSize // chunkSize
    
    trainX = np.array_split(trainX, numChunks)
    trainy = np.array_split(trainy, numChunks)
    testX = np.array_split(testX, numChunks)
    testy = np.array_split(testy, numChunks)

    chunks = []
    for i in range(0, len(trainX)):
        chunks.append([(trainX[i], trainy[i]), (testX[i], testy[i])])
    return chunks
    

def knnEuclideanClassifier(trainX, trainy, testX, numNeighbors):
    print(str(trainX.shape))
    print(str(testX.shape))


    trainX = trainX.reshape(len(trainX), len(trainX[0])*len(trainX[0][0]))
    testX = testX.reshape(len(testX), len(testX[0])*len(testX[0][0]))
    neigh = KNeighborsClassifier(n_neighbors=numNeighbors, metric = distance.euclidean)
    neigh.fit(trainX, trainy)
    print(str(trainX.shape))
    print(str(trainX[0].shape))
    print(str(testX.shape))
    print(str(testX[0].shape))
    return neigh.predict(testX)
    
    

def main():
    trainingSampleSize = 5000
    testingSampleSize = 100
    chunkSize = 1000
    trainingIndex = 0 #number between 0 and (60000/chunksize - 1)
 
    #Load the MNIST dataset
    (trainX, trainy), (testX, testy) = mnist.load_data()
    
    #Need to preprocess the data by chunking it into sizes of chunkSize datapoints in each chunk
    chunks = prepareTrainingData(trainX, trainy, testX, testy, chunkSize)
    (trainX, trainy), (testX, testy) = chunks[trainingIndex]
    
    numNeighbors = 3
    
    labelPredictions = knnEuclideanClassifier(trainX, trainy, testX, numNeighbors)
    
    print(testy)
    print(labelPredictions)
    
    #Run the kNearestNeighbor algorithm with euclidean distance
    
    
    
    
    
    
    
    
    
    pyplot.imshow(trainX[0], cmap=pyplot.get_cmap('gray'))
    pyplot.show()
    

    pyplot.show()

main()