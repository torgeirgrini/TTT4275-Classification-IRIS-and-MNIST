import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn import metrics
import timeit

def reshapeAndChunk(trainX, trainy, testX, testy, numTrainChunks, numTestChunks): 
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
    neigh = KNeighborsClassifier(n_neighbors=numNeighbors, metric = 'euclidean')
    start = timeit.default_timer()
    neigh.fit(trainX, trainy)
    calcTime = timeit.default_timer() - start
    start = timeit.default_timer()
    predy = neigh.predict(testX)
    predTime = timeit.default_timer() - start
    return predy, calcTime, predTime


def binByLabel(X, y, imageW, imageH):
    classes = dict()
    for i in np.unique(y):
        classIndexes = np.nonzero(y == i)[0]
        classes[i] = [X[j].reshape(imageW*imageH) for j in classIndexes]
    return classes


def clusterDataSet(classes, numTemplates):
    clusterLabels = []
    clusterCenters = []
    kmeans = KMeans(n_clusters=numTemplates, random_state=0)
    for label, datapoints in classes.items():
        kmeans.fit(datapoints)
        clusterLabels.extend([label] * numTemplates)
        clusterCenters.extend(kmeans.cluster_centers_)
        
    return clusterCenters, clusterLabels

def createAndPlotConfusionMatrix(testy, predy):
    confusionMatrix = metrics.confusion_matrix(testy, predy)
    dispConfusionMatrix = metrics.ConfusionMatrixDisplay(confusionMatrix)
    dispConfusionMatrix.plot()
    plt.show()
    
def plotClassifications(testX, testy, predy, imageW, imageH, numPlotImages, correctClassifications):
    if correctClassifications:
        classifiedIndex = np.where(testy == predy)[0]
    else: 
        classifiedIndex = np.where(testy != predy)[0]
    classifiedTrue, classifiedFalse = [],[]
    for i in range(min(len(classifiedIndex), numPlotImages)):  
        classifiedImage = testX[classifiedIndex[i]]
        classifiedImage = classifiedImage.reshape(imageW,imageH)
        plt.subplot(330 + 1 + i)
        plt.imshow(classifiedImage, cmap=plt.get_cmap('gray'))
        classifiedTrue.append(testy[classifiedIndex[i]])
        classifiedFalse.append(predy[classifiedIndex[i]])
    return classifiedTrue, classifiedFalse