from keras.datasets import mnist
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
    start = timeit.default_timer()
    for i in np.unique(y):
        classIndexes = np.nonzero(y == i)[0]
        classes[i] = [X[j].reshape(imageW*imageH) for j in classIndexes]
    binTime = timeit.default_timer() - start
    return classes, binTime


def clusterDataSet(classes, numTemplates):
    clusterLabels = []
    clusterCenters = []
    start = timeit.default_timer()
    kmeans = KMeans(n_clusters=numTemplates, random_state=0)
    for label, datapoints in classes.items():
        kmeans.fit(datapoints)
        clusterLabels.extend([label] * numTemplates)
        clusterCenters.extend(kmeans.cluster_centers_)
    clusterTime = timeit.default_timer() - start
    return clusterCenters, clusterLabels, clusterTime

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


if __name__ == '__main__':
    
    #Load the MNIST dataset
    (trainX_orig, trainy_orig), (testX_orig, testy_orig) = mnist.load_data()
    numTrainingSamples, imageW, imageH = trainX_orig.shape
    numTestingSamples = len(testX_orig)
    
    trainChunkIndex = 0
    testChunkIndex = 0
    numPlotImages = 9
    numTemplates = 64

    # Change to np.array and type to int/float
    trainX_orig = np.asarray(trainX_orig).astype(float)
    trainy_orig = np.asarray(trainy_orig).astype(int)
    testX_orig = np.asarray(testX_orig).astype(float)
    testy_orig = np.asarray(testy_orig).astype(int)
    

    ans = '-1'
    while ans != '0':
        print("""
        SELECT TASK:
        0. QUIT
        1. Task 1: kNearestNeighbour without clustering
        2. Task 2: kNearestNeighbour with kmeans clustering
        3. Comparison of Task 1 and Task 2
        4. kNN, K = 7
        """)
        ans = input("Select a task: ")

        #Adjustable parameters
        trainingSampleSize = int(input("Training sample size: ")) #number between 0 and 60000
        testingSampleSize = int(input("Testing sample size: ")) #number between 0 and 10000
        numNeighbors = int(input("KNN, K = "))

        assert len(trainX_orig) == len(trainy_orig)
        assert len(testX_orig) == len(testy_orig)
        assert trainingSampleSize <= numTrainingSamples
        assert testingSampleSize <= numTestingSamples
        
        numTrainChunks = numTrainingSamples//trainingSampleSize
        numTestChunks = numTestingSamples//testingSampleSize

        #Need to preprocess the data by chunking it into sizes of chunkSize datapoints in each chunk
        trainChunks, testChunks = reshapeAndChunk(trainX_orig, trainy_orig, testX_orig, testy_orig, numTrainChunks, numTestChunks)
        (trainX, trainy), (testX, testy) = trainChunks[trainChunkIndex], testChunks[testChunkIndex]

        if ans == '1' or ans == '2':
            if ans == '2':
                classes, binTime = binByLabel(trainX, trainy, imageW, imageH)     
                trainX, trainy, clusterTime = clusterDataSet(classes, numTemplates)
                print("Time of sorting data into classes: ", binTime, " seconds")
                print("Time of clustering data into : ", numTemplates, " templates: ", clusterTime," seconds")
            
            #Run the kNearestNeighbor algorithm with euclidean distance
            predy, calcTime, predTime = knnEuclideanClassifier(trainX, trainy, testX, numNeighbors)
            
            print("Time of fitting data: ", calcTime, " seconds")
            print("Time of prediciting data: ", predTime, " seconds")
            
            #Creating and plotting the confusion matrix
            createAndPlotConfusionMatrix(testy, predy)
            
            #Creating and printing the error rate
            accRate = metrics.accuracy_score(testy, predy)
            print("Accuracy Rate of the classification: ", accRate)
            print("Error Rate of the classification: ", 1-accRate)
            
            #Identify and plot some of the misclassifications
            misclassifiedTrue, misclassifiedFalse = plotClassifications(testX, testy, predy, imageW, imageH, numPlotImages, False)   
            print("Actual values of misclassified digits: ", misclassifiedTrue)
            print("Predicted values of misclassified digits: ", misclassifiedFalse)    
            plt.show()
            
            #Identify and plot some of the correct classifications
            classifiedTrue, _ = plotClassifications(testX, testy, predy, imageW, imageH, numPlotImages, True)
            print("Correct classifications: ", classifiedTrue)
            plt.show()
            
        if ans == '3':
            classes, binTime = binByLabel(trainX, trainy, imageW, imageH)     
            trainX_C, trainy_C, clusterTime = clusterDataSet(classes, numTemplates)
            print("Time of sorting data into classes: ", binTime, " seconds")
            print("Time of clustering data into : ", numTemplates, " templates: ", clusterTime," seconds")
            
            predy, calcTime, predTime = knnEuclideanClassifier(trainX, trainy, testX, numNeighbors)
            print("Time of fitting data: ", calcTime, " seconds")
            print("Time of prediciting data: ", predTime, " seconds")        
            
            #Run the kNearestNeighbor algorithm with euclidean distance
            predyC, calcTimeC, predTimeC = knnEuclideanClassifier(trainX_C, trainy_C, testX, numNeighbors)
            print("Time of fitting clustered data: ", calcTimeC, " seconds")
            print("Time of prediciting data when clustered: ", predTimeC, " seconds")
            
            #Creating and printing the error rate
            accRate = metrics.accuracy_score(testy, predy)
            accRateC = metrics.accuracy_score(testy, predyC)
            print("Accuracy Rate of the classification without clustering: ", accRate)
            print("Error Rate of the classification without clustering: ", 1-accRate)
            print("Accuracy Rate of the classification with clustering: ", accRateC)
            print("Error Rate of the classification with clustering: ", 1-accRateC)
        if ans == '4':
            classes, binTime = binByLabel(trainX, trainy, imageW, imageH)     
            trainX_C, trainy_C, clusterTime = clusterDataSet(classes, numTemplates)
            print("Time of sorting data into classes: ", binTime, " seconds")
            print("Time of clustering data into : ", numTemplates, " templates: ", clusterTime," seconds")
            
            #Run the kNearestNeighbor algorithm with euclidean distance
            predyC, calcTimeC, predTimeC = knnEuclideanClassifier(trainX_C, trainy_C, testX, numNeighbors)
            print("Time of fitting clustered data: ", calcTimeC, " seconds")
            print("Time of prediciting data when clustered: ", predTimeC, " seconds")
            
            #Creating and plotting the confusion matrix
            createAndPlotConfusionMatrix(testy, predyC)
                        
            #Creating and printing the error rate
            accRateC = metrics.accuracy_score(testy, predyC)
            print("Accuracy Rate of the classification with clustering: ", accRateC)
            print("Error Rate of the classification with clustering: ", 1-accRateC)